import os
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
import pdb
import torch
from torch import nn, Tensor
import torch.nn.functional as F
try:
    from torch_geometric.nn import GATConv
except ImportError:
    print("Warning: torch_geometric not installed. GAT layers will not work.")
    GATConv = None


def build_grid_edge_index(P, P_w, device):
    """
    Build directed grid adjacency for a P x P_w grid.
    Returns edge_index of shape [2, E].
    """
    edges = []
    def idx(i, j):
        return i * P_w + j

    for i in range(P):
        for j in range(P_w):
            u = idx(i, j)
            # 4-neighborhood (up/down/left/right)
            if i - 1 >= 0:
                v = idx(i - 1, j)
                edges.append([u, v])
            if i + 1 < P:
                v = idx(i + 1, j)
                edges.append([u, v])
            if j - 1 >= 0:
                v = idx(i, j - 1)
                edges.append([u, v])
            if j + 1 < P_w:
                v = idx(i, j + 1)
                edges.append([u, v])
            # Always add self-loop
            edges.append([u, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index.to(device)


class Convblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Convblock, self).__init__()
        self.encoder = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.ebn = nn.BatchNorm2d(output_channels)
    def forward(self, x):
        out = F.relu(F.max_pool2d(self.ebn(self.encoder(x)), 2, 2))
        return out


class Convblock1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Convblock1, self).__init__()
        self.encoder = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.ebn = nn.BatchNorm2d(output_channels)
    def forward(self, x):
        out = F.relu(self.ebn(self.encoder(x)))
        return out


class Upblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Upblock, self).__init__()
        self.decoder = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.dbn = nn.BatchNorm2d(output_channels)
    def forward(self, x):
        out = F.relu(F.interpolate(self.dbn(self.decoder(x)), scale_factor=(2, 2), mode='bilinear', align_corners=False))
        return out


class GATBlock(nn.Module):
    """
    Simplified GAT block that operates at current spatial resolution
    without internal pooling/upsampling. Can be used as a replacement
    for intermediate conv layers in multi-scale processing.
    """
    def __init__(self, input_channels, output_channels, heads=4):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.heads = heads
        self.reduced_ch = max(1, input_channels // 2)
        
        # Channel reduction
        self.reduce = nn.Conv2d(input_channels, self.reduced_ch, kernel_size=1, bias=False)
        
        # GAT layer
        if GATConv is not None:
            self.gat = GATConv(
                in_channels=self.reduced_ch,
                out_channels=self.reduced_ch,
                heads=heads,
                concat=False
            )
        
        # Restore channels
        self.restore = nn.Conv2d(self.reduced_ch, output_channels, kernel_size=1, bias=False)
        
        # Residual
        if input_channels != output_channels:
            self.residual = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()
        
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward_gat(self, x_reduced):
        """Apply GAT at current spatial resolution (always, even for small grids)"""
        B, C, H, W = x_reduced.shape
        device = x_reduced.device
        edge_index = build_grid_edge_index(H, W, device)
        out_nodes = []
        for b in range(B):
            nodes = x_reduced[b].permute(1, 2, 0).reshape(H * W, C)
            nodes_out = self.gat(nodes, edge_index)
            nodes_out = nodes_out.view(H, W, C).permute(2, 0, 1)
            out_nodes.append(nodes_out)
        out = torch.stack(out_nodes, dim=0)
        return out
    
    def forward(self, x):
        # Residual connection
        res = self.residual(x)
        
        # Reduce channels
        xr = self.reduce(x)
        
        # Apply GAT at current spatial resolution
        if GATConv is not None:
            out = self.forward_gat(xr)
        else:
            # Fallback: just use reduced features
            out = xr
        
        # Restore channels
        out = self.restore(out)
        
        # Add residual and activate
        out = out + res
        out = self.bn(out)
        out = self.act(out)
        
        return out

class GAT4(nn.Module):
    """
    GAT-based skip connection with 4-level processing (similar to SMM4)
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.filters = [input_channels, input_channels//2, input_channels//4, input_channels//8]
        
        self.Convstage1 = Convblock(input_channels, self.filters[1])
        self.Convstage2 = Convblock(self.filters[1], self.filters[2])
        self.Convstage3 = Convblock(self.filters[2], self.filters[3])
        self.Convstage4 = Convblock(self.filters[3], self.filters[3])
        self.GATstage = GATBlock(self.filters[3], self.filters[3], heads=4)
        
        self.Upstage4 = Upblock(self.filters[3], self.filters[3])
        self.Upstage3 = Upblock(self.filters[3], self.filters[2])
        self.Upstage2 = Upblock(self.filters[2], self.filters[1])
        self.Upstage1 = Upblock(self.filters[1], self.filters[0])
    
    def forward(self, x):
        ### Stage 1
        out = self.Convstage1(x)
        t1 = out
        
        ### Stage 2
        out = self.Convstage2(out)
        t2 = out
        
        ### Stage 3
        out = self.Convstage3(out)
        t3 = out
        
        ### Stage 4 (with GAT)
        out = self.Convstage4(out)
        out = self.GATstage(out)
        
        ### Stage 3
        out = self.Upstage4(out)
        if out.shape[-2:] != t3.shape[-2:]:
            t3 = F.interpolate(t3, size=out.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.add(out, t3)
        
        ### Stage 2
        out = self.Upstage3(out)
        if out.shape[-2:] != t2.shape[-2:]:
            t2 = F.interpolate(t2, size=out.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.add(out, t2)
        
        ### Stage 1
        out = self.Upstage2(out)
        if out.shape[-2:] != t1.shape[-2:]:
            t1 = F.interpolate(t1, size=out.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.add(out, t1)
        
        ### Stage 0
        out = self.Upstage1(out)
        return out


class GAT3(nn.Module):
    """
    GAT-based skip connection with 3-level processing (similar to SMM3)
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.filters = [input_channels, input_channels//2, input_channels//4]
        
        self.Convstage1 = Convblock(input_channels, self.filters[1])
        self.Convstage2 = Convblock(self.filters[1], self.filters[2])
        self.Convstage3 = Convblock(self.filters[2], self.filters[2])
        self.GATstage = GATBlock(self.filters[2], self.filters[2], heads=4)
        
        self.Upstage3 = Upblock(self.filters[2], self.filters[2])
        self.Upstage2 = Upblock(self.filters[2], self.filters[1])
        self.Upstage1 = Upblock(self.filters[1], self.filters[0])
    
    def forward(self, x):
        ### Stage 1
        out = self.Convstage1(x)
        t1 = out
        
        ### Stage 2
        out = self.Convstage2(out)
        t2 = out
        
        ### Stage 3 (with GAT)
        out = self.Convstage3(out)
        out = self.GATstage(out)
        
        ### Stage 2
        out = self.Upstage3(out)
        if out.shape[-2:] != t2.shape[-2:]:
            t2 = F.interpolate(t2, size=out.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.add(out, t2)
        
        ### Stage 1
        out = self.Upstage2(out)
        if out.shape[-2:] != t1.shape[-2:]:
            t1 = F.interpolate(t1, size=out.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.add(out, t1)
        
        ### Stage 0
        out = self.Upstage1(out)
        return out


class GAT2(nn.Module):
    """
    GAT-based skip connection with 2-level processing (similar to SMM2)
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.filters = [input_channels, input_channels//2]
        
        self.Convstage1 = Convblock(input_channels, self.filters[1])
        self.Convstage2 = Convblock(self.filters[1], self.filters[1])
        self.GATstage = GATBlock(self.filters[1], self.filters[1], heads=4)
        
        self.Upstage2 = Upblock(self.filters[1], self.filters[1])
        self.Upstage1 = Upblock(self.filters[1], self.filters[0])
    
    def forward(self, x):
        ### Stage 1
        out = self.Convstage1(x)
        t1 = out
        
        ### Stage 2 (with GAT)
        out = self.Convstage2(out)
        out = self.GATstage(out)
        
        ### Stage 1
        out = self.Upstage2(out)
        if out.shape[-2:] != t1.shape[-2:]:
            t1 = F.interpolate(t1, size=out.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.add(out, t1)
        
        ### Stage 0
        out = self.Upstage1(out)
        return out


class GAT1(nn.Module):
    """
    GAT-based skip connection with 1-level processing (similar to SMM1)
    Simple pass-through with optional GAT attention
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.Convstage1 = Convblock1(input_channels, input_channels)
    
    def forward(self, x):
        out = self.Convstage1(x)
        return out


class GATBottleneck(nn.Module):
    """
    SHALLOW GAT bottleneck module for the deepest layer of the network.
    Applies graph attention at the smallest spatial resolution (e.g., 12x12)
    with the highest channel count (e.g., 1024 channels).
    """
    def __init__(self, input_channels, output_channels, heads=8, reduction=2):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.heads = heads
        
        # Channel reduction for efficiency (1024 -> 512 for GAT processing)
        self.reduced_ch = max(1, input_channels // reduction)
        
        # Channel reduction
        self.reduce = nn.Conv2d(input_channels, self.reduced_ch, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(self.reduced_ch)
        
        # GAT layer with more heads for bottleneck (no dropout for stability)
        if GATConv is not None:
            self.gat = GATConv(
                in_channels=self.reduced_ch,
                out_channels=self.reduced_ch,
                heads=heads,
                concat=False
            )
        
        # Restore channels
        self.restore = nn.Conv2d(self.reduced_ch, output_channels, kernel_size=1, bias=False)
        self.bn_restore = nn.BatchNorm2d(output_channels)
        
        # Residual connection
        if input_channels != output_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.residual = nn.Identity()
        
        self.act = nn.ReLU(inplace=True)
    
    def forward_gat(self, x_reduced):
        """Apply GAT on the bottleneck features"""
        B, C, H, W = x_reduced.shape
        device = x_reduced.device
        
        # Build grid graph
        edge_index = build_grid_edge_index(H, W, device)
        
        # Process each sample in batch
        out_nodes = []
        for b in range(B):
            # Reshape to (num_nodes, channels)
            nodes = x_reduced[b].permute(1, 2, 0).reshape(H * W, C)
            
            # Apply GAT
            nodes_out = self.gat(nodes, edge_index)
            
            # Reshape back to (C, H, W)
            nodes_out = nodes_out.view(H, W, C).permute(2, 0, 1)
            out_nodes.append(nodes_out)
        
        out = torch.stack(out_nodes, dim=0)
        return out
    
    def forward(self, x):
        # Residual connection
        res = self.residual(x)
        
        # Reduce channels
        xr = self.reduce(x)
        xr = self.bn_reduce(xr)
        xr = self.act(xr)
        
        # Apply GAT at bottleneck resolution
        if GATConv is not None:
            out = self.forward_gat(xr)
        else:
            # Fallback: just use reduced features
            out = xr
        
        # Restore channels
        out = self.restore(out)
        out = self.bn_restore(out)
        
        # Add residual and activate
        out = out + res
        out = self.act(out)
        
        return out


class DeepGATBottleneck(nn.Module):
    """
    DEEP multi-level GAT bottleneck for the deepest layer of the network.
    Similar to GAT4 but designed specifically for bottleneck processing.
    Applies multiple GAT layers at different spatial resolutions.
    
    Architecture:
    Input (1024ch, 12×12) → Down (512ch, 6×6) → Down (256ch, 3×3) → GAT
                         ↑ Skip + GAT ← Up ← Skip + GAT ← Up ← 
    Output (1024ch, 12×12)
    """
    def __init__(self, input_channels, output_channels, heads=6):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Define channel progression
        self.filters = [
            input_channels,      # 1024
            input_channels // 2,  # 512
            input_channels // 4   # 256
        ]
        
        # Encoder (Downsampling)
        self.Convstage1 = Convblock(self.filters[0], self.filters[1])  # 1024 -> 512, 12x12 -> 6x6
        self.Convstage2 = Convblock(self.filters[1], self.filters[2])  # 512 -> 256, 6x6 -> 3x3
        self.Convstage3 = Convblock1(self.filters[2], self.filters[2]) # 256 -> 256, 3x3 (no pool)
        
        # GAT at deepest level
        self.GATstage_deep = GATBlock(self.filters[2], self.filters[2], heads=heads)
        
        # Decoder (Upsampling)
        self.Upstage2 = Upblock(self.filters[2], self.filters[2])  # 256 -> 256, 3x3 -> 6x6
        self.GATstage_mid = GATBlock(self.filters[2], self.filters[2], heads=heads)
        
        self.Upstage1 = Upblock(self.filters[2], self.filters[1])  # 256 -> 512, 6x6 -> 12x12
        self.GATstage_shallow = GATBlock(self.filters[1], self.filters[1], heads=heads)
        
        # Final restoration
        self.restore = nn.Conv2d(self.filters[1], output_channels, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(output_channels)
        
        # Residual connection
        if input_channels != output_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.residual = nn.Identity()
        
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Store for residual
        res = self.residual(x)
        
        # Encoder
        out = self.Convstage1(x)  # 1024 -> 512, 12x12 -> 6x6
        skip1 = out
        
        out = self.Convstage2(out)  # 512 -> 256, 6x6 -> 3x3
        skip2 = out
        
        # Deepest level with GAT
        out = self.Convstage3(out)  # 256 -> 256, 3x3
        out = self.GATstage_deep(out)  # GAT at 3x3
        
        # Decoder with GAT at each level
        out = self.Upstage2(out)  # 256 -> 256, 3x3 -> 6x6
        if out.shape[-2:] != skip2.shape[-2:]:
            skip2 = F.interpolate(skip2, size=out.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.add(out, skip2)  # Skip connection
        out = self.GATstage_mid(out)  # GAT at 6x6
        
        out = self.Upstage1(out)  # 256 -> 512, 6x6 -> 12x12
        if out.shape[-2:] != skip1.shape[-2:]:
            skip1 = F.interpolate(skip1, size=out.shape[-2:], mode='bilinear', align_corners=False)
        out = torch.add(out, skip1)  # Skip connection
        out = self.GATstage_shallow(out)  # GAT at 12x12
        
        # Restore to original channels
        out = self.restore(out)
        out = self.bn_final(out)
        
        # Add residual and activate
        out = out + res
        out = self.act(out)
        
        return out
