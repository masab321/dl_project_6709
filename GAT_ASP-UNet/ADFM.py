#Advanced Deep Feature Module (ADFM) for Image Processing

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    Standard Attention Block:
    - Input: (B, C, H, W)
    - Operation: Flatten -> MultiHeadAttention -> Residual Add -> LayerNorm -> Reshape
    - Output: (B, C, H, W)
    """
    def __init__(self, in_channels, num_heads=4):
        super(AttentionBlock, self).__init__()
        # Ensure num_heads is valid
        if in_channels % num_heads != 0:
            num_heads = 1

        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)

        # Self-Attention (Query, Key, Value are same)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)

        # Add & Norm (Residual)
        x_out = self.norm(x_flat + attn_out)

        # Reshape back to feature map
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        return x_out

class DownsampledAttentionBlock(nn.Module):
    """
    Applies Multi-Head Attention to a spatially reduced version of the feature map.
    Uses Adaptive Pooling to ensure a consistent computational cost.
    """
    def __init__(self, in_channels, num_heads=4, target_grid_size=16):
        super(DownsampledAttentionBlock, self).__init__()
        if in_channels % num_heads != 0:
            num_heads = 1

        self.target_grid_size = target_grid_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((target_grid_size, target_grid_size))

        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Check if we actually need to downsample
        if H <= self.target_grid_size or W <= self.target_grid_size:
            x_down = x
        else:
            x_down = self.adaptive_pool(x)

        _, _, H_d, W_d = x_down.shape

        # 2. Flatten for Attention: (B, C, H_d, W_d) -> (B, H_d*W_d, C)
        x_flat = x_down.flatten(2).transpose(1, 2)

        # 3. Self-Attention
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        x_out_flat = self.norm(x_flat + attn_out)

        # 4. Reshape back
        x_out_down = x_out_flat.transpose(1, 2).reshape(B, C, H_d, W_d)

        # 5. Upsample back to original resolution if it was reduced
        if H != H_d or W != W_d:
            x_out = F.interpolate(x_out_down, size=(H, W), mode='bilinear', align_corners=False)
        else:
            x_out = x_out_down

        return x_out

class ConvLayer(nn.Module):
    """ Standard Conv-BN-ReLU Block """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(ConvLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class ASPP(nn.Module):
    """ Atrous Spatial Pyramid Pooling """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.blocks = nn.ModuleList()
        for rate in dilations:
            padding = 0 if rate == 1 else rate
            kernel = 1 if rate == 1 else 3
            self.blocks.append(ConvLayer(in_channels, out_channels, kernel, 1, padding, dilation=rate))

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvLayer(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        res = [block(x) for block in self.blocks]
        gap = self.gap(x)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(gap)
        return torch.cat(res, dim=1)

class AdvancedFeatureModule(nn.Module):
    def __init__(self, in_channels, branch_channels=32, target_grid=16):
        super(AdvancedFeatureModule, self).__init__()

        # 1. Kernel Branches (Local Features)
        self.k1_s1 = ConvLayer(in_channels, branch_channels, 1, 1, 0)
        self.k1_s2 = ConvLayer(in_channels, branch_channels, 1, 2, 0)
        self.k3_s1 = ConvLayer(in_channels, branch_channels, 3, 1, 1)
        self.k3_s2 = ConvLayer(in_channels, branch_channels, 3, 2, 1)
        self.k5_s1 = ConvLayer(in_channels, branch_channels, 5, 1, 2)
        self.k5_s2 = ConvLayer(in_channels, branch_channels, 5, 2, 2)
        self.k_dil7 = ConvLayer(in_channels, branch_channels, 3, 1, 7, dilation=7)
        self.aspp = ASPP(in_channels, branch_channels)
        self.avg_pool_branch = nn.Sequential(nn.AvgPool2d(3, 1, 1), ConvLayer(in_channels, branch_channels, 1, 1, 0))
        self.max_pool_conv = ConvLayer(in_channels, branch_channels, 1, 1, 0)

        # 2. Parallel Self-Attention (One per branch)
        self.sa_k1s1 = DownsampledAttentionBlock(branch_channels, target_grid_size=target_grid)
        self.sa_k1s2 = DownsampledAttentionBlock(branch_channels, target_grid_size=target_grid)
        self.sa_k3s1 = DownsampledAttentionBlock(branch_channels, target_grid_size=target_grid)
        self.sa_k3s2 = DownsampledAttentionBlock(branch_channels, target_grid_size=target_grid)
        self.sa_k5s1 = DownsampledAttentionBlock(branch_channels, target_grid_size=target_grid)
        self.sa_k5s2 = DownsampledAttentionBlock(branch_channels, target_grid_size=target_grid)
        self.sa_dil7 = DownsampledAttentionBlock(branch_channels, target_grid_size=target_grid)
        self.sa_aspp = DownsampledAttentionBlock(branch_channels * 5, target_grid_size=target_grid)
        self.sa_avg  = DownsampledAttentionBlock(branch_channels, target_grid_size=target_grid)
        self.sa_max  = DownsampledAttentionBlock(branch_channels, target_grid_size=target_grid)

        # 3. Global Multi-Head Attention
        # Total: 7 basic branches + ASPP (5) + Avg (1) + Max (1) = 14 * branch_channels
        total_channels = branch_channels * 14
        self.global_mha = DownsampledAttentionBlock(total_channels, num_heads=8, target_grid_size=target_grid)

        # 4. Fusion & Residual
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        identity = x
        H, W = x.shape[2:]

        # Helper: Conv -> Attention -> Upsample
        def run_b(conv, sa, x_in, stride2=False):
            feat = conv(x_in)
            feat = sa(feat)
            if stride2:
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            return feat

        # Execute Parallel Branches
        f1 = run_b(self.k1_s1, self.sa_k1s1, x)
        f2 = run_b(self.k1_s2, self.sa_k1s2, x, True)
        f3 = run_b(self.k3_s1, self.sa_k3s1, x)
        f4 = run_b(self.k3_s2, self.sa_k3s2, x, True)
        f5 = run_b(self.k5_s1, self.sa_k5s1, x)
        f6 = run_b(self.k5_s2, self.sa_k5s2, x, True)
        f7 = run_b(self.k_dil7, self.sa_dil7, x)

        f_aspp = self.sa_aspp(self.aspp(x))
        f_avg  = self.sa_avg(self.avg_pool_branch(x))

        f_max  = F.adaptive_max_pool2d(x, (H, W)) # Local max pool to keep size
        f_max  = self.sa_max(self.max_pool_conv(f_max))

        # Concatenate All
        concat = torch.cat([f1, f2, f3, f4, f5, f6, f7, f_aspp, f_avg, f_max], dim=1)

        # Global Multi-Head Attention
        global_feat = self.global_mha(concat)

        # Fusion and Final Residual
        out = self.fusion(global_feat)
        return out + identity

