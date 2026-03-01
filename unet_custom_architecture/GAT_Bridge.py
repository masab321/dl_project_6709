# graph_gat_bridge.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv as GATConv


def build_grid_edge_index(P, P_w, device):
    """
    Build directed grid adjacency for a P x P_w grid (usually square).
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
                v = idx(i - 1, j); edges.append([u, v])
            if i + 1 < P:
                v = idx(i + 1, j); edges.append([u, v])
            if j - 1 >= 0:
                v = idx(i, j - 1); edges.append([u, v])
            if j + 1 < P_w:
                v = idx(i, j + 1); edges.append([u, v])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index.to(device)


class GraphGATBridge(nn.Module):
    """
    True GAT-based bridge module:
    - reduces channels
    - pools spatially to P x P
    - builds grid adjacency (true graph)
    - applies GATConv (PyG) per batch element
    - upsamples and restores channels
    - adds residual skip
    """

    def __init__(self,
                 in_ch,
                 out_ch=None,
                 pool_size=8,
                 reduction=4,
                 heads=4,
                 concat=False):
        """
        in_ch: channels coming from encoder (e.g., 256)
        out_ch: channels to output (skip->decoder). If None, set = in_ch
        pool_size: spatial size for graph nodes (P). e.g., 8
        reduction: channel reduction factor before GAT (in_ch//reduction)
        heads: number of attention heads in GATConv
        concat: whether to concat heads in GAT (if True output dim = out_per_head * heads)
        """
        super().__init__()
        out_ch = out_ch or in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.pool_size = pool_size

        # channel reduction before graph (keeps GAT small)
        self.reduced_ch = max(1, in_ch // reduction)
        self.reduce = nn.Conv2d(in_ch, self.reduced_ch, kernel_size=1, bias=False)
        # GATConv: keep output dims = reduced_ch (concat=False) to simplify reshape
        self.gat = GATConv(in_channels=self.reduced_ch,
                           out_channels=self.reduced_ch,
                           heads=heads,
                           concat=concat)  # concat=False -> maintains reduced_ch
        # restore channels after upsample
        self.restore = nn.Conv2d(self.reduced_ch, out_ch, kernel_size=1, bias=False)

        # residual projection to match out_ch if needed
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        returns: (B, out_ch, H, W)
        """
        B, C, H, W = x.shape
        device = x.device

        # residual to add later
        res = self.residual(x)

        # 1) reduce channels
        xr = self.reduce(x)  # (B, C_r, H, W)

        # 2) pool to small grid (pool_size x pool_size)
        # use adaptive pooling so it works for variable H,W
        x_small = F.adaptive_avg_pool2d(xr, (self.pool_size, self.pool_size))  # (B, C_r, P, P)
        P_h, P_w = x_small.shape[-2], x_small.shape[-1]

        # build adjacency once for this small grid
        edge_index = build_grid_edge_index(P_h, P_w, device)

        # 3) per-batch GAT (PyG expects node-feature matrix shape [N_nodes, C])
        out_nodes = []
        for b in range(B):
            nodes = x_small[b].permute(1, 2, 0).reshape(P_h * P_w, self.reduced_ch)  # (N, C_r)
            # GATConv returns tensor shape (N, C_out)
            nodes_out = self.gat(nodes, edge_index)  # (N, C_r) if concat=False
            # reshapes back to (C_r, P_h, P_w)
            nodes_out = nodes_out.view(P_h, P_w, self.reduced_ch).permute(2, 0, 1)
            out_nodes.append(nodes_out)
        out = torch.stack(out_nodes, dim=0)  # (B, C_r, P_h, P_w)

        # 4) upsample back to original HxW
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)  # (B, C_r, H, W)

        # 5) restore channels
        out = self.restore(out)  # (B, out_ch, H, W)

        # 6) add residual and activate
        out = out + res
        out = self.bn(out)
        out = self.act(out)
        return out


