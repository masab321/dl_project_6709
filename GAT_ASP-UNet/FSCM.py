# fix sixe compressor module
    
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Classes (Same as before) ---
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, 
                                   dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class EfficientASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.branches.append(DSConv(in_ch, out_ch, 3, 1, padding=d, dilation=d))

        self.project = nn.Sequential(
            nn.Conv2d(len(dilations) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.project(x)

# --- Main Module ---
class FixedVolumeCompressor(nn.Module):
    def __init__(self, in_ch, out_ch=32, target_size=(32, 32), bottleneck_ratio=4, apply_deformable=True):
        super().__init__()
        
        # 1. Configuration
        self.target_size = target_size  # (32, 32)
        mid = max(1, in_ch // bottleneck_ratio)

        # 2. Reduce Channels (Cheapest step first)
        self.reduce1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )

        # 3. Deformable Conv (Refine features at original resolution)
        if apply_deformable:
            # Placeholder for DeformConv2d using standard conv
            self.deform = nn.Conv2d(mid, mid, 3, padding=1, bias=False) 
        else:
            self.deform = nn.Identity()

        # 4. FORCE SPATIAL SIZE (The Magic Step)
        # AdaptiveAvgPool2d forces the H,W to be exactly target_size
        self.spatial_resize = nn.AdaptiveAvgPool2d(self.target_size)

        # 5. Efficient ASPP (Now runs cheaply on the small 32x32 grid)
        self.aspp = EfficientASPP(mid, mid, dilations=(1, 6, 12))

        # 6. Final Projection to target channel count (e.g., 32)
        self.project = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (B, in_ch, H_any, W_any)
        
        x = self.reduce1(x)         # (B, mid, H, W)
        x = self.deform(x)          # (B, mid, H, W)
        x = self.spatial_resize(x)  # (B, mid, 32, 32)  <-- Forced Resize
        x = self.aspp(x)            # (B, mid, 32, 32)
        x = self.project(x)         # (B, 32, 32, 32)
        
        return x