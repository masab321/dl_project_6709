import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostModule(nn.Module):
    """
    Lightweight convolution module that generates more features from cheap operations.
    Reduces parameters/FLOPs compared to standard Conv2d.
    """
    def __init__(self, imp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(imp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention focuses on 'where' the object is.
    Better than SE-Block for segmentation as it keeps spatial info.
    """
    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return x * a_h * a_w

class LiteRFB(nn.Module):
    """
    Lightweight Receptive Field Block.
    Uses multi-branch dilated depthwise convs to capture multi-scale context.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super(LiteRFB, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        branch_channels = in_ch // 4

        # Branch 0: 1x1 conv
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_ch, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 1: 1x1 conv + 3x3 depthwise (dilation 1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 conv + 3x3 depthwise (dilation 3)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=3, dilation=3, groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 conv + 3x3 depthwise (dilation 5)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=5, dilation=5, groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(branch_channels*4, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        x_cat = torch.cat((x0, x1, x2, x3), dim=1)
        x_cat = self.conv_cat(x_cat)
        
        return self.relu(x_cat + self.shortcut(x))

import math

class GhostRFBCoordBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GhostRFBCoordBottleneck, self).__init__()
        
        # 1. Expand features efficiently using Ghost Module
        self.ghost_expand = GhostModule(in_channels, out_channels)
        
        # 2. Capture Multi-scale Context (Improves IoU)
        self.rfb = LiteRFB(out_channels, out_channels)
        
        # 3. Refine features with Coordinate Attention (Improves Dice)
        self.coord_att = CoordinateAttention(out_channels)
        
    def forward(self, x):
        x = self.ghost_expand(x)
        x = self.rfb(x)
        x = self.coord_att(x)
        return x


# Backwards-compatible alias expected by `unet_model.py`
# Some codebases import `DSTransBottleneck`; provide a simple alias
# so existing imports continue to work without changing other files.
class DSTransBottleneck(GhostRFBCoordBottleneck):
    pass