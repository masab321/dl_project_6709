# border enhancment module

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    Standard Attention Block containing Multi-Head Attention + LayerNorm.
    Can be used for both the 'Self-Attention' (per branch) and 'Multi-Head Attention' (global) steps.
    """
    def __init__(self, in_channels, num_heads=4):
        super(AttentionBlock, self).__init__()
        # Ensure num_heads divides in_channels; default to 1 if channels are small (e.g., < 4)
        if in_channels % num_heads != 0:
            num_heads = 1
            
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Self-Attention: Query=x, Key=x, Value=x
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        
        # Add & Norm
        x_out = self.norm(x_flat + attn_out)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        return x_out

class ConvBlock(nn.Module):
    """
    Basic Convolutional Block: Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Block
    Returns concatenated features of standard ASPP rates.
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.blocks = nn.ModuleList()
        
        # 1x1 conv
        self.blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        ))
        
        # 3x3 convs with dilation
        for rate in dilations[1:]:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU()
            ))
            
        # Global Average Pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    def forward(self, x):
        res = []
        for block in self.blocks:
            res.append(block(x))
            
        gap = self.global_avg_pool(x)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(gap)
        
        return torch.cat(res, dim=1)

class CustomEnhancementModule(nn.Module):
    def __init__(self, in_channels, branch_channels=32):
        super(CustomEnhancementModule, self).__init__()
        
        self.branch_channels = branch_channels

        # ==========================================
        # 1. DEFINE CONVOLUTIONAL BRANCHES
        # ==========================================
        
        # 1x1 Branches
        self.conv_1x1_s1 = ConvBlock(in_channels, branch_channels, 1, 1, 0)
        self.conv_1x1_s2 = ConvBlock(in_channels, branch_channels, 1, 2, 0)

        # 3x3 Branches
        self.conv_3x3_s1 = ConvBlock(in_channels, branch_channels, 3, 1, 1)
        self.conv_3x3_s2 = ConvBlock(in_channels, branch_channels, 3, 2, 1)

        # 5x5 Branches
        self.conv_5x5_s1 = ConvBlock(in_channels, branch_channels, 5, 1, 2)
        self.conv_5x5_s2 = ConvBlock(in_channels, branch_channels, 5, 2, 2)

        # Dilation 7x7 Branch
        self.conv_dil_7 = ConvBlock(in_channels, branch_channels, 3, 1, 7, dilation=7)

        # Pooling Branches
        self.avg_pool_branch = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            ConvBlock(in_channels, branch_channels, 1, 1, 0) # Reduce channels after pool
        )
        
        self.aspp = ASPP(in_channels, branch_channels) # ASPP Output channels = branch_channels * 5

        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool_conv = ConvBlock(in_channels, branch_channels, 1, 1, 0)

        # ==========================================
        # 2. DEFINE "SELF-ATTENTION" (PER BRANCH)
        # ==========================================
        # We apply attention individually to every branch output
        
        self.sa_1x1_s1 = AttentionBlock(branch_channels)
        self.sa_1x1_s2 = AttentionBlock(branch_channels)
        
        self.sa_3x3_s1 = AttentionBlock(branch_channels)
        self.sa_3x3_s2 = AttentionBlock(branch_channels)
        
        self.sa_5x5_s1 = AttentionBlock(branch_channels)
        self.sa_5x5_s2 = AttentionBlock(branch_channels)
        
        self.sa_dil_7  = AttentionBlock(branch_channels)
        self.sa_avg    = AttentionBlock(branch_channels)
        
        # ASPP has 5 internal branches combined, so input dim is branch_channels * 5
        self.sa_aspp   = AttentionBlock(branch_channels * 5) 
        self.sa_max    = AttentionBlock(branch_channels)

        # ==========================================
        # 3. DEFINE "MULTI-HEAD ATTENTION" (GLOBAL)
        # ==========================================
        # Calculate total channels after concatenation
        # 6 basic branches + 1 dilation + 1 avg + 1 max + ASPP(5*branch)
        total_concat_channels = (branch_channels * 9) + (branch_channels * 5)
        
        self.global_mha = AttentionBlock(total_concat_channels, num_heads=8)

        # ==========================================
        # 4. FUSION
        # ==========================================
        self.fusion = nn.Sequential(
            nn.Conv2d(total_concat_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        target_size = x.shape[-2:]
        
        # --- Helper for processing: Conv -> SelfAttention -> Upsample(if needed) ---
        def process_branch(conv_layer, attn_layer, input_tensor, needs_upsample=False):
            feat = conv_layer(input_tensor)          # 1. Kernel Base
            feat = attn_layer(feat)                  # 2. Self Head Attention
            if needs_upsample:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            return feat

        # 1. Process all branches (Kernel + Self Attention)
        f1 = process_branch(self.conv_1x1_s1, self.sa_1x1_s1, x)
        f2 = process_branch(self.conv_1x1_s2, self.sa_1x1_s2, x, needs_upsample=True)
        
        f3 = process_branch(self.conv_3x3_s1, self.sa_3x3_s1, x)
        f4 = process_branch(self.conv_3x3_s2, self.sa_3x3_s2, x, needs_upsample=True)
        
        f5 = process_branch(self.conv_5x5_s1, self.sa_5x5_s1, x)
        f6 = process_branch(self.conv_5x5_s2, self.sa_5x5_s2, x, needs_upsample=True)
        
        f7 = process_branch(self.conv_dil_7, self.sa_dil_7, x)
        
        # Special Branches
        f8 = self.avg_pool_branch(x)
        f8 = self.sa_avg(f8)
        
        f9 = self.aspp(x)
        f9 = self.sa_aspp(f9)
        
        f10 = self.global_max_pool(x)
        f10 = self.max_pool_conv(f10) # (B, C, 1, 1)
        f10 = self.sa_max(f10) # Attention on 1x1 feature map
        f10 = F.interpolate(f10, size=target_size, mode='nearest')

        # 2. Concatenate
        # This combines all "Self-Attended" features
        concatenated = torch.cat([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dim=1)
        
        # 3. Multi-Head Attention (Global)
        # Applied after concatenation as requested
        global_feat = self.global_mha(concatenated)
        
        # 4. Final Fusion
        out = self.fusion(global_feat)
        
        return out

# --- Verification ---

#if __name__ == "__main__":
#    # Test with standard inputs
#    dummy_input = torch.randn(2, 64, 64, 64)
#    model = CustomEnhancementModule(in_channels=64, branch_channels=16)
#    output = model(dummy_input)
#    print(f"Input: {dummy_input.shape}")
#    print(f"Output: {output.shape}")