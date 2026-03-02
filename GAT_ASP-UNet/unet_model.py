import torch
import torch.nn as nn
import torch.nn.functional as F

# Import custom modules
from ADFM import AdvancedFeatureModule
from FSCM import FixedVolumeCompressor
from bottolneck import GhostRFBCoordBottleneck
from GAT_Bridge import GraphGATBridge

# ==========================================
# HELPER BLOCKS
# ==========================================

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class ChannelAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.adapt = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.adapt = nn.Identity()
    def forward(self, x): return self.adapt(x)

# ==========================================
# OPTIMIZED UNET
# ==========================================

class CustomUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_filters=64, fscm_out_ch=32, fscm_target_size=(32, 32)):
        super().__init__()
        
        filters = [base_filters, base_filters*2, base_filters*4, base_filters*8] # [64, 128, 256, 512]
        
        # --- ENCODER BLOCKS ---
        self.e1 = EncoderBlock(in_channels, filters[0])
        self.e2 = EncoderBlock(filters[0], filters[1])
        self.e3 = EncoderBlock(filters[1], filters[2])
        self.e4 = EncoderBlock(filters[2], filters[3])

        # --- SHARED FEATURE BLOCKS (FSCM + ADFM) ---
        # OPTIMIZATION: Defined ONCE per level, used for both next-encoder AND bridge
        self.shared_feat1 = nn.Sequential(
            FixedVolumeCompressor(filters[0], out_ch=fscm_out_ch, target_size=fscm_target_size),
            AdvancedFeatureModule(fscm_out_ch)
        )
        self.shared_feat2 = nn.Sequential(
            FixedVolumeCompressor(filters[1], out_ch=fscm_out_ch, target_size=fscm_target_size),
            AdvancedFeatureModule(fscm_out_ch)
        )
        self.shared_feat3 = nn.Sequential(
            FixedVolumeCompressor(filters[2], out_ch=fscm_out_ch, target_size=fscm_target_size),
            AdvancedFeatureModule(fscm_out_ch)
        )

        # --- ENCODER FUSION ADAPTERS ---
        # 1. Adapt shared features (fscm_ch) to filter size
        self.enc_adapt1 = ChannelAdapter(fscm_out_ch, filters[0])
        self.enc_adapt2 = ChannelAdapter(fscm_out_ch, filters[1])
        self.enc_adapt3 = ChannelAdapter(fscm_out_ch, filters[2])
        
        # 2. Downsampling pooling
        self.pool = nn.MaxPool2d(2)

        # 3. Fusion Convs (Merge Pooled + Shared)
        self.enc_fusion1 = nn.Conv2d(filters[0]*2, filters[0], 1)
        self.enc_fusion2 = nn.Conv2d(filters[1]*2, filters[1], 1)
        self.enc_fusion3 = nn.Conv2d(filters[2]*2, filters[2], 1)

        # --- BOTTLENECK ---
        self.bottleneck = GhostRFBCoordBottleneck(filters[3], filters[3]*2) # 512 -> 1024

        # --- BRIDGE GAT BLOCKS ---
        # Takes the SHARED feature as input
        self.gat1 = GraphGATBridge(fscm_out_ch, filters[0]) # Output matches decoder target
        self.gat2 = GraphGATBridge(fscm_out_ch, filters[1])
        self.gat3 = GraphGATBridge(fscm_out_ch, filters[2])
        self.gat4 = GraphGATBridge(filters[3], filters[3]) # Direct from E4

        # --- DECODER BLOCKS ---
        # D4
        self.d4_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d4_conv = DecoderBlock(filters[3]*2 + filters[3], filters[3]) # 1024 (bottle) + 512 (gat4) -> 512
        self.d4_adfm = AdvancedFeatureModule(filters[3])

        # D3 (Concat: Direct E3 + GAT3 + Prev D4)
        self.d3_prev_adapt = ChannelAdapter(filters[3], filters[2]) # 512 -> 256
        self.d3_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input channels: filters[2](Direct) + filters[2](GAT) + filters[2](Prev)
        self.d3_conv = DecoderBlock(filters[2]*3, filters[2]) 
        self.d3_adfm = AdvancedFeatureModule(filters[2])

        # D2
        self.d2_prev_adapt = ChannelAdapter(filters[2], filters[1]) # 256 -> 128
        self.d2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d2_conv = DecoderBlock(filters[1]*3, filters[1])
        self.d2_adfm = AdvancedFeatureModule(filters[1])

        # D1
        self.d1_prev_adapt = ChannelAdapter(filters[1], filters[0]) # 128 -> 64
        self.d1_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1_conv = DecoderBlock(filters[0]*3, filters[0])
        self.d1_adfm = AdvancedFeatureModule(filters[0])

        self.final_conv = nn.Conv2d(filters[0], num_classes, 1)

    def _safe_interpolate(self, x, target_tensor):
        """Helper to interpolate x to match target_tensor spatial size"""
        if x.shape[2:] != target_tensor.shape[2:]:
            return F.interpolate(x, size=target_tensor.shape[2:], mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        # ----------------------
        # ENCODER 1
        # ----------------------
        e1 = self.e1(x)
        
        # Optimization: Calculate Shared Feature ONCE
        # Used for: 1. Next Encoder Input, 2. GAT Bridge
        feat1_shared = self.shared_feat1(e1) # (B, 32, 32, 32)
        
        # Path to Next Encoder
        p1 = self.pool(e1)
        feat1_adapted = self.enc_adapt1(feat1_shared)
        feat1_spatial = self._safe_interpolate(feat1_adapted, p1)
        e1_next_input = self.enc_fusion1(torch.cat([p1, feat1_spatial], dim=1))

        # ----------------------
        # ENCODER 2
        # ----------------------
        e2 = self.e2(e1_next_input)
        
        # Shared Feature
        feat2_shared = self.shared_feat2(e2)
        
        # Path to Next Encoder
        p2 = self.pool(e2)
        feat2_adapted = self.enc_adapt2(feat2_shared)
        feat2_spatial = self._safe_interpolate(feat2_adapted, p2)
        e2_next_input = self.enc_fusion2(torch.cat([p2, feat2_spatial], dim=1))

        # ----------------------
        # ENCODER 3
        # ----------------------
        e3 = self.e3(e2_next_input)
        
        # Shared Feature
        feat3_shared = self.shared_feat3(e3)
        
        # Path to Next Encoder
        p3 = self.pool(e3)
        feat3_adapted = self.enc_adapt3(feat3_shared)
        feat3_spatial = self._safe_interpolate(feat3_adapted, p3)
        e3_next_input = self.enc_fusion3(torch.cat([p3, feat3_spatial], dim=1))

        # ----------------------
        # ENCODER 4 & BOTTLENECK
        # ----------------------
        e4 = self.e4(e3_next_input)
        p4 = self.pool(e4)
        b_out = self.bottleneck(p4)

        # ----------------------
        # CALCULATE BRIDGES
        # ----------------------
        # We reuse the 'shared' features computed above!
        br1_gat = self.gat1(feat1_shared) 
        br2_gat = self.gat2(feat2_shared)
        br3_gat = self.gat3(feat3_shared)
        br4_gat = self.gat4(e4) # Direct bridge for e4

        # ----------------------
        # DECODER 4
        # ----------------------
        d4_up = self.d4_up(b_out)
        d4_up = self._safe_interpolate(d4_up, e4)
        br4_gat = self._safe_interpolate(br4_gat, e4)
        
        d4 = self.d4_conv(torch.cat([d4_up, br4_gat], dim=1))
        d4 = self.d4_adfm(d4)

        # ----------------------
        # DECODER 3 (3-Way Concat)
        # ----------------------
        # 1. Previous Decoder (D4)
        d3_up = self.d3_up(d4)
        d3_from_prev = self.d3_prev_adapt(d3_up)
        d3_from_prev = self._safe_interpolate(d3_from_prev, e3)
        
        # 2. Direct Encoder (E3)
        d3_from_enc = e3 # Already correct size usually, but check
        
        # 3. GAT Bridge (from shared)
        d3_from_bridge = self._safe_interpolate(br3_gat, e3)
        
        d3 = self.d3_conv(torch.cat([d3_from_enc, d3_from_bridge, d3_from_prev], dim=1))
        d3 = self.d3_adfm(d3)

        # ----------------------
        # DECODER 2 (3-Way Concat)
        # ----------------------
        d2_up = self.d2_up(d3)
        d2_from_prev = self.d2_prev_adapt(d2_up)
        d2_from_prev = self._safe_interpolate(d2_from_prev, e2)
        
        d2_from_enc = e2
        d2_from_bridge = self._safe_interpolate(br2_gat, e2)

        d2 = self.d2_conv(torch.cat([d2_from_enc, d2_from_bridge, d2_from_prev], dim=1))
        d2 = self.d2_adfm(d2)

        # ----------------------
        # DECODER 1 (3-Way Concat)
        # ----------------------
        d1_up = self.d1_up(d2)
        d1_from_prev = self.d1_prev_adapt(d1_up)
        d1_from_prev = self._safe_interpolate(d1_from_prev, e1)
        
        d1_from_enc = e1
        d1_from_bridge = self._safe_interpolate(br1_gat, e1)

        d1 = self.d1_conv(torch.cat([d1_from_enc, d1_from_bridge, d1_from_prev], dim=1))
        d1 = self.d1_adfm(d1)

        return self.final_conv(d1)

if __name__ == "__main__":
    print("Testing Optimized Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = CustomUNet(in_channels=3, num_classes=1).to(device)
    model.eval()
    
    # Calculate params to compare efficiency
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    x = torch.randn(2, 3, 256, 256).to(device)
    try:
        y = model(x)
        print(f"Input: {x.shape} -> Output: {y.shape}")
        print("Success! Dimensions match and logic is optimized.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

