# GAT_ASP-UNet: Ensemble-Based U-Net for Medical Image Segmentation

> A novel encoder–decoder architecture integrating Graph Attention Networks and Fixed-Volume Compression for robust medical image segmentation.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-GATConv-orange.svg)](https://pyg.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![BUET](https://img.shields.io/badge/Institution-BUET-blue.svg)](https://www.buet.ac.bd)

---

## 📋 Overview

Medical image segmentation demands precise boundary delineation even under dense-distribution and fuzzy-edge conditions. **GAT_ASP-UNet** couples a **Fixed-Volume Compressor (FSCM)** with a **Graph Attention (GAT) Bridge** to introduce structured relational reasoning into skip connections at constant complexity **O(1024)**, independent of input resolution.

| Highlight | Detail |
|---|---|
| 🎯 Task | Medical Image Segmentation |
| 🏗️ Architecture | Dual-path Encoder + Triple-concat Decoder |
| 🧠 Key Innovation | GAT-enhanced skip connections at fixed O(1024) complexity |
| 🏆 Best Result | **90.88% Dice** on ISIC2018 (dermoscopic) |
| 🏫 Institution | Bangladesh University of Engineering and Technology (BUET) |

---

## 🏗️ Architecture

GAT_ASP-UNet is built around a **dual-path encoder** and **triple-concatenation decoder** with four key components:

### 1. Advanced Deep Feature Module (ADFM)
The primary engine for multi-scale context capture. Employs **10 parallel branches** with 1×1, 3×3, 5×5, and dilated kernels alongside an ASPP block. Each branch incorporates downsampled self-attention, followed by a global Multi-Head Attention (MHA) layer and 1×1 fusion with a residual connection.

### 2. Fixed Volume Compressor (FSCM)
Forces features into a compact **32×32 spatial volume** via 1×1 channel reduction and adaptive average pooling. This bounds the complexity of all subsequent attention and GAT operations to a constant **O(1024)**, making the most intensive parts of the network resolution-independent.

```
Input Tensor (B, in_ch, H, W)
    → Channel Reduction (1×1 Conv, BN, ReLU)
    → Feature Refinement (Deformable Conv 3×3)
    → Forced Spatial Resize (AdaptiveAvgPool2d → 32×32)
    → Multi-Scale ASPP (DSConv dil=12, dil=6, Conv 1×1)
    → Final Projection (1×1 Conv, BN, ReLU)
Output Tensor (B, out_ch, 32, 32)
```

### 3. Graph Attention (GAT) Bridge
Converts spatial features into a **4-neighbourhood grid graph** and applies `GATConv` (PyTorch Geometric) for relational reasoning between spatial nodes. Features are pooled to a P×P grid, processed, then upsampled back to the original skip connection dimensions.

### 4. GhostRFBCoordBottleneck
A lightweight bottleneck combining:
- **GhostModule** for cheap feature expansion
- **LiteRFB** (multi-branch dilated depthwise convolutions) to expand the receptive field
- **CoordinateAttention** for spatial-aware channel gating

### Encoder / Decoder Strategy
```
Dual-Path Encoder (per stage):
  Path A: Encoder Output → FSCM → ADFM → ChannelAdapt → Upsample
  Path B: Encoder Output → MaxPool/Downsample
  → Concatenate & Fuse (1×1 Conv)

Triple-Concat Decoder:
  (1) Direct encoder features
  (2) GAT Bridge outputs
  (3) Previous decoder features re-processed via FSCM → ADFM
```

---

## 📊 Results

### External Dataset Performance

| Dataset | Modality | Split | IoU (%) | Dice (%) | Precision (%) | Recall (%) | Accuracy (%) |
|---|---|---|---|---|---|---|---|
| Kvasir-SEG | Endoscopic | Val | 76.17 | 86.14 | 88.99 | 84.27 | 95.86 |
| Kvasir-SEG | Endoscopic | Train | 74.48 | 85.03 | 88.07 | 83.13 | 95.60 |
| **ISIC2018** | **Dermoscopic** | **Val** | **83.61** | **90.88** | **93.64** | **88.81** | **96.24** |
| ISIC2018 | Dermoscopic | Train | 82.62 | 90.28 | 92.37 | 88.88 | 96.04 |
| Breast US B | Ultrasound | Val | 66.64 | 79.40 | 82.75 | 76.88 | 98.17 |
| Breast US B | Ultrasound | Train | 63.54 | 77.37 | 73.91 | 83.34 | 97.86 |

### Comparative Analysis on CVC-ClinicDB

| Model / Variant | Val Loss | Val IoU (%) | Val Dice (%) | Val Acc (%) | Val Prec (%) | Val Rec (%) |
|---|---|---|---|---|---|---|
| DDSUNet – Dice loss | 0.11762 | 86.74 | 92.63 | 98.61 | 94.69 | 91.33 |
| DDSUNet – Dice + BCE | 0.04925 | 85.28 | 91.77 | 98.51 | 93.77 | 90.53 |
| DDSUNet + GATConv | 0.12696 | 86.39 | 92.47 | 98.60 | 93.77 | **91.91** |
| Proposed – Focal Tversky | 0.21342 | 72.58 | 83.22 | 96.76 | 83.59 | 84.99 |
| Proposed – Dice + BCE | 0.23182 | 76.30 | 86.14 | 97.54 | 88.98 | 84.58 |

> **Key finding:** The GATConv baseline variant achieved the highest validation recall (91.91%), empirically supporting the thesis that graph-based relational reasoning captures complex non-local boundaries.

---

## 🗃️ Datasets

| Dataset | Modality / Target | Images |
|---|---|---|
| CVC-ClinicDB | Endoscopic / Polyp detection | 612 |
| ISIC2018 | Dermoscopic / Skin lesions | 2,596 |
| Kvasir-SEG | Endoscopic / Polyp images | 1,000 |
| Breast Ultrasound B | Ultrasound / Breast lesions | 163 |

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Dataset Split | 80% Train / 20% Validation |
| Epochs | 100 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Primary Loss | Dice + BCE |
| Secondary Loss | Focal Tversky + IoU |
| Augmentations | 9 techniques (elastic transform, colour jitter, random occlusion, and more) |
| Evaluation Metrics | IoU, Dice, Precision, Recall, Accuracy |

---

## 🔬 Key Findings & Limitations

- ✅ **Dermoscopic strength**: Model generalises exceptionally well to ISIC2018 (Dice: 90.88%)
- ✅ **GAT recall boost**: Graph-based skip connections improve boundary recall over standard baselines
- ⚠️ **Overfitting on CVC-ClinicDB**: Validation performance lags behind DDSUNet baselines
- ⚠️ **Ultrasound domain gap**: Lower Breast Ultrasound B scores suggest need for domain-specific augmentation

---

## 🚀 Future Work

- **Enhanced Regularisation** — Dropout and weight decay within ADFM and GAT Bridge for improved robustness
- **Structural Ablations** — Investigate GAT layer depth and attention head counts to optimise relational reasoning vs. parameter count
- **Data Augmentation** — Stronger elastic transforms and random occlusion for clinical endoscopic variability

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{chowdhury2026gataspunet,
  title     = {GAT\_ASP-UNet: Unified Deep Learning Approaches — Ensemble-Based U-Net for Medical Image Segmentation},
  author    = {Chowdhury, Rahul Drabit and Hasnain, Masab and Islam, Mareful},
  institution = {Bangladesh University of Engineering and Technology},
  year      = {2026}
}
```

---

## 👥 Authors

| Name | Student ID | Email |
|---|---|---|
| Rahul Drabit Chowdhury | 0424057003 | rahuldrabit@gmail.com |
| Masab Hasnain | 0424052099 | masabhasnain1@gmail.com |
| Mareful Islam | 0424056005 | 0424056005@grad.cse.buet.ac.bd |

*Department of Computer Science and Engineering, Bangladesh University of Engineering and Technology (BUET)*

---

## 📚 References

1. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI, 2015.
2. Y. Ou et al., "Enhanced medical image segmentation via deep dynamic self-adjusting U-Net with multi-scale attention and semantic mitigation," *The Visual Computer*, vol. 41, 2025.
3. Y. Wang, S. Wang, and J. He, "MFA U-Net: a U-Net like multi-stage feature analysis network," *Pattern Analysis and Applications*, vol. 27, 2024.
4. M. R. Ahmed et al., "DoubleU-NetPlus: a novel attention and context-guided dual U-Net," *Neural Computing and Applications*, vol. 35, 2023.
5. H. Wang et al., "UCTransNet: rethinking the skip connections in U-Net from a channel-wise perspective with transformer," *AAAI*, vol. 36, 2022.