"""Loss utilities for UNet training.

Requested combined loss:
  Loss = BCE + Dice + FocalTversky + \u03bb * GPU_Hausdorff, where \u03bb = 0.1

Notes:
- BCE uses logits (BCEWithLogitsLoss).
- Dice/FocalTversky use probabilities (sigmoid(logits)).
- GPU_Hausdorff here is a torch-only, differentiable surrogate based on
  multi-scale boundary matching via pooling (runs on GPU if tensors are on GPU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def _soft_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Compute a differentiable morphological gradient (dilation - erosion)."""
    mask = _ensure_bchw(mask)
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dil - ero).clamp_min(0.0)


def gpu_hausdorff_surrogate_from_probs(
    probs: torch.Tensor,
    target: torch.Tensor,
    max_dist: int = 10,
    boundary_kernel: int = 3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Torch-only Hausdorff-like boundary matching term.

    This is a *surrogate* that encourages predicted and GT boundaries to match
    within increasing dilation radii. It is differentiable w.r.t. probs.
    """
    probs = _ensure_bchw(probs)
    target = _ensure_bchw(target)
    if probs.shape != target.shape:
        raise ValueError(f"probs and target must have same shape, got {probs.shape} vs {target.shape}")

    pred_b = _soft_boundary(probs, kernel_size=boundary_kernel)
    gt_b = _soft_boundary(target, kernel_size=boundary_kernel)

    total = probs.new_zeros(())
    weight_sum = probs.new_zeros(())

    for d in range(1, int(max_dist) + 1):
        # multi-scale boundary dilation via pooling
        pred_d = F.max_pool2d(pred_b, kernel_size=2 * d + 1, stride=1, padding=d)
        gt_d = F.max_pool2d(gt_b, kernel_size=2 * d + 1, stride=1, padding=d)

        w = float(d) / float(max_dist)
        total = total + w * (
            (pred_d - gt_b).abs().mean() +
            (gt_d - pred_b).abs().mean()
        )
        weight_sum = weight_sum + w

    return total / (weight_sum + eps)


def dice_loss_from_probs(probs, target, smooth=1e-6, eps=1e-7):
    """
    probs, target : tensors with shape (B, 1, H, W) or (B, H, W)
    returns mean dice loss across batch
    """
    assert probs.shape == target.shape, "probs and target must have same shape"
    prob_flat = probs.contiguous().view(probs.shape[0], -1)
    targ_flat = target.contiguous().view(target.shape[0], -1)

    intersection = (prob_flat * targ_flat).sum(dim=1)
    denom = prob_flat.sum(dim=1) + targ_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denom + smooth + eps)
    return (1.0 - dice).mean()


def focal_tversky_loss_from_probs(probs,
                                  target,
                                  alpha=0.7,
                                  beta=0.3,
                                  gamma=0.75,
                                  smooth=1e-6,
                                  eps=1e-7):
    """
    Focal Tversky Loss:
      T = TP / (TP + alpha * FP + beta * FN)
      FocalTversky = (1 - T)^gamma
    probs, target : same shape (B, ...), floats in [0,1]
    """
    assert probs.shape == target.shape
    prob_flat = probs.contiguous().view(probs.shape[0], -1)
    targ_flat = target.contiguous().view(target.shape[0], -1)

    TP = (prob_flat * targ_flat).sum(dim=1)
    FP = (prob_flat * (1.0 - targ_flat)).sum(dim=1)
    FN = ((1.0 - prob_flat) * targ_flat).sum(dim=1)

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth + eps)
    ft = torch.pow((1.0 - tversky), gamma)
    return ft.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(self,
                 weight_bce=1.0,
                 weight_dice=1.0,
                 weight_focaltversky=1.0,
                 weight_hausdorff=0.1,
                 # Focal Tversky params
                 tv_alpha=0.7,
                 tv_beta=0.3,
                 tv_gamma=0.75,
                 # GPU Hausdorff params
                 hd_max_dist=10,
                 hd_boundary_kernel=3):
        """
        Combined loss: w_bce * BCE + w_dice * Dice + w_ft * FocalTversky + w_hd * GPU_Hausdorff

        Typical defaults:
          - weight_bce=1.0
          - weight_dice=1.0
          - weight_focaltversky=1.0
                    - weight_hausdorff=0.1  (requested \u03bb)
        """
        super().__init__()
        self.w_bce = weight_bce
        self.w_dice = weight_dice
        self.w_ft = weight_focaltversky
        self.w_hd = weight_hausdorff

        self.tv_alpha = tv_alpha
        self.tv_beta = tv_beta
        self.tv_gamma = tv_gamma

        self.hd_max_dist = hd_max_dist
        self.hd_boundary_kernel = hd_boundary_kernel

        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, target):
        """
        logits: raw logits from network, shape (B,1,H,W) or (B,H,W)
        target: binary {0,1} same shape, float or long tensors
        """
        # Ensure target float tensor
        if not (target.dtype == torch.float32 or target.dtype == torch.float64):
            target = target.float()

        bce_loss = self.bce(logits, target)

        probs = torch.sigmoid(logits)

        # Dice
        d_loss = dice_loss_from_probs(probs, target)

        # Focal Tversky
        ft_loss = focal_tversky_loss_from_probs(probs,
                                               target,
                                               alpha=self.tv_alpha,
                                               beta=self.tv_beta,
                                               gamma=self.tv_gamma)

        # GPU Hausdorff surrogate (torch-only)
        hd_loss = gpu_hausdorff_surrogate_from_probs(
            probs,
            target,
            max_dist=self.hd_max_dist,
            boundary_kernel=self.hd_boundary_kernel,
        )

        total = (self.w_bce * bce_loss
                 + self.w_dice * d_loss
                 + self.w_ft * ft_loss
                 + self.w_hd * hd_loss)

        return total
