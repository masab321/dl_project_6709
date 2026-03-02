import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        """
        alpha: weight for false positives
        beta: weight for false negatives
        gamma: focal parameter (>0) to focus on hard examples
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: (B, 1, H, W) or (B, H, W); target: same shape, floats 0/1
        probs = torch.sigmoid(logits)
        b = target.size(0)
        probs = probs.view(b, -1)
        target = target.view(b, -1)
        TP = (probs * target).sum(dim=1)
        FP = (probs * (1 - target)).sum(dim=1)
        FN = ((1 - probs) * target).sum(dim=1)
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = (1.0 - tversky) ** self.gamma
        return loss.mean()

def iou_loss(logits, target, smooth=1e-6):
    probs = torch.sigmoid(logits)
    b = target.size(0)
    probs = probs.view(b, -1)
    target = target.view(b, -1)
    inter = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1) - inter
    iou = (inter + smooth) / (union + smooth)
    return (1.0 - iou).mean()

class ComboLoss(nn.Module):
    def __init__(self, weights=(0.4, 0.4, 0.2), alpha=0.7, beta=0.3, gamma=0.75):
        """
        weights: (w_bce, w_ft, w_iou)
        """
        super().__init__()
        self.w_bce, self.w_ft, self.w_iou = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.ft = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)
        ft_loss = self.ft(logits, target)
        iou_l = iou_loss(logits, target)
        return self.w_bce * bce_loss + self.w_ft * ft_loss + self.w_iou * iou_l
