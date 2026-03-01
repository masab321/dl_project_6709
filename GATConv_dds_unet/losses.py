import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['IBDLoss', 'BCEDiceLoss', 'BCELoss', 'DiceLoss', 'LovaszHingeLoss', 'FocalLoss', 'BCEDiceBoundaryLoss']

# def structure_loss(pred, mask):
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wbce + wiou).mean()
#
#
# # PyTorch
# class IoULoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(IoULoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1):
#         # comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)
#
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         # intersection is equivalent to True Positive count
#         # union is the mutually inclusive area of all labels & predictions
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         union = total - intersection
#
#         IoU = (intersection + smooth) / (union + smooth)
#
#         return 1 - IoU

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss

class IBDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)

        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        union = (input + target)
        iou = (intersection.sum(1) + 1) / (union.sum(1) - intersection.sum(1) + 1)
        iou = 1 - iou.sum() / num
        return 0.5 * bce + dice + 0.5 * iou

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        bce = F.binary_cross_entropy_with_logits(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice




class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        return bce

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class BCEDiceBoundaryLoss(nn.Module):
    """
    Final loss for high-Dice binary segmentation.
    - BCE: pixel confidence
    - Dice: region overlap
    - Boundary: edge alignment
    Works with logits (NO sigmoid in model).
    """

    def __init__(self,
                 w_bce=0.5,
                 w_dice=1.0,
                 w_boundary=0.2,
                 smooth=1e-5):
        super().__init__()

        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss()

        # Sobel kernels (auto-moved to GPU)
        self.register_buffer(
            "sobel_x",
            torch.tensor([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        )

    def forward(self, logits, target):
        # ---------- BCE ----------
        loss_bce = self.bce(logits, target)

        # ---------- Dice ----------
        pred = torch.sigmoid(logits)
        b = target.size(0)

        pred_flat = pred.view(b, -1)
        target_flat = target.view(b, -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / \
               (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)
        loss_dice = 1 - dice.mean()

        # ---------- Boundary ----------
        pred_edge = torch.abs(F.conv2d(pred, self.sobel_x, padding=1)) + \
                    torch.abs(F.conv2d(pred, self.sobel_y, padding=1))
        gt_edge = torch.abs(F.conv2d(target, self.sobel_x, padding=1)) + \
                  torch.abs(F.conv2d(target, self.sobel_y, padding=1))
        loss_boundary = F.l1_loss(pred_edge, gt_edge)

        # ---------- Total ----------
        return (
            self.w_bce * loss_bce +
            self.w_dice * loss_dice +
            self.w_boundary * loss_boundary
        )

