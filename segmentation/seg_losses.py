# segmentation/seg_losses.py
#
# Loss and metrics for Hard Exudate segmentation. Plain BCE alone struggles
# with the <1% foreground/background imbalance we confirmed in the data
# pipeline (foreground fractions ~0.5-2%) — it can drive loss down while
# still predicting all-background. Dice loss (and the dice/IoU metrics)
# directly reward overlap with the sparse foreground, so combining the two
# gives a stable gradient signal early on plus calibrated probabilities.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation. Applies sigmoid internally, so
    it takes raw logits (matching the model's activation=None output) —
    never pass already-sigmoided probabilities in.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCEWithLogitsLoss + DiceLoss, both operating on raw logits."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred_logits, target):
        return self.bce(pred_logits, target) + self.dice(pred_logits, target)


@torch.no_grad()
def dice_coefficient(pred_logits, target, threshold=0.5, smooth=1.0):
    """Hard Dice coefficient at a fixed probability threshold. Returns a plain float."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


@torch.no_grad()
def iou_score(pred_logits, target, threshold=0.5, smooth=1.0):
    """Intersection-over-Union at a fixed probability threshold. Returns a plain float."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
