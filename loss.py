# -*- coding: utf-8 -*-
"""
BTES Tree Extraction - Improved Loss

Focal + Dice + Tversky Loss
Designed for tree canopy semantic segmentation
with minority foreground pixels.
"""

import torch
import torch.nn as nn


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        probs = torch.sigmoid(preds)

        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=0.75,
        gamma=2.0
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):

        bce = nn.functional.binary_cross_entropy_with_logits(
            preds,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(preds)

        # Probability of the correct class
        pt = torch.where(
            targets == 1,
            probs,
            1.0 - probs
        )

        # Alpha weighting
        alpha_t = torch.where(
            targets == 1,
            self.alpha,
            1.0 - self.alpha
        )

        focal = alpha_t * (
            (1.0 - pt) ** self.gamma
        ) * bce

        return focal.mean()


# ============================================================
# TVERSKY LOSS
# ============================================================

class TverskyLoss(nn.Module):

    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        smooth=1.0
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):

        probs = torch.sigmoid(preds)

        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)

        true_positive = (probs * targets).sum(dim=1)

        false_positive = (
            probs * (1.0 - targets)
        ).sum(dim=1)

        false_negative = (
            (1.0 - probs) * targets
        ).sum(dim=1)

        tversky = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        return 1.0 - tversky.mean()


# ============================================================
# COMBINED LOSS
# ============================================================

class FocalDiceTverskyLoss(nn.Module):

    def __init__(
        self,
        focal_weight=0.30,
        dice_weight=0.40,
        tversky_weight=0.30,

        focal_alpha=0.75,
        focal_gamma=2.0,

        tversky_alpha=0.3,
        tversky_beta=0.7
    ):
        super().__init__()

        self.focal = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma
        )

        self.dice = DiceLoss()

        self.tversky = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta
        )

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight

    def forward(self, preds, targets):

        focal_loss = self.focal(
            preds,
            targets
        )

        dice_loss = self.dice(
            preds,
            targets
        )

        tversky_loss = self.tversky(
            preds,
            targets
        )

        total_loss = (
            self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
            + self.tversky_weight * tversky_loss
        )

        return total_loss


# ============================================================
# GET LOSS
# ============================================================

def get_loss():

    from config import CFG

    return FocalDiceTverskyLoss(
        focal_weight=CFG.FOCAL_WEIGHT,
        dice_weight=CFG.DICE_WEIGHT,
        tversky_weight=CFG.TVERSKY_WEIGHT,

        focal_alpha=CFG.FOCAL_ALPHA,
        focal_gamma=CFG.FOCAL_GAMMA,

        tversky_alpha=CFG.TVERSKY_ALPHA,
        tversky_beta=CFG.TVERSKY_BETA
    )