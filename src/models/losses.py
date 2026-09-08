"""
Loss functions for SpaceNetra Change Detection model training.

Includes:
- DiceLoss: Soft Dice Loss for imbalanced binary change segmentation.
- BCEDiceLoss: Combined Binary Cross-Entropy + Soft Dice Loss.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary change detection.

    Optimizes region overlap (IoU/F1 score) directly, handling class imbalance
    where change pixels are sparse (<10% of total image pixels).
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits or probabilities tensor.
            targets: Binary ground truth mask (same shape as inputs).

        Returns:
            Scalar Dice loss value in range [0.0, 1.0].
        """
        if self.from_logits:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy (BCE) and Soft Dice Loss.

    Loss = alpha * BCE + beta * Dice
    BCE provides smooth pixel-level gradient landscape, while Dice Loss
    counteracts class imbalance in building/infrastructure change maps.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss(smooth=smooth, from_logits=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw network output logits before Sigmoid activation.
            targets: Binary ground truth mask tensor.

        Returns:
            Weighted sum scalar loss.
        """
        targets_f = targets.float()
        bce = self.bce_loss(logits, targets_f)
        dice = self.dice_loss(logits, targets_f)
        return self.bce_weight * bce + self.dice_weight * dice
