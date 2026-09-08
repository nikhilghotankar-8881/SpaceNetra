"""
Unit tests for SpaceNetra loss functions (DiceLoss and BCEDiceLoss).
"""

import torch
import pytest

from src.models.losses import DiceLoss, BCEDiceLoss


def test_bce_dice_perfect_match():
    """Near-perfect alignment between logits and ground truth should yield loss close to 0."""
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    # Large positive logits for 1.0, large negative logits for 0.0
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])

    loss = criterion(logits, target)
    assert loss.item() < 0.05, f"Perfect match loss expected < 0.05, got {loss.item()}"


def test_bce_dice_complete_mismatch():
    """Complete mismatch between logits and ground truth should yield high loss (> 1.5)."""
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    target = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    # Large negative logits predicting 0 everywhere
    logits = torch.tensor([[[[-10.0, -10.0], [-10.0, -10.0]]]])

    loss = criterion(logits, target)
    assert loss.item() > 1.5, f"Complete mismatch loss expected > 1.5, got {loss.item()}"


def test_bce_dice_zero_target():
    """All-zero change targets should be evaluated cleanly without division by zero (smoothness)."""
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0, smooth=1.0)

    target = torch.zeros((2, 1, 256, 256))
    logits = torch.randn((2, 1, 256, 256))

    loss = criterion(logits, target)
    assert not torch.isnan(loss), "Loss computed on all-zero targets returned NaN!"
    assert not torch.isinf(loss), "Loss computed on all-zero targets returned Inf!"
    assert loss.item() > 0.0


def test_bce_dice_gradient_flow():
    """Backward pass through BCEDiceLoss must compute non-NaN, non-zero gradients for logits."""
    criterion = BCEDiceLoss()

    logits = torch.randn((4, 1, 64, 64), requires_grad=True)
    target = torch.randint(0, 2, (4, 1, 64, 64)).float()

    loss = criterion(logits, target)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert logits.grad.abs().sum().item() > 0.0


def test_bce_dice_weights():
    """bce_weight and dice_weight should scale their respective loss components linearly."""
    logits = torch.randn((2, 1, 32, 32))
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()

    loss_both = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)(logits, target)
    loss_bce_only = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)(logits, target)
    loss_dice_only = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)(logits, target)

    expected_sum = loss_bce_only + loss_dice_only
    assert torch.allclose(loss_both, expected_sum, atol=1e-5)
