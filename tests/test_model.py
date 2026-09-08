"""
Unit tests for SpaceNetra Siamese U-Net change detection model.
"""

import torch
import pytest

from src.models.siamese_unet import SiameseUNet


def test_siamese_unet_output_shape():
    """Batch (2, 3, 256, 256) should produce change mask of shape (2, 1, 256, 256)."""
    model = SiameseUNet(backbone="resnet18", pretrained=False)
    model.eval()

    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        out = model(x1, x2)

    assert out.shape == (2, 1, 256, 256)


def test_siamese_unet_value_range():
    """Sigmoid output (return_logits=False) must be strictly bounded in [0.0, 1.0]."""
    model = SiameseUNet(backbone="resnet18", pretrained=False)
    model.eval()

    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        probs = model(x1, x2, return_logits=False)
        logits = model(x1, x2, return_logits=True)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    # Logits should match sigmoid when converted manually
    assert torch.allclose(torch.sigmoid(logits), probs, atol=1e-5)


def test_siamese_unet_symmetric():
    """Swapping input order (x1, x2) vs (x2, x1) must produce identical predictions."""
    model = SiameseUNet(backbone="resnet18", pretrained=False)
    model.eval()

    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out_a = model(x1, x2)
        out_b = model(x2, x1)

    diff = torch.max(torch.abs(out_a - out_b)).item()
    assert diff < 1e-6, f"Siamese model output is not symmetric! Max diff: {diff}"


def test_siamese_unet_gradient_flow():
    """Backward pass should successfully calculate gradients for all parameters."""
    model = SiameseUNet(backbone="resnet18", pretrained=False)
    model.train()

    x1 = torch.randn(2, 3, 256, 256, requires_grad=True)
    x2 = torch.randn(2, 3, 256, 256, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()

    logits = model(x1, x2, return_logits=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    # Check that all model parameters received non-None gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter '{name}' received no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter '{name}' grad is NaN!"


def test_siamese_unet_backbone_options():
    """Model should instantiate and execute with both resnet18 and resnet34 backbones."""
    for backbone in ["resnet18", "resnet34"]:
        model = SiameseUNet(backbone=backbone, pretrained=False)
        model.eval()

        x1 = torch.randn(1, 3, 128, 128)
        x2 = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            out = model(x1, x2)

        assert out.shape == (1, 1, 128, 128)
