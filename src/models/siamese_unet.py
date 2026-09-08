"""
Siamese U-Net Change Detection Model for SpaceNetra.

Architecture:
- Shared ResNet-18 or ResNet-34 encoder for bi-temporal image pair (T1, T2).
- Absolute multi-scale feature differencing (|f1 - f2|).
- U-Net style decoder with skip connections and upsampling blocks.
- Single-channel change map output (logits or sigmoid probabilities).
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class SiameseEncoder(nn.Module):
    """
    Weight-shared ResNet encoder extracting feature maps at 5 resolution stages.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        backbone_str = backbone.lower()
        if backbone_str == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                resnet = tv_models.resnet18(weights=weights)
            except Exception:
                resnet = tv_models.resnet18(weights=None)
        elif backbone_str == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            try:
                resnet = tv_models.resnet34(weights=weights)
            except Exception:
                resnet = tv_models.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: '{backbone}'. Supported: 'resnet18', 'resnet34'")

        # Stage 0: Stem (conv1 -> bn1 -> relu) -> H/2, W/2 (64 ch)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool  # H/4, W/4

        # Stages 1 to 4
        self.layer1 = resnet.layer1  # 64 ch,  H/4,  W/4
        self.layer2 = resnet.layer2  # 128 ch, H/8,  W/8
        self.layer3 = resnet.layer3  # 256 ch, H/16, W/16
        self.layer4 = resnet.layer4  # 512 ch, H/32, W/32

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            List of 5 feature tensors [f0, f1, f2, f3, f4]
        """
        f0 = self.stem(x)              # (B, 64, H/2, W/2)
        x_pooled = self.maxpool(f0)    # (B, 64, H/4, W/4)

        f1 = self.layer1(x_pooled)     # (B, 64, H/4, W/4)
        f2 = self.layer2(f1)           # (B, 128, H/8, W/8)
        f3 = self.layer3(f2)           # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)           # (B, 512, H/32, W/32)

        return [f0, f1, f2, f3, f4]


class DecoderBlock(nn.Module):
    """
    U-Net decoder block with upsampling, skip connection concatenation, and double conv.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        concat_channels = in_channels + skip_channels

        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample by factor of 2
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # Concatenate along channel dimension
        x_cat = torch.cat([x_up, skip], dim=1)

        out = self.relu(self.bn1(self.conv1(x_cat)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for Change Detection on bi-temporal satellite images.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
        classes: int = 1,
    ):
        super().__init__()
        self.encoder = SiameseEncoder(backbone=backbone, pretrained=pretrained)

        # Decoder stages
        self.up4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.up3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.up2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.up1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Output head: upsample to full resolution and project to `classes` channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, kernel_size=1),
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for bi-temporal image pair.

        Args:
            x1: T1 image batch of shape (B, C, H, W)
            x2: T2 image batch of shape (B, C, H, W)
            return_logits: If True, returns un-normalized logits; if False, applies Sigmoid.

        Returns:
            Change mask prediction of shape (B, classes, H, W)
        """
        # Shared encoder pass
        feats1 = self.encoder(x1)
        feats2 = self.encoder(x2)

        # Absolute feature differencing across all stages
        diffs = [torch.abs(f1 - f2) for f1, f2 in zip(feats1, feats2)]
        diff0, diff1, diff2, diff3, diff4 = diffs

        # Decoder pass
        d4 = self.up4(diff4, diff3)  # -> (B, 256, H/16, W/16)
        d3 = self.up3(d4, diff2)     # -> (B, 128, H/8, W/8)
        d2 = self.up2(d3, diff1)     # -> (B, 64, H/4, W/4)
        d1 = self.up1(d2, diff0)     # -> (B, 64, H/2, W/2)

        # Final upsampling to original H, W
        d1_up = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        logits = self.final_conv(d1_up)

        if return_logits:
            return logits
        return torch.sigmoid(logits)
