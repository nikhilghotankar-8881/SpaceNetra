"""
ChangeFormer Architecture for SpaceNetra.

Transformer-based change detection network featuring:
- Shared hierarchical vision transformer encoder (MiT backbone).
- Multi-scale feature differencing at each stage (|f1_i - f2_i|).
- Lightweight MLP decoder fusing feature difference maps across resolutions.
- Single-channel change prediction output.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixFFN(nn.Module):
    """
    Mix Feed-Forward Network featuring 3x3 depthwise convolution for localized spatial context.
    """

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, padding=1, groups=hidden_features
        )
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.dwconv(out)
        out = self.act(out)
        out = self.fc2(out)
        return out


class EfficientAttention(nn.Module):
    """
    Spatial Reduction Multi-Head Self-Attention for high-resolution satellite imagery.
    """

    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = (
            self.q(x)
            .reshape(B, self.num_heads, self.head_dim, H * W)
            .permute(0, 1, 3, 2)
        )

        if self.sr_ratio > 1:
            x_sr = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.norm(x_sr).permute(0, 2, 1).reshape(
                B, C, H // self.sr_ratio, W // self.sr_ratio
            )
            kv = self.kv(x_sr)
        else:
            kv = self.kv(x)

        kv = kv.reshape(B, 2, self.num_heads, self.head_dim, -1).permute(
            1, 0, 2, 4, 3
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out)


class ChangeFormerStage(nn.Module):
    """
    Stage block combining spatial reduction attention, MixFFN, and BatchNorm layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        sr_ratio: int = 1,
    ):
        super().__init__()
        if stride > 1:
            self.downsample = nn.Conv2d(
                in_channels, out_channels, kernel_size=stride, stride=stride
            )
        elif in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.downsample = nn.Identity()

        self.attn = EfficientAttention(dim=out_channels, sr_ratio=sr_ratio)
        self.ffn = MixFFN(
            in_features=out_channels, hidden_features=out_channels * 4
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ChangeFormerEncoder(nn.Module):
    """
    Weight-shared hierarchical vision transformer encoder.
    """

    def __init__(
        self, in_channels: int = 3, embed_dims: List[int] = [64, 128, 320, 512]
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                embed_dims[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ChangeFormerStage(
            embed_dims[0], embed_dims[0], stride=2, sr_ratio=4
        )  # H/4, W/4
        self.stage2 = ChangeFormerStage(
            embed_dims[0], embed_dims[1], stride=2, sr_ratio=2
        )  # H/8, W/8
        self.stage3 = ChangeFormerStage(
            embed_dims[1], embed_dims[2], stride=2, sr_ratio=1
        )  # H/16, W/16
        self.stage4 = ChangeFormerStage(
            embed_dims[2], embed_dims[3], stride=2, sr_ratio=1
        )  # H/32, W/32

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.stem(x)
        s1 = self.stage1(x0)  # (B, 64, H/4, W/4)
        s2 = self.stage2(s1)  # (B, 128, H/8, W/8)
        s3 = self.stage3(s2)  # (B, 320, H/16, W/16)
        s4 = self.stage4(s3)  # (B, 512, H/32, W/32)
        return [s1, s2, s3, s4]


class ChangeFormerDecoder(nn.Module):
    """
    Lightweight MLP decoder fusing multi-scale feature difference maps.
    """

    def __init__(
        self,
        embed_dims: List[int] = [64, 128, 320, 512],
        decode_dim: int = 256,
    ):
        super().__init__()
        self.proj1 = nn.Conv2d(embed_dims[0], decode_dim, 1)
        self.proj2 = nn.Conv2d(embed_dims[1], decode_dim, 1)
        self.proj3 = nn.Conv2d(embed_dims[2], decode_dim, 1)
        self.proj4 = nn.Conv2d(embed_dims[3], decode_dim, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(decode_dim * 4, decode_dim, 1, bias=False),
            nn.BatchNorm2d(decode_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(decode_dim, 1, 1)

    def forward(
        self, diffs: List[torch.Tensor], target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        d1, d2, d3, d4 = diffs
        h4, w4 = d1.shape[2:]

        p1 = self.proj1(d1)
        p2 = F.interpolate(
            self.proj2(d2), size=(h4, w4), mode="bilinear", align_corners=False
        )
        p3 = F.interpolate(
            self.proj3(d3), size=(h4, w4), mode="bilinear", align_corners=False
        )
        p4 = F.interpolate(
            self.proj4(d4), size=(h4, w4), mode="bilinear", align_corners=False
        )

        cat_feat = torch.cat([p1, p2, p3, p4], dim=1)
        fused = self.fuse(cat_feat)

        out_feat = F.interpolate(
            fused, size=target_shape, mode="bilinear", align_corners=False
        )
        return self.head(out_feat)


class ChangeFormer(nn.Module):
    """
    ChangeFormer architecture for bi-temporal satellite change detection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        classes: int = 1,
        embed_dims: List[int] = [64, 128, 320, 512],
    ):
        super().__init__()
        self.encoder = ChangeFormerEncoder(
            in_channels=in_channels, embed_dims=embed_dims
        )
        self.decoder = ChangeFormerDecoder(
            embed_dims=embed_dims, decode_dim=256
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
            return_logits: If True, returns raw unnormalized logits.

        Returns:
            Change prediction tensor (B, 1, H, W)
        """
        feats1 = self.encoder(x1)
        feats2 = self.encoder(x2)

        diffs = [torch.abs(f1 - f2) for f1, f2 in zip(feats1, feats2)]
        target_shape = (x1.shape[2], x1.shape[3])

        logits = self.decoder(diffs, target_shape)
        if return_logits:
            return logits
        return torch.sigmoid(logits)
