"""
Unit tests for SpaceNetra tiling and stitching utilities.
"""

import numpy as np
import pytest

from src.data.tiling import (
    split_into_patches,
    stitch_patches,
    split_and_stitch_triplet,
)


def test_split_no_overlap_count():
    """1024x1024 image with stride=256 should yield 16 non-overlapping patches (4x4)."""
    image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    patches, positions, padded_size = split_into_patches(image, patch_size=256, stride=256)

    assert len(patches) == 16
    assert len(positions) == 16
    assert padded_size == (1024, 1024)
    assert all(p.shape == (256, 256, 3) for p in patches)


def test_split_with_overlap_count():
    """1024x1024 image with stride=192 should yield 25 overlapping patches (5x5)."""
    image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    patches, positions, padded_size = split_into_patches(image, patch_size=256, stride=192)

    assert len(patches) == 25
    assert len(positions) == 25
    assert padded_size == (1024, 1024)
    assert all(p.shape == (256, 256, 3) for p in patches)


def test_split_non_divisible_padding():
    """1000x1000 image with stride=256 should pad to 1024x1024 and yield 16 patches."""
    image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
    patches, positions, padded_size = split_into_patches(image, patch_size=256, stride=256)

    assert len(patches) == 16
    assert padded_size == (1024, 1024)
    assert all(p.shape == (256, 256, 3) for p in patches)


def test_roundtrip_no_overlap():
    """Splitting and stitching without overlap should reconstruct the image perfectly (MSE = 0)."""
    image = np.random.uniform(0, 1, (1024, 1024)).astype(np.float32)
    patches, positions, padded_size = split_into_patches(image, patch_size=256, stride=256)

    reconstructed = stitch_patches(
        patches=patches,
        positions=positions,
        original_size=image.shape,
        padded_size=padded_size,
        patch_size=256,
    )

    assert reconstructed.shape == image.shape
    mse = np.mean((image - reconstructed) ** 2)
    assert mse == pytest.approx(0.0, abs=1e-7)


def test_roundtrip_with_overlap():
    """Splitting and stitching with overlap (stride=192) should reconstruct the image seamlessly."""
    image = np.random.uniform(0, 1, (1024, 1024, 3)).astype(np.float32)
    patches, positions, padded_size = split_into_patches(image, patch_size=256, stride=192)

    reconstructed = stitch_patches(
        patches=patches,
        positions=positions,
        original_size=(1024, 1024),
        padded_size=padded_size,
        patch_size=256,
    )

    assert reconstructed.shape == image.shape
    mse = np.mean((image - reconstructed) ** 2)
    assert mse < 1e-5


def test_triplet_split_alignment():
    """split_and_stitch_triplet should produce matching aligned patch pairs for T1 and T2."""
    img_a = np.ones((512, 512, 3), dtype=np.float32) * 1.0
    img_b = np.ones((512, 512, 3), dtype=np.float32) * 2.0

    patch_pairs, positions, orig_size, padded_size = split_and_stitch_triplet(
        img_a, img_b, patch_size=256, stride=256
    )

    assert len(patch_pairs) == 4
    assert orig_size == (512, 512)
    assert padded_size == (512, 512)

    for patch_a, patch_b in patch_pairs:
        assert np.all(patch_a == 1.0)
        assert np.all(patch_b == 2.0)
