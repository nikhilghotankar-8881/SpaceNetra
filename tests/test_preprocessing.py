"""
Unit Tests for Sentinel-2 Pre-processing & Cloud Masking Module.
"""

import numpy as np
import pytest
from src.ingestion.preprocessing import (
    DEFAULT_INVALID_SCL_CLASSES,
    Sentinel2Preprocessor,
    mask_clouds_scl,
    normalize_bands,
    scale_reflectance,
)


def test_scale_reflectance():
    raw_dn = np.array([0, 2500, 5000, 10000, 12000], dtype=np.uint16)
    scaled = scale_reflectance(raw_dn, scale_factor=10000.0)

    assert scaled.dtype == np.float32
    assert np.allclose(scaled, [0.0, 0.25, 0.5, 1.0, 1.0])


def test_mask_clouds_scl_2d():
    # 3x3 synthetic image
    data = np.ones((3, 3), dtype=np.float32)
    # SCL: 4=veg, 8=cloud_medium, 3=cloud_shadow
    scl = np.array([[4, 4, 4], [4, 8, 4], [3, 4, 4]], dtype=np.uint8)

    masked = mask_clouds_scl(data, scl, fill_value=-1.0)

    assert masked[0, 0] == 1.0
    assert masked[1, 1] == -1.0  # SCL 8 masked
    assert masked[2, 0] == -1.0  # SCL 3 masked
    assert masked[2, 2] == 1.0


def test_mask_clouds_scl_3d_hwc():
    # Shape (4, 4, 3) HWC
    data = np.ones((4, 4, 3), dtype=np.float32) * 0.8
    scl = np.zeros((4, 4), dtype=np.uint8)
    scl[1, 1] = 9  # High prob cloud
    scl[2, 3] = 10  # Thin cirrus

    masked = mask_clouds_scl(data, scl, fill_value=0.0)

    assert np.all(masked[1, 1, :] == 0.0)
    assert np.all(masked[2, 3, :] == 0.0)
    assert np.all(masked[0, 0, :] == 0.8)


def test_mask_clouds_scl_3d_chw():
    # Shape (4, 5, 5) CHW
    data = np.ones((4, 5, 5), dtype=np.float32) * 0.5
    scl = np.zeros((5, 5), dtype=np.uint8)
    scl[0, 2] = 11  # Snow

    masked = mask_clouds_scl(data, scl, fill_value=0.0, channel_first=True)

    assert np.all(masked[:, 0, 2] == 0.0)
    assert np.all(masked[:, 0, 0] == 0.5)


def test_mask_clouds_scl_invalid_dimensions():
    data = np.ones((4, 4), dtype=np.float32)
    scl_bad = np.ones((5, 5, 2), dtype=np.uint8)  # 3D SCL not squeezable to 2D

    with pytest.raises(ValueError):
        mask_clouds_scl(data, scl_bad)


def test_normalize_bands():
    data = np.ones((2, 2, 3), dtype=np.float32) * 0.5
    mean = [0.5, 0.5, 0.5]
    std = [0.1, 0.1, 0.1]

    normalized = normalize_bands(data, mean=mean, std=std)

    assert np.allclose(normalized, 0.0, atol=1e-5)


def test_sentinel2_preprocessor_pipeline():
    preprocessor = Sentinel2Preprocessor(
        scale_factor=10000.0,
        fill_value=0.0,
        mean=[0.2, 0.2, 0.2],
        std=[0.1, 0.1, 0.1],
    )

    # Raw DN HWC image (4, 4, 3)
    raw_dn = np.full((4, 4, 3), 4000, dtype=np.uint16)  # 4000/10000 = 0.4
    scl = np.zeros((4, 4), dtype=np.uint8)
    scl[3, 3] = 9  # High prob cloud

    processed = preprocessor.process(raw_dn, scl=scl, is_scaled=False)

    # Non-cloudy pixel: (0.4 - 0.2) / 0.1 = 2.0
    assert np.allclose(processed[0, 0, :], 2.0, atol=1e-4)

    # Cloudy pixel masked to fill_value = 0.0 (masked during step 2)
    # Note: step 3 normalizes 0.0 -> (0.0 - 0.2)/0.1 = -2.0
    assert np.allclose(processed[3, 3, :], (0.0 - 0.2) / 0.1, atol=1e-4)
