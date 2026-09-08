"""
Unit tests for SpaceNetra satellite ingestion pipeline (Sentinel-2 & GeoTIFF).
"""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.ingestion.geotiff import GeoTIFFReader
from src.ingestion.sentinel import Sentinel2Handler
from scripts.download_sentinel import build_parser


def test_sentinel2_band_stacking():
    """Sentinel2Handler.stack_10m_bands should stack B04, B03, B02, B08 into shape (H, W, 4)."""
    b02 = np.full((128, 128), 1000, dtype=np.uint16)
    b03 = np.full((128, 128), 2000, dtype=np.uint16)
    b04 = np.full((128, 128), 3000, dtype=np.uint16)
    b08 = np.full((128, 128), 4000, dtype=np.uint16)

    stacked_4ch = Sentinel2Handler.stack_10m_bands(b02, b03, b04, b08, normalize=True)
    assert stacked_4ch.shape == (128, 128, 4)
    assert stacked_4ch.dtype == np.float32
    assert stacked_4ch.min() >= 0.0
    assert stacked_4ch.max() <= 1.0

    # Order check: B04 (0.3), B03 (0.2), B02 (0.1), B08 (0.4)
    assert np.allclose(stacked_4ch[0, 0, 0], 0.3, atol=1e-4)
    assert np.allclose(stacked_4ch[0, 0, 1], 0.2, atol=1e-4)
    assert np.allclose(stacked_4ch[0, 0, 2], 0.1, atol=1e-4)
    assert np.allclose(stacked_4ch[0, 0, 3], 0.4, atol=1e-4)


def test_geotiff_metadata_extraction():
    """GeoTIFFReader should extract spatial metadata and raster array."""
    dummy_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "dummy.tif"
        Image.fromarray(dummy_img).save(file_path)

        reader = GeoTIFFReader(file_path)
        raster_array, metadata = reader.read_raster(normalize=True)

        assert raster_array.shape == (64, 64, 3)
        assert metadata["width"] == 64
        assert metadata["height"] == 64
        assert metadata["count"] == 3


def test_cloud_cover_filtering():
    """Sentinel2Handler.filter_scenes should reject scenes exceeding cloud_cover threshold."""
    handler = Sentinel2Handler(max_cloud_cover=10.0)
    scenes = [
        {"scene_id": "scene_a", "cloud_cover": 5.0},
        {"scene_id": "scene_b", "cloud_cover": 15.0},
        {"scene_id": "scene_c", "cloud_cover": 9.9},
    ]

    filtered = handler.filter_scenes(scenes)
    assert len(filtered) == 2
    assert set(sc["scene_id"] for sc in filtered) == {"scene_a", "scene_c"}


def test_download_sentinel_cli_parser():
    """download_sentinel.py CLI parser should parse custom flags correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "--aoi", "data/aois/mumbai.geojson",
        "--cloud-cover", "15.0",
        "--output-dir", "data/sentinel",
        "--dry-run"
    ])

    assert args.aoi == "data/aois/mumbai.geojson"
    assert args.cloud_cover == 15.0
    assert args.output_dir == "data/sentinel"
    assert args.dry_run is True
