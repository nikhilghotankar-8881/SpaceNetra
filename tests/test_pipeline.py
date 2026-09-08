"""
Unit Tests for Sentinel-2 End-to-End Change Detection Pipeline Engine.
"""

from pathlib import Path
import numpy as np
import pytest
from predict_sentinel import build_parser, load_config
from src.ingestion.geotiff import GeoTIFFHandler
from src.pipeline.sentinel_cd import SentinelChangeDetector


def test_sentinel_change_detector_synthetic_pair():
    detector = SentinelChangeDetector(model_or_arch="siamese_unet", tile_size=256)

    # Synthetic 512x512 RGB scenes (raw DN range 0..10000)
    t1 = np.random.randint(500, 5000, (512, 512, 3), dtype=np.uint16)
    t2 = np.random.randint(500, 5000, (512, 512, 3), dtype=np.uint16)
    scl1 = np.zeros((512, 512), dtype=np.uint8)
    scl2 = np.zeros((512, 512), dtype=np.uint8)

    results = detector.predict_pair(
        img_t1=t1,
        img_t2=t2,
        scl_t1=scl1,
        scl_t2=scl2,
        threshold=0.5,
    )

    assert "probability_map" in results
    assert "change_mask" in results

    prob_map = results["probability_map"]
    change_mask = results["change_mask"]

    assert prob_map.shape == (512, 512)
    assert change_mask.shape == (512, 512)
    assert prob_map.dtype == np.float32
    assert change_mask.dtype == np.uint8
    assert np.all((change_mask == 0) | (change_mask == 255))


def test_sentinel_change_detector_changeformer_arch():
    detector = SentinelChangeDetector(model_or_arch="changeformer", tile_size=256)

    t1 = np.random.randint(500, 5000, (256, 256, 3), dtype=np.uint16)
    t2 = np.random.randint(500, 5000, (256, 256, 3), dtype=np.uint16)

    results = detector.predict_pair(img_t1=t1, img_t2=t2)
    assert results["probability_map"].shape == (256, 256)


def test_predict_sentinel_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--dry-run", "--threshold", "0.6"])

    assert args.dry_run is True
    assert args.threshold == 0.6


def test_sentinel_change_detector_geotiff_roundtrip(tmp_path: Path):
    t1_path = str(tmp_path / "t1.tif")
    t2_path = str(tmp_path / "t2.tif")
    output_path = str(tmp_path / "output_change.tif")

    # Create dummy GeoTIFFs
    data_t1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    data_t2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    GeoTIFFHandler.write_raster(t1_path, data_t1)
    GeoTIFFHandler.write_raster(t2_path, data_t2)

    detector = SentinelChangeDetector(model_or_arch="siamese_unet", tile_size=256)
    out_file = detector.predict_geotiff_pair(
        t1_path=t1_path,
        t2_path=t2_path,
        output_path=output_path,
    )

    assert Path(out_file).exists()

    # Read back generated GeoTIFF
    read_data, meta = GeoTIFFHandler.read_raster(out_file)
    assert read_data.shape == (256, 256)
