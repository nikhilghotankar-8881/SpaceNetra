"""
Unit Tests for Spatial Change Event Builder & GeoJSON Exporter Engine.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from build_events import build_parser
from src.temporal.event_builder import EventBuilder


def test_event_builder_spatial_clustering():
    builder = EventBuilder(min_area_pixels=10, pixel_resolution_m=10.0)

    # 100x100 mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255  # 10x10 = 100 pixels (10,000 m²)
    mask[50:60, 50:70] = 255  # 10x20 = 200 pixels (20,000 m²)
    mask[0:2, 0:2] = 255  # 2x2 = 4 pixels (should be filtered out < min_area_pixels=10)

    events = builder.extract_events(mask)

    assert len(events) == 2
    assert events[0]["event_id"] == "EVENT_0001"
    assert events[0]["pixel_count"] == 100
    assert np.isclose(events[0]["area_m2"], 10000.0)

    assert events[1]["pixel_count"] == 200
    assert np.isclose(events[1]["area_m2"], 20000.0)


def test_event_builder_geojson_export(tmp_path: Path):
    builder = EventBuilder(min_area_pixels=10, pixel_resolution_m=10.0)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255

    events = builder.extract_events(mask)
    out_path = tmp_path / "test_events.geojson"

    out_file = builder.export_geojson(events, out_path)
    assert Path(out_file).exists()

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    feat = data["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Polygon"
    assert feat["properties"]["event_id"] == "EVENT_0001"


def test_build_events_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--dry-run", "--min-area", "15", "--resolution", "20.0"])

    assert args.dry_run is True
    assert args.min_area == 15
    assert args.resolution == 20.0
