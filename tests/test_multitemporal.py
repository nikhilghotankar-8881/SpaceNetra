"""
Unit Tests for Multi-Temporal Analysis Engine & Change Chronology Tracking.
"""

from pathlib import Path
import numpy as np
import pytest
from analyze_series import build_parser
from src.temporal.multitemporal import MultiTemporalAnalyzer


def test_multitemporal_analyzer_synthetic_stack():
    analyzer = MultiTemporalAnalyzer(model_or_arch="siamese_unet")

    # Stack of 4 synthetic scenes
    scenes = [
        np.random.randint(500, 4000, (256, 256, 3), dtype=np.uint16)
        for _ in range(4)
    ]
    timestamps = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]

    results = analyzer.analyze_series(
        scene_list=scenes,
        timestamps=timestamps,
        threshold=0.5,
    )

    assert "step_transitions" in results
    assert "baseline_transitions" in results
    assert "cumulative_frequency" in results
    assert "first_change_onset" in results

    step_trans = results["step_transitions"]
    baseline_trans = results["baseline_transitions"]
    cum_freq = results["cumulative_frequency"]
    first_onset = results["first_change_onset"]

    assert len(step_trans) == 3
    assert len(baseline_trans) == 3
    assert cum_freq.shape == (256, 256)
    assert first_onset.shape == (256, 256)
    assert cum_freq.dtype == np.uint16
    assert first_onset.dtype == np.uint8


def test_multitemporal_analyzer_invalid_stack():
    analyzer = MultiTemporalAnalyzer(model_or_arch="siamese_unet")
    scenes = [np.ones((256, 256, 3), dtype=np.uint8)]  # Only 1 scene

    with pytest.raises(ValueError):
        analyzer.analyze_series(scenes)


def test_analyze_series_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--dry-run", "--threshold", "0.7", "--output-dir", "outputs/test_series"])

    assert args.dry_run is True
    assert args.threshold == 0.7
    assert args.output-dir if hasattr(args, "output-dir") else args.output_dir == "outputs/test_series"
