"""
Unit Tests for Confidence Scoring & Alignment Quality Engine.
"""

import numpy as np
import pytest
from score_quality import build_parser
from src.confidence.quality_engine import ConfidenceEngine


def test_confidence_engine_alignment_score():
    engine = ConfidenceEngine()

    t1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    t2 = np.full((100, 100, 3), 100, dtype=np.uint8)

    # Identical images must yield perfect alignment score 1.0
    score = engine.compute_alignment_quality(t1, t2)
    assert np.isclose(score, 1.0)

    # Mismatched shapes must raise ValueError
    t3 = np.full((50, 50, 3), 100, dtype=np.uint8)
    with pytest.raises(ValueError):
        engine.compute_alignment_quality(t1, t3)


def test_confidence_engine_cloud_contamination():
    engine = ConfidenceEngine()

    scl1 = np.zeros((10, 10), dtype=np.uint8)
    scl1[0, 0] = 9  # High prob cloud (1/100 pixels = 1%)

    cloud_ratio = engine.compute_cloud_contamination(scl_t1=scl1)
    assert np.isclose(cloud_ratio, 0.01)


def test_confidence_engine_prediction_confidence():
    engine = ConfidenceEngine()

    # Confident probability map (all 0s or 1s) -> Certainty 1.0
    prob_high = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert np.isclose(engine.compute_prediction_confidence(prob_high), 1.0)

    # Uncertain probability map (all 0.5s) -> Certainty 0.0
    prob_uncertain = np.full((2, 2), 0.5, dtype=np.float32)
    assert np.isclose(engine.compute_prediction_confidence(prob_uncertain), 0.0)


def test_confidence_engine_evaluate_quality():
    engine = ConfidenceEngine()

    t1 = np.full((64, 64, 3), 120, dtype=np.uint8)
    t2 = np.full((64, 64, 3), 120, dtype=np.uint8)
    prob_map = np.ones((64, 64), dtype=np.float32)

    report = engine.evaluate_quality(img_t1=t1, img_t2=t2, prob_map=prob_map)

    assert "composite_confidence" in report
    assert "status" in report
    assert report["status"] == "HIGH_CONFIDENCE"
    assert np.isclose(report["composite_confidence"], 1.0)


def test_score_quality_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--dry-run", "--t1", "scene1.tif", "--t2", "scene2.tif"])

    assert args.dry_run is True
    assert args.t1 == "scene1.tif"
    assert args.t2 == "scene2.tif"
