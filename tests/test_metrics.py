"""
Unit tests for SpaceNetra evaluation metrics engine.
"""

import torch
import pytest

from src.evaluation.metrics import ChangeDetectionMetrics


def test_metrics_hand_computed_values():
    """Known 2x2 prediction/target tensors must produce exact hand-calculated metric scores."""
    metrics = ChangeDetectionMetrics(threshold=0.5)

    # TP=2, FP=1, FN=0, TN=1
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    preds = torch.tensor([[[[0.9, 0.9], [0.8, 0.1]]]])

    metrics.update(preds, target)
    results = metrics.compute()

    assert results["precision"] == pytest.approx(2 / 3, abs=1e-5)
    assert results["recall"] == pytest.approx(1.0, abs=1e-5)
    assert results["f1"] == pytest.approx(0.8, abs=1e-5)
    assert results["iou"] == pytest.approx(2 / 3, abs=1e-5)
    assert results["overall_accuracy"] == pytest.approx(0.75, abs=1e-5)


def test_metrics_accumulator_reset():
    """reset() should clear all accumulated counts back to 0."""
    metrics = ChangeDetectionMetrics()
    target = torch.ones((1, 1, 4, 4))
    preds = torch.ones((1, 1, 4, 4))

    metrics.update(preds, target)
    assert metrics.tp == 16.0

    metrics.reset()
    assert metrics.tp == 0.0
    assert metrics.fp == 0.0
    assert metrics.fn == 0.0
    assert metrics.tn == 0.0


def test_metrics_thresholding():
    """Varying threshold from 0.5 to 0.8 must alter confusion matrix counts correctly."""
    metrics_low = ChangeDetectionMetrics(threshold=0.5)
    metrics_high = ChangeDetectionMetrics(threshold=0.8)

    preds = torch.tensor([[[[0.6, 0.7], [0.8, 0.9]]]])
    target = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])

    metrics_low.update(preds, target)
    metrics_high.update(preds, target)

    res_low = metrics_low.compute()
    res_high = metrics_high.compute()

    # At 0.5 threshold, 0.6 and 0.7 become False Positives
    assert res_low["precision"] == pytest.approx(2 / 4, abs=1e-5)
    # At 0.8 threshold, 0.6 and 0.7 are filtered out -> Precision = 2 / 2 = 1.0
    assert res_high["precision"] == pytest.approx(1.0, abs=1e-5)


def test_metrics_zero_change_edge_case():
    """All-zero ground truth targets must evaluate without division-by-zero or NaN outputs."""
    metrics = ChangeDetectionMetrics()
    preds = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
    target = torch.zeros((1, 1, 2, 2))

    metrics.update(preds, target)
    results = metrics.compute()

    assert not torch.isnan(torch.tensor(results["f1"]))
    assert not torch.isnan(torch.tensor(results["kappa"]))
    assert results["precision"] == pytest.approx(0.0, abs=1e-5)
    assert results["recall"] == pytest.approx(0.0, abs=1e-5)
    assert results["overall_accuracy"] == pytest.approx(1.0, abs=1e-5)


def test_metrics_kappa_coefficient():
    """Cohen's Kappa score should reflect high agreement for perfect predictions."""
    metrics = ChangeDetectionMetrics()
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    preds = torch.tensor([[[[0.9, 0.1], [0.1, 0.9]]]])

    metrics.update(preds, target)
    results = metrics.compute()

    assert results["kappa"] == pytest.approx(1.0, abs=1e-5)
