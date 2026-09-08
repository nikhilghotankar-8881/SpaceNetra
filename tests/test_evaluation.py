"""
Unit tests for SpaceNetra evaluation pipeline and CLI parser.
"""

import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.models.siamese_unet import SiameseUNet
from src.evaluation.evaluator import ModelEvaluator
from evaluate import build_parser


def test_evaluator_runs_inference():
    """ModelEvaluator.evaluate should process test dataloader and return complete metrics dict."""
    model = SiameseUNet(backbone="resnet18", pretrained=False)
    evaluator = ModelEvaluator(model=model, device="cpu")

    t1 = torch.randn(4, 3, 128, 128)
    t2 = torch.randn(4, 3, 128, 128)
    target = torch.randint(0, 2, (4, 1, 128, 128)).float()

    dataset = TensorDataset(t1, t2, target)
    loader = DataLoader(dataset, batch_size=2)

    results = evaluator.evaluate(loader)

    assert "f1" in results
    assert "precision" in results
    assert "recall" in results
    assert "iou" in results
    assert "overall_accuracy" in results
    assert "kappa" in results
    assert "ms_per_tile" in results
    assert results["total_tiles"] == 4.0


def test_evaluator_report_generation():
    """generate_report should export evaluation_report.txt and evaluation_metrics.json."""
    model = SiameseUNet(backbone="resnet18", pretrained=False)
    evaluator = ModelEvaluator(model=model, device="cpu")

    sample_metrics = {
        "precision": 0.88,
        "recall": 0.84,
        "f1": 0.86,
        "iou": 0.77,
        "overall_accuracy": 0.96,
        "kappa": 0.82,
        "ms_per_tile": 15.4,
        "total_tiles": 16.0,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        txt_path, json_path = evaluator.generate_report(sample_metrics, output_dir=tmp_dir)

        assert txt_path.exists()
        assert json_path.exists()

        content = txt_path.read_text(encoding="utf-8")
        assert "MODEL EVALUATION REPORT" in content
        assert "PASSED [OK]" in content


def test_evaluate_cli_parser():
    """evaluate.py CLI parser should parse custom flags correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "--checkpoint", "checkpoints/best_model.pth",
        "--config", "configs/baseline.yaml",
        "--output-dir", "outputs/metrics",
        "--dry-run"
    ])

    assert args.checkpoint == "checkpoints/best_model.pth"
    assert args.config == "configs/baseline.yaml"
    assert args.output-dir == "outputs/metrics" if hasattr(args, "output-dir") else args.output_dir == "outputs/metrics"
    assert args.dry_run is True
