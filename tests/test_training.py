"""
Unit tests for SpaceNetra training engine and CLI parser.
"""

import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.training.trainer import ChangeDetectionTrainer
from train import build_parser, load_config


def test_trainer_one_epoch_dry_run():
    """Trainer should run 1 epoch on synthetic data and return complete metrics dict."""
    t1 = torch.randn(4, 3, 128, 128)
    t2 = torch.randn(4, 3, 128, 128)
    target = torch.randint(0, 2, (4, 1, 128, 128)).float()

    dataset = TensorDataset(t1, t2, target)
    loader = DataLoader(dataset, batch_size=2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ChangeDetectionTrainer(device="cpu", checkpoint_dir=tmp_dir)
        metrics = trainer.fit(loader, loader, epochs=1)

        assert "train_loss" in metrics
        assert "val_loss" in metrics
        assert "val_f1" in metrics
        assert "val_iou" in metrics
        assert "val_precision" in metrics
        assert "val_recall" in metrics
        assert metrics["train_loss"] > 0.0


def test_trainer_checkpoint_saving():
    """Trainer should create best_model.pth when val_f1 improves."""
    t1 = torch.randn(2, 3, 128, 128)
    t2 = torch.randn(2, 3, 128, 128)
    target = torch.ones((2, 1, 128, 128))

    dataset = TensorDataset(t1, t2, target)
    loader = DataLoader(dataset, batch_size=2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ChangeDetectionTrainer(device="cpu", checkpoint_dir=tmp_dir)
        trainer.fit(loader, loader, epochs=1)

        checkpoint_file = Path(tmp_dir) / "best_model.pth"
        assert checkpoint_file.exists(), f"Checkpoint file not found at {checkpoint_file}"

        ckpt = torch.load(checkpoint_file, weights_only=True)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "best_f1" in ckpt


def test_train_cli_parser():
    """CLI argument parser should parse custom flags correctly."""
    parser = build_parser()
    args = parser.parse_args(["--config", "configs/baseline.yaml", "--epochs", "10", "--batch-size", "16", "--dry-run"])

    assert args.config == "configs/baseline.yaml"
    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.dry_run is True


def test_load_config():
    """Config loader should read existing baseline.yaml file correctly."""
    config = load_config("configs/baseline.yaml")
    assert "model" in config
    assert "data" in config
    assert "training" in config
    assert config["model"]["architecture"] == "siamese_unet"
