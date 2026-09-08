"""
Unit tests for SpaceNetra training callbacks (EarlyStopping and ModelCheckpoint).
"""

import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.trainer import ChangeDetectionTrainer


def test_early_stopping_trigger():
    """EarlyStopping should flag should_stop=True after patience non-improving steps."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="max")

    assert es.step(0.50) is False
    assert es.step(0.50) is False  # counter = 1
    assert es.step(0.50) is False  # counter = 2
    assert es.step(0.50) is True   # counter = 3 -> trigger!


def test_early_stopping_reset_on_improvement():
    """EarlyStopping should reset counter to 0 when metric improves by > min_delta."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="max")

    es.step(0.50)
    es.step(0.50)  # counter = 1
    assert es.counter == 1

    # Improvement > 0.01
    es.step(0.55)
    assert es.counter == 0
    assert es.best_score == 0.55


def test_model_checkpoint_saves_best_and_last():
    """ModelCheckpoint should save best_model.pth and last_model.pth."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt = ModelCheckpoint(save_dir=tmp_dir, monitor="val_f1", mode="max")

        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        is_best, path = ckpt.step({"val_f1": 0.85}, epoch=1, model=model, optimizer=optimizer)
        assert is_best is True
        assert path == Path(tmp_dir) / "best_model.pth"
        assert (Path(tmp_dir) / "best_model.pth").exists()
        assert (Path(tmp_dir) / "last_model.pth").exists()


def test_trainer_early_stopping_integration():
    """Trainer fit loop should terminate early when EarlyStopping triggers."""
    t1 = torch.randn(4, 3, 32, 32)
    t2 = torch.randn(4, 3, 32, 32)
    target = torch.ones((4, 1, 32, 32))

    dataset = TensorDataset(t1, t2, target)
    loader = DataLoader(dataset, batch_size=2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ChangeDetectionTrainer(device="cpu", checkpoint_dir=tmp_dir)
        es = EarlyStopping(patience=2, min_delta=0.1, mode="max")

        # Request 10 epochs, but patience=2 should stop after ~3 epochs since targets are static
        metrics = trainer.fit(loader, loader, epochs=10, early_stopping=es)
        assert es.should_stop is True
