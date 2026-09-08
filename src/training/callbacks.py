"""
Training callbacks for SpaceNetra models.

Includes:
- EarlyStopping: Monitors validation metrics to halt training when performance plateaus.
- ModelCheckpoint: Saves best-performing and latest model weights and optimizer states.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Halts training when a monitored validation metric stops improving.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        if self.mode not in ("max", "min"):
            raise ValueError(f"Mode must be 'max' or 'min', got '{mode}'")

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """
        Updates early stopping state based on current metric value.

        Args:
            value: Monitored metric score (e.g. val_f1 or val_loss).

        Returns:
            True if early stopping criterion is met, False otherwise.
        """
        if self.best_score is None:
            self.best_score = value
            return False

        if self.mode == "max":
            improved = value > (self.best_score + self.min_delta)
        else:
            improved = value < (self.best_score - self.min_delta)

        if improved:
            self.best_score = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """
    Saves model checkpoints during training based on validation metrics.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        monitor: str = "val_f1",
        mode: str = "max",
        save_best: bool = True,
        save_last: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode.lower()
        if self.mode not in ("max", "min"):
            raise ValueError(f"Mode must be 'max' or 'min', got '{mode}'")

        self.save_best = save_best
        self.save_last = save_last
        self.best_score: Optional[float] = None

    def step(
        self,
        metrics: Dict[str, float],
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[bool, Optional[Path]]:
        """
        Evaluates metrics and saves model checkpoints if conditions are met.

        Args:
            metrics: Dictionary of metric scores for current epoch.
            epoch: Current epoch number.
            model: PyTorch model instance.
            optimizer: PyTorch optimizer instance.

        Returns:
            (is_best, best_checkpoint_path)
        """
        val_score = metrics.get(self.monitor, 0.0)
        is_best = False
        best_path: Optional[Path] = None

        if self.best_score is None:
            is_best = True
        elif self.mode == "max" and val_score > self.best_score:
            is_best = True
        elif self.mode == "min" and val_score < self.best_score:
            is_best = True

        state_dict: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_score": val_score if is_best else self.best_score,
        }

        if is_best:
            self.best_score = val_score
            if self.save_best:
                best_path = self.save_dir / "best_model.pth"
                torch.save(state_dict, best_path)

        if self.save_last:
            last_path = self.save_dir / "last_model.pth"
            torch.save(state_dict, last_path)

        return is_best, best_path
