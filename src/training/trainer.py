"""
Training engine and trainer class for SpaceNetra Change Detection models.

Handles model training, validation, metrics computation (Precision, Recall, F1, IoU),
learning rate scheduling, and checkpoint management.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.siamese_unet import SiameseUNet
from src.models.losses import BCEDiceLoss


class ChangeDetectionTrainer:
    """
    Trainer for Siamese U-Net Change Detection models.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model if model is not None else SiameseUNet(backbone="resnet18", pretrained=False)
        self.model.to(self.device)

        self.criterion = criterion if criterion is not None else BCEDiceLoss()
        self.criterion.to(self.device)

        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        )

        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_f1 = -1.0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Runs one epoch of training over dataloader."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            t1, t2, target = batch
            t1 = t1.to(self.device)
            t2 = t2.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(t1, t2, return_logits=True)
            loss = self.criterion(logits, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(len(dataloader), 1)

    def validate(self, dataloader: DataLoader, threshold: float = 0.5) -> Dict[str, float]:
        """Runs validation evaluation over dataloader and computes metrics."""
        self.model.eval()
        total_loss = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        eps = 1e-7

        with torch.no_grad():
            for batch in dataloader:
                t1, t2, target = batch
                t1 = t1.to(self.device)
                t2 = t2.to(self.device)
                target = target.to(self.device)

                logits = self.model(t1, t2, return_logits=True)
                loss = self.criterion(logits, target)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).float()
                target_b = target.float()

                total_tp += (preds * target_b).sum().item()
                total_fp += (preds * (1 - target_b)).sum().item()
                total_fn += ((1 - preds) * target_b).sum().item()

        precision = total_tp / (total_tp + total_fp + eps)
        recall = total_tp / (total_tp + total_fn + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        iou = total_tp / (total_tp + total_fp + total_fn + eps)
        avg_loss = total_loss / max(len(dataloader), 1)

        return {
            "val_loss": avg_loss,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_iou": iou,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5) -> Dict[str, float]:
        """Runs full training loop across specified number of epochs."""
        metrics = {}
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            metrics = val_metrics
            metrics["train_loss"] = train_loss

            val_f1 = val_metrics["val_f1"]
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.save_checkpoint("best_model.pth")

        return metrics

    def save_checkpoint(self, filename: str = "best_model.pth"):
        """Saves model weights to checkpoint directory."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_f1": self.best_f1,
            },
            checkpoint_path,
        )
