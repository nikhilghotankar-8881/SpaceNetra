"""
Evaluation metrics engine for SpaceNetra Change Detection models.

Computes exact, accumulated confusion matrix metrics across validation/testing splits:
Precision, Recall, F1 Score, IoU (Jaccard Index), Overall Accuracy, and Cohen's Kappa Coefficient.
"""

from typing import Dict, Union
import torch


class ChangeDetectionMetrics:
    """
    Accumulates confusion matrix counts across batch predictions and computes global metrics.
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        """Resets accumulated confusion matrix counters."""
        self.tp: float = 0.0
        self.fp: float = 0.0
        self.fn: float = 0.0
        self.tn: float = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Accumulates confusion matrix counts from batch predictions and targets.

        Args:
            preds: Predicted probability map or raw logits tensor.
            targets: Binary ground truth target tensor.
        """
        # If logits are provided (outside [0, 1] range), convert via Sigmoid
        if preds.min() < 0.0 or preds.max() > 1.0:
            probs = torch.sigmoid(preds)
        else:
            probs = preds

        binary_preds = (probs >= self.threshold).float()
        binary_targets = targets.float()

        self.tp += (binary_preds * binary_targets).sum().item()
        self.fp += (binary_preds * (1.0 - binary_targets)).sum().item()
        self.fn += ((1.0 - binary_preds) * binary_targets).sum().item()
        self.tn += ((1.0 - binary_preds) * (1.0 - binary_targets)).sum().item()

    def compute(self) -> Dict[str, float]:
        """
        Computes global evaluation metrics from accumulated confusion matrix.

        Returns:
            Dictionary containing: precision, recall, f1, iou, overall_accuracy, kappa.
        """
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = (2.0 * precision * recall) / (precision + recall + self.eps)
        iou = self.tp / (self.tp + self.fp + self.fn + self.eps)

        total = self.tp + self.fp + self.fn + self.tn
        if total == 0:
            oa = 0.0
            kappa = 0.0
        else:
            oa = (self.tp + self.tn) / total

            # Cohen's Kappa calculation
            p_e1 = ((self.tp + self.fp) * (self.tp + self.fn)) / (total * total)
            p_e0 = ((self.tn + self.fp) * (self.tn + self.fn)) / (total * total)
            p_e = p_e1 + p_e0

            if 1.0 - p_e == 0:
                kappa = 1.0
            else:
                kappa = (oa - p_e) / (1.0 - p_e + self.eps)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "overall_accuracy": float(oa),
            "kappa": float(kappa),
        }
