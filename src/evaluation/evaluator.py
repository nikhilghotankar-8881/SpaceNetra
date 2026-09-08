"""
Evaluation pipeline and ModelEvaluator class for SpaceNetra models.

Handles full dataset evaluation on test sets, latency benchmarks (ms/tile),
and exports formatted evaluation reports to text and JSON formats.
"""

import json
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import ChangeDetectionMetrics


class ModelEvaluator:
    """
    Evaluator for running inference benchmarks and metric computations on test sets.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        threshold: float = 0.5,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.threshold = threshold
        self.metrics_calculator = ChangeDetectionMetrics(threshold=threshold)

    def _parse_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parses batch whether dictionary (from LEVIRCDDataset) or tuple."""
        if isinstance(batch, dict):
            t1 = batch["img_a"]
            t2 = batch["img_b"]
            target = batch["mask"]
        elif isinstance(batch, (list, tuple)):
            t1, t2, target = batch[:3]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        return t1, t2, target

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Runs evaluation loop over dataloader and calculates metrics + latency benchmarks.

        Args:
            dataloader: PyTorch DataLoader for evaluation dataset.

        Returns:
            Dictionary of metrics and benchmark statistics.
        """
        self.model.eval()
        self.metrics_calculator.reset()

        total_tiles = 0
        total_time_sec = 0.0

        with torch.no_grad():
            for batch in dataloader:
                t1, t2, target = self._parse_batch(batch)
                t1 = t1.to(self.device)
                t2 = t2.to(self.device)
                target = target.to(self.device)

                batch_size = t1.size(0)
                total_tiles += batch_size

                start_time = time.perf_counter()
                preds = self.model(t1, t2, return_logits=False)
                end_time = time.perf_counter()

                total_time_sec += (end_time - start_time)
                self.metrics_calculator.update(preds, target)

        results = self.metrics_calculator.compute()
        ms_per_tile = (total_time_sec / max(total_tiles, 1)) * 1000.0

        results["ms_per_tile"] = float(ms_per_tile)
        results["total_tiles"] = float(total_tiles)
        return results

    def generate_report(
        self,
        metrics: Dict[str, float],
        output_dir: Union[str, Path] = "outputs/metrics",
        checkpoint_name: str = "best_model.pth",
    ) -> Tuple[Path, Path]:
        """
        Formats and saves evaluation metrics to evaluation_report.txt and evaluation_metrics.json.

        Args:
            metrics: Dictionary returned by evaluate().
            output_dir: Target directory for report files.
            checkpoint_name: Name of evaluated checkpoint.

        Returns:
            Tuple of (txt_report_path, json_metrics_path)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        txt_file = out_path / "evaluation_report.txt"
        json_file = out_path / "evaluation_metrics.json"

        # Benchmarks & Targets from EVALUATION.md
        targets = {
            "precision": (0.85, "> 85.0%"),
            "recall": (0.83, "> 83.0%"),
            "f1": (0.85, "> 85.0%"),
            "iou": (0.75, "> 75.0%"),
            "overall_accuracy": (0.95, "> 95.0%"),
            "kappa": (0.80, "> 0.800"),
        }

        report_lines = [
            "==================================================================",
            "                   MODEL EVALUATION REPORT                        ",
            "==================================================================",
            f"Checkpoint:  {checkpoint_name}",
            f"Device:      {self.device.type.upper()}",
            f"Tiles Tested:{int(metrics.get('total_tiles', 0))}",
            f"Latency:     {metrics.get('ms_per_tile', 0.0):.2f} ms / tile",
            "------------------------------------------------------------------",
            "METRIC              VALUE      TARGET      STATUS",
            "------------------------------------------------------------------",
        ]

        for key, (target_val, target_str) in targets.items():
            val = metrics.get(key, 0.0)
            if key in ("kappa",):
                val_str = f"{val:.4f}"
                passed = val >= target_val
            else:
                val_str = f"{val * 100:.2f}%"
                passed = val >= target_val
            status = "PASSED [OK]" if passed else "BELOW TARGET"
            report_lines.append(f"{key.upper():<18} {val_str:<10} {target_str:<11} {status}")

        report_lines.extend([
            "==================================================================",
        ])

        txt_content = "\n".join(report_lines)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(txt_content)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        return txt_file, json_file
