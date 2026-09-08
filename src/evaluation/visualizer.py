"""
Visual evaluation and error map visualization utilities for SpaceNetra.

Generates:
1. 4-Panel Comparison Grids (T1 | T2 | Ground Truth | Prediction).
2. Color-Coded Error Maps (TP=Green, FP=Red, FN=Blue, TN=Black).
3. Continuous Confidence Heatmaps.
"""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ChangeVisualizer:
    """
    Visualizer class for rendering diagnostic maps and evaluation grids.
    """

    @staticmethod
    def plot_comparison_grid(
        t1: np.ndarray,
        t2: np.ndarray,
        gt: np.ndarray,
        pred: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Creates a 4-panel side-by-side comparison figure: T1 | T2 | Ground Truth | Prediction.

        Args:
            t1: T1 RGB image array (H, W, 3) in range [0, 255] or [0.0, 1.0].
            t2: T2 RGB image array (H, W, 3) in range [0, 255] or [0.0, 1.0].
            gt: Binary ground truth mask (H, W) in range {0, 1} or {0, 255}.
            pred: Binary predicted mask or probability map (H, W) in range [0.0, 1.0].
            save_path: Optional output file path for saving PNG figure.

        Returns:
            RGB numpy array of rendered figure.
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Normalize display ranges
        t1_disp = (t1 * 255).astype(np.uint8) if t1.max() <= 1.0 else t1.astype(np.uint8)
        t2_disp = (t2 * 255).astype(np.uint8) if t2.max() <= 1.0 else t2.astype(np.uint8)

        gt_disp = (gt > 0).astype(np.uint8)
        pred_disp = (pred >= 0.5).astype(np.uint8) if pred.max() <= 1.0 else (pred > 0).astype(np.uint8)

        axes[0].imshow(t1_disp)
        axes[0].set_title("T1 (Pre-Change)", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(t2_disp)
        axes[1].set_title("T2 (Post-Change)", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(gt_disp, cmap="gray")
        axes[2].set_title("Ground Truth", fontsize=12)
        axes[2].axis("off")

        axes[3].imshow(pred_disp, cmap="gray")
        axes[3].set_title("Predicted Mask", fontsize=12)
        axes[3].axis("off")

        plt.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, bbox_inches="tight", dpi=150)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return rgba[:, :, :3]

    @staticmethod
    def plot_error_map(
        gt: np.ndarray,
        pred: np.ndarray,
        threshold: float = 0.5,
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Generates color-coded error map overlay:
        - True Positives (TP): Green [0, 255, 0]
        - False Positives (FP): Red [255, 0, 0]
        - False Negatives (FN): Blue [0, 0, 255]
        - True Negatives (TN): Black [0, 0, 0]

        Args:
            gt: Ground truth mask (H, W) in range {0, 1} or {0, 255}.
            pred: Prediction mask or probability map (H, W).
            threshold: Binary threshold for predictions.
            save_path: Optional output file path.

        Returns:
            RGB array of shape (H, W, 3).
        """
        b_gt = (gt > 0).astype(bool)
        b_pred = (pred >= threshold).astype(bool) if pred.max() <= 1.0 else (pred > 0).astype(bool)

        tp = b_gt & b_pred
        fp = (~b_gt) & b_pred
        fn = b_gt & (~b_pred)

        h, w = gt.shape[:2]
        error_map = np.zeros((h, w, 3), dtype=np.uint8)

        error_map[tp] = [0, 255, 0]    # Green (True Positive)
        error_map[fp] = [255, 0, 0]    # Red (False Positive)
        error_map[fn] = [0, 0, 255]    # Blue (False Negative)

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(error_map).save(path)

        return error_map

    @staticmethod
    def plot_confidence_heatmap(
        prob_map: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Renders continuous colormap confidence heatmap for probability predictions.

        Args:
            prob_map: Continuous probability map (H, W) in range [0.0, 1.0].
            save_path: Optional output file path.

        Returns:
            RGB numpy array of rendered heatmap.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(prob_map, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title("Change Probability Heatmap", fontsize=12)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, bbox_inches="tight", dpi=150)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return rgba[:, :, :3]
