"""
Data Visualization Script for SpaceNetra (LEVIR-CD Dataset).
Generates visual sample grids (T1 | T2 | Mask | Overlay) and change ratio distributions.
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Add root directory to path for src import
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import config


def create_overlay(img_b_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Creates a red overlay on image B where the change mask is non-zero (255).
    """
    overlay = img_b_np.copy()
    if len(overlay.shape) == 2:  # Grayscale to RGB
        overlay = np.stack([overlay] * 3, axis=-1)

    # Red mask highlight
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = 255  # Red channel

    mask_binary = (mask_np > 0)
    overlay[mask_binary] = (
        (1 - alpha) * overlay[mask_binary] + alpha * red_mask[mask_binary]
    ).astype(np.uint8)

    return overlay


def visualize_samples(data_dir: Path, output_path: Path, num_samples_per_split: int = 3):
    """
    Generates a grid of visual samples: Split | T1 (Image A) | T2 (Image B) | Ground Truth Mask | Red Overlay
    """
    splits = ["train", "val", "test"]
    fig_rows = len(splits) * num_samples_per_split
    fig_cols = 4  # T1, T2, Mask, Overlay

    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(16, 4 * fig_rows))
    fig.suptitle("SpaceNetra - LEVIR-CD Data Inspection Samples", fontsize=18, fontweight="bold", y=0.995)

    row_idx = 0

    for split in splits:
        split_dir = data_dir / split
        a_dir = split_dir / "A"
        b_dir = split_dir / "B"
        label_dir = split_dir / "label"

        a_files = sorted(list(a_dir.glob("*.png"))) if a_dir.exists() else []

        if not a_files:
            continue

        # Select samples deterministically or randomly
        selected_files = a_files[:num_samples_per_split] if len(a_files) <= num_samples_per_split else random.sample(a_files, num_samples_per_split)

        for a_path in selected_files:
            filename = a_path.name
            b_path = b_dir / filename
            label_path = label_dir / filename

            img_a = np.array(Image.open(a_path))
            img_b = np.array(Image.open(b_path))
            img_label = np.array(Image.open(label_path))

            overlay = create_overlay(img_b, img_label)

            # Col 0: Image A (T1)
            axes[row_idx, 0].imshow(img_a)
            axes[row_idx, 0].set_title(f"[{split.upper()}] T1 (Pre) - {filename}", fontsize=10)
            axes[row_idx, 0].axis("off")

            # Col 1: Image B (T2)
            axes[row_idx, 1].imshow(img_b)
            axes[row_idx, 1].set_title(f"[{split.upper()}] T2 (Post) - {filename}", fontsize=10)
            axes[row_idx, 1].axis("off")

            # Col 2: Label Mask
            axes[row_idx, 2].imshow(img_label, cmap="gray", vmin=0, vmax=255)
            axes[row_idx, 2].set_title(f"[{split.upper()}] Change Mask", fontsize=10)
            axes[row_idx, 2].axis("off")

            # Col 3: Red Overlay
            axes[row_idx, 3].imshow(overlay)
            axes[row_idx, 3].set_title(f"[{split.upper()}] T2 + Change Overlay", fontsize=10)
            axes[row_idx, 3].axis("off")

            row_idx += 1

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved dataset sample visualization to: {output_path}")


def visualize_change_distribution(data_dir: Path, output_path: Path):
    """
    Plots the histogram/distribution of change pixel percentages per split.
    """
    splits = ["train", "val", "test"]
    split_ratios = {}

    for split in splits:
        label_dir = data_dir / split / "label"
        ratios = []
        if label_dir.exists():
            for label_path in label_dir.glob("*.png"):
                mask_np = np.array(Image.open(label_path))
                change_pct = (np.count_nonzero(mask_np > 0) / mask_np.size) * 100.0
                ratios.append(change_pct)
        split_ratios[split] = ratios

    plt.figure(figsize=(10, 6))

    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

    for split in splits:
        ratios = split_ratios[split]
        if ratios:
            plt.hist(ratios, bins=20, alpha=0.6, label=f"{split.upper()} (n={len(ratios)})", color=colors[split])

    plt.title("LEVIR-CD Change Pixel % Distribution Across Splits", fontsize=14, fontweight="bold")
    plt.xlabel("Change Pixel Percentage (%)", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Saved change ratio distribution plot to: {output_path}")


def main():
    print("=" * 60)
    print("[*] SpaceNetra - Data Visualizer Pipeline")
    print("=" * 60)

    data_dir = config.paths.levir_cd_dir
    samples_out = config.paths.visualizations_dir / "data_samples.png"
    dist_out = config.paths.visualizations_dir / "change_ratio_distribution.png"

    if not data_dir.exists():
        print(f"[-] ERROR: Data directory does not exist: {data_dir}")
        sys.exit(1)

    random.seed(42)
    visualize_samples(data_dir, samples_out, num_samples_per_split=3)
    visualize_change_distribution(data_dir, dist_out)

    print("\n[OK] Data visualization complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
