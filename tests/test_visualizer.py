"""
Unit tests for SpaceNetra visualizer module and inference CLI parser.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.evaluation.visualizer import ChangeVisualizer
from inference import build_parser


def test_visualizer_error_map_rgb_encoding():
    """plot_error_map should encode TP=Green, FP=Red, FN=Blue, and TN=Black."""
    gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    pred = np.array([[1, 1], [0, 0]], dtype=np.float32)

    # (0,0): GT=1, Pred=1 -> TP (Green)
    # (0,1): GT=0, Pred=1 -> FP (Red)
    # (1,0): GT=1, Pred=0 -> FN (Blue)
    # (1,1): GT=0, Pred=0 -> TN (Black)

    error_map = ChangeVisualizer.plot_error_map(gt, pred, threshold=0.5)

    assert np.array_equal(error_map[0, 0], [0, 255, 0])   # Green
    assert np.array_equal(error_map[0, 1], [255, 0, 0])   # Red
    assert np.array_equal(error_map[1, 0], [0, 0, 255])   # Blue
    assert np.array_equal(error_map[1, 1], [0, 0, 0])     # Black


def test_visualizer_saving_files():
    """Visualizer methods should export PNG image files cleanly."""
    t1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    t2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    gt = np.random.randint(0, 2, (64, 64), dtype=np.uint8)
    prob_map = np.random.uniform(0.0, 1.0, (64, 64)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        grid_path = Path(tmp_dir) / "grid.png"
        error_path = Path(tmp_dir) / "error.png"
        heatmap_path = Path(tmp_dir) / "heatmap.png"

        ChangeVisualizer.plot_comparison_grid(t1, t2, gt, prob_map, save_path=grid_path)
        ChangeVisualizer.plot_error_map(gt, prob_map, save_path=error_path)
        ChangeVisualizer.plot_confidence_heatmap(prob_map, save_path=heatmap_path)

        assert grid_path.exists()
        assert error_path.exists()
        assert heatmap_path.exists()


def test_inference_cli_parser():
    """inference.py CLI parser should parse options correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "--t1", "data/sample_t1.png",
        "--t2", "data/sample_t2.png",
        "--checkpoint", "checkpoints/best_model.pth",
        "--output", "outputs/visualizations",
        "--stride", "192",
        "--dry-run"
    ])

    assert args.t1 == "data/sample_t1.png"
    assert args.t2 == "data/sample_t2.png"
    assert args.checkpoint == "checkpoints/best_model.pth"
    assert args.output == "outputs/visualizations"
    assert args.stride == 192
    assert args.dry_run is True
