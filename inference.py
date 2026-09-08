"""
Standalone CLI inference tool for running bi-temporal change detection on satellite images.

Usage:
    python inference.py --t1 path/to/t1.png --t2 path/to/t2.png --checkpoint checkpoints/best_model.pth --output outputs/visualizations/
    python inference.py --dry-run
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.models.siamese_unet import SiameseUNet
from src.data.tiling import split_into_patches, stitch_patches
from src.evaluation.visualizer import ChangeVisualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceNetra Satellite Change Detection Inference CLI")
    parser.add_argument("--t1", type=str, default=None, help="Path to T1 (pre-change) image file")
    parser.add_argument("--t2", type=str, default=None, help="Path to T2 (post-change) image file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained PyTorch model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/visualizations",
        help="Directory to save predicted change mask and visualization plots",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Tiling stride in pixels (256 for fast non-overlapping, 192 for smooth overlapping)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference on synthetic 256x256 image pair to verify pipeline without files",
    )
    return parser


def run_inference(
    img_a: np.ndarray,
    img_b: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    stride: int = 256,
) -> np.ndarray:
    """Runs sliding-window inference on arbitrary HxW image pair."""
    h, w = img_a.shape[:2]
    patches_a, positions, padded_size = split_into_patches(img_a, patch_size=256, stride=stride)
    patches_b, _, _ = split_into_patches(img_b, patch_size=256, stride=stride)

    predicted_patches = []

    model.eval()
    with torch.no_grad():
        for pa, pb in zip(patches_a, patches_b):
            # Preprocess to (1, 3, 256, 256) float32 in [0, 1]
            ta = torch.from_numpy(pa.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            tb = torch.from_numpy(pb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

            ta = ta.to(device)
            tb = tb.to(device)

            probs = model(ta, tb, return_logits=False)
            patch_prob = probs.squeeze(0).squeeze(0).cpu().numpy()
            predicted_patches.append(patch_prob)

    full_prediction = stitch_patches(
        patches=predicted_patches,
        positions=positions,
        original_size=(h, w),
        padded_size=padded_size,
        patch_size=256,
    )
    return full_prediction


def main():
    parser = build_parser()
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device.type.upper()}")

    model = SiameseUNet(backbone="resnet18", pretrained=False)
    ckpt_path = Path(args.checkpoint)

    if ckpt_path.exists():
        print(f"Loading checkpoint weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Checkpoint '{args.checkpoint}' not found. Using initialized model weights.")

    model.to(device)

    if args.dry_run:
        print("Running DRY-RUN inference on synthetic 256x256 image pair...")
        img_a = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img_b = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        gt = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    else:
        if not args.t1 or not args.t2:
            raise ValueError("Please specify both --t1 and --t2 image file paths or pass --dry-run")

        img_a = np.array(Image.open(args.t1).convert("RGB"))
        img_b = np.array(Image.open(args.t2).convert("RGB"))
        gt = None

    print("Executing sliding-window change detection inference...")
    prob_map = run_inference(img_a, img_b, model, device, stride=args.stride)
    binary_mask = (prob_map >= 0.5).astype(np.uint8) * 255

    mask_path = out_dir / "change_mask.png"
    Image.fromarray(binary_mask).save(mask_path)
    print(f"Saved binary change mask to: {mask_path}")

    # Generate diagnostic visualizations
    heatmap_path = out_dir / "confidence_heatmap.png"
    ChangeVisualizer.plot_confidence_heatmap(prob_map, save_path=heatmap_path)
    print(f"Saved confidence heatmap to: {heatmap_path}")

    if gt is not None:
        grid_path = out_dir / "comparison_grid.png"
        ChangeVisualizer.plot_comparison_grid(img_a, img_b, gt, prob_map, save_path=grid_path)
        print(f"Saved 4-panel comparison grid to: {grid_path}")

        error_path = out_dir / "error_map.png"
        ChangeVisualizer.plot_error_map(gt, prob_map, save_path=error_path)
        print(f"Saved color-coded error map to: {error_path}")

    print("\nInference pipeline executed successfully!")


if __name__ == "__main__":
    main()
