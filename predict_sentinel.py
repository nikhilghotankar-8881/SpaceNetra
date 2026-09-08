"""
CLI Entry Point for SpaceNetra Sentinel-2 Pair Change Detection.

Usage:
    python predict_sentinel.py --t1 data/scene1.tif --t2 data/scene2.tif --output outputs/change_map.tif
    python predict_sentinel.py --dry-run
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import yaml

from src.pipeline.sentinel_cd import SentinelChangeDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpaceNetra End-to-End Sentinel-2 Pair Change Detection CLI"
    )
    parser.add_argument("--t1", type=str, default=None, help="Path to T1 Sentinel-2 GeoTIFF image")
    parser.add_argument("--t2", type=str, default=None, help="Path to T2 Sentinel-2 GeoTIFF image")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions/sentinel_change_map.tif",
        help="Path to write output change map GeoTIFF",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to model architecture YAML configuration file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary change map classification threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic end-to-end scene pair inference dry-run without file dependencies",
    )
    return parser


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"model": {"architecture": "siamese_unet"}}


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    arch = config.get("model", {}).get("architecture", "siamese_unet")

    detector = SentinelChangeDetector(
        model_or_arch=arch,
        checkpoint_path=args.checkpoint,
    )

    if args.dry_run:
        print("Running Sentinel-2 End-to-End Change Detection DRY-RUN...")
        # Create synthetic 512x512 RGB scenes (values 0..10000 raw DN)
        img_t1 = np.random.randint(500, 4000, (512, 512, 3), dtype=np.uint16)
        img_t2 = np.random.randint(500, 4000, (512, 512, 3), dtype=np.uint16)

        results = detector.predict_pair(
            img_t1=img_t1,
            img_t2=img_t2,
            threshold=args.threshold,
        )

        prob_map = results["probability_map"]
        change_mask = results["change_mask"]

        print("Dry-Run Inference Output Summary:")
        print(f"  - Probability Map Shape: {prob_map.shape}, Range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
        print(f"  - Change Mask Shape: {change_mask.shape}, Change Pixels: {np.sum(change_mask > 0)}")
        print("Sentinel-2 Change Detection Dry-Run completed successfully!")

    else:
        if not args.t1 or not args.t2:
            parser.error("Both --t1 and --t2 arguments are required unless --dry-run is specified.")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing Sentinel-2 Change Detection:")
        print(f"  - Pre-change (T1):  {args.t1}")
        print(f"  - Post-change (T2): {args.t2}")
        print(f"  - Output Target:    {args.output}")

        saved_file = detector.predict_geotiff_pair(
            t1_path=args.t1,
            t2_path=args.t2,
            output_path=args.output,
            threshold=args.threshold,
        )
        print(f"Output change map successfully saved to: {saved_file}")


if __name__ == "__main__":
    main()
