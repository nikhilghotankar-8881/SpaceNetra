"""
CLI Entry Point for SpaceNetra Multi-Temporal Satellite Series Analysis.

Usage:
    python analyze_series.py --input-dir data/sentinel_stack/ --output-dir outputs/multitemporal/
    python analyze_series.py --dry-run
"""

import argparse
from pathlib import Path
import numpy as np
import yaml

from src.temporal.multitemporal import MultiTemporalAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpaceNetra Multi-Temporal Satellite Image Series Analysis CLI"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing ordered satellite scene images / GeoTIFF files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/multitemporal",
        help="Directory to save multi-temporal change maps and chronology reports",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to model architecture configuration file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary change classification threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic 4-date time-series analysis dry-run without file dependencies",
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

    analyzer = MultiTemporalAnalyzer(model_or_arch=arch)

    if args.dry_run:
        print("Running Multi-Temporal Time-Series Analysis DRY-RUN...")

        # Create 4 synthetic 512x512 scenes representing T1, T2, T3, T4
        scenes = [
            np.random.randint(500, 4000, (512, 512, 3), dtype=np.uint16)
            for _ in range(4)
        ]
        timestamps = ["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"]

        results = analyzer.analyze_series(
            scene_list=scenes,
            timestamps=timestamps,
            threshold=args.threshold,
        )

        step_trans = results["step_transitions"]
        baseline_trans = results["baseline_transitions"]
        cum_freq = results["cumulative_frequency"]
        first_onset = results["first_change_onset"]

        print("Multi-Temporal Time-Series Analysis Summary:")
        print(f"  - Input Time-Series Length: {len(scenes)} scenes ({timestamps[0]} -> {timestamps[-1]})")
        print(f"  - Pairwise Step Transitions Computed: {len(step_trans)}")
        print(f"  - Baseline Transitions Computed:     {len(baseline_trans)}")
        print(f"  - Cumulative Frequency Map Shape:    {cum_freq.shape}, Max Frequency: {cum_freq.max()}")
        print(f"  - First Change Onset Map Shape:       {first_onset.shape}")
        print("Multi-Temporal Time-Series Analysis Dry-Run completed successfully!")

    else:
        if not args.input_dir:
            parser.error("--input-dir is required unless --dry-run is specified.")

        input_path = Path(args.input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Processing Multi-Temporal Series from: {args.input_dir}")
        # Load image files from directory sorted by name
        image_files = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
        if len(image_files) < 2:
            raise ValueError(f"Found fewer than 2 image files in {args.input_dir}")

        from src.ingestion.geotiff import GeoTIFFHandler
        scenes = [GeoTIFFHandler.read_raster(f)[0] for f in image_files]
        timestamps = [f.stem for f in image_files]

        results = analyzer.analyze_series(scene_list=scenes, timestamps=timestamps, threshold=args.threshold)

        # Save cumulative frequency and first change onset maps
        GeoTIFFHandler.write_raster(output_path / "cumulative_frequency.tif", results["cumulative_frequency"])
        GeoTIFFHandler.write_raster(output_path / "first_change_onset.tif", results["first_change_onset"])

        print(f"Multi-temporal analysis completed! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
