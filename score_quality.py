"""
CLI Entry Point for SpaceNetra Satellite Change Detection Quality Assessment.

Usage:
    python score_quality.py --t1 data/scene1.tif --t2 data/scene2.tif
    python score_quality.py --dry-run
"""

import argparse
from pathlib import Path
import numpy as np

from src.confidence.quality_engine import ConfidenceEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpaceNetra Satellite Scene Quality Assessment & Confidence Scoring CLI"
    )
    parser.add_argument("--t1", type=str, default=None, help="Path to pre-change (T1) scene GeoTIFF")
    parser.add_argument("--t2", type=str, default=None, help="Path to post-change (T2) scene GeoTIFF")
    parser.add_argument("--prob-map", type=str, default=None, help="Path to prediction probability map GeoTIFF")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic quality scoring dry-run without file dependencies",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    engine = ConfidenceEngine()

    if args.dry_run:
        print("Running Satellite Change Detection Quality Assessment DRY-RUN...")

        # Create synthetic 512x512 scene pair and probability map
        t1 = np.random.randint(500, 4000, (512, 512, 3), dtype=np.uint16)
        t2 = t1 + np.random.randint(-100, 100, (512, 512, 3), dtype=np.int16)
        t2 = np.clip(t2, 0, 10000).astype(np.uint16)

        prob_map = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
        scl1 = np.zeros((512, 512), dtype=np.uint8)
        scl1[10:50, 10:50] = 9  # 40x40 cloud patch

        results = engine.evaluate_quality(
            img_t1=t1,
            img_t2=t2,
            prob_map=prob_map,
            scl_t1=scl1,
        )

        print("\nSatellite Change Detection Quality Report:")
        print(f"  - Composite Confidence Score: {results['composite_confidence']:.4f} ({results['status']})")
        print(f"  - Spatial Alignment Score:   {results['alignment_score']:.4f}")
        print(f"  - Cloud-Free Score:          {results['cloud_free_score']:.4f} (Contamination: {results['cloud_contamination_ratio']*100:.2f}%)")
        print(f"  - Prediction Certainty Score:{results['prediction_certainty_score']:.4f}")
        print("\nQuality Assessment Dry-Run completed successfully!")

    else:
        if not args.t1 or not args.t2:
            parser.error("Both --t1 and --t2 arguments are required unless --dry-run is specified.")

        from src.ingestion.geotiff import GeoTIFFHandler

        t1, _ = GeoTIFFHandler.read_raster(args.t1)
        t2, _ = GeoTIFFHandler.read_raster(args.t2)
        prob_map = GeoTIFFHandler.read_raster(args.prob_map)[0] if args.prob_map else None

        results = engine.evaluate_quality(img_t1=t1, img_t2=t2, prob_map=prob_map)

        print(f"\nQuality Report for ({args.t1} vs {args.t2}):")
        print(f"  - Composite Confidence Score: {results['composite_confidence']:.4f} [{results['status']}]")
        print(f"  - Alignment Score:            {results['alignment_score']:.4f}")
        print(f"  - Cloud-Free Score:           {results['cloud_free_score']:.4f}")
        print(f"  - Prediction Certainty Score: {results['prediction_certainty_score']:.4f}")


if __name__ == "__main__":
    main()
