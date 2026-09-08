"""
CLI Entry Point for SpaceNetra Spatial Change Event Creation & GeoJSON Export.

Usage:
    python build_events.py --input outputs/predictions/sentinel_change_map.tif --output outputs/events/change_events.geojson
    python build_events.py --dry-run
"""

import argparse
from pathlib import Path
import numpy as np

from src.temporal.event_builder import EventBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpaceNetra Spatial Change Event Creation & GeoJSON Exporter CLI"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to binary change mask image / GeoTIFF file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/events/change_events.geojson",
        help="Path to save output GeoJSON feature collection file",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Minimum spatial cluster pixel count to register as an event",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=10.0,
        help="Spatial pixel resolution in meters (default 10.0m for Sentinel-2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic change event extraction dry-run without file dependencies",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    builder = EventBuilder(
        min_area_pixels=args.min_area,
        pixel_resolution_m=args.resolution,
    )

    if args.dry_run:
        print("Running Spatial Change Event Creation DRY-RUN...")

        # Create synthetic 512x512 binary change mask with 3 distinct change patches
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[50:100, 50:100] = 255  # 50x50 = 2500 pixels (250,000 m²)
        mask[200:230, 300:340] = 255  # 30x40 = 1200 pixels (120,000 m²)
        mask[400:415, 400:415] = 255  # 15x15 = 225 pixels (22,500 m²)
        mask[10:12, 10:12] = 255  # 2x2 = 4 pixels (filtered out < min_area)

        prob_map = np.random.uniform(0.7, 0.99, (512, 512)).astype(np.float32)

        events = builder.extract_events(change_mask=mask, prob_map=prob_map)

        print("\nExtracted Spatial Change Events Summary:")
        print(f"  - Total Discrete Events Identified: {len(events)}")
        for event in events:
            print(
                f"  [{event['event_id']}] Area: {event['area_m2']:,.0f} m² | "
                f"Pixels: {event['pixel_count']} | Mean Prob: {event['mean_probability']:.4f}"
            )

        # Export GeoJSON
        out_path = Path(args.output)
        out_file = builder.export_geojson(events, out_path)
        print(f"\nGeoJSON FeatureCollection successfully saved to: {out_file}")
        print("Spatial Change Event Creation Dry-Run completed successfully!")

    else:
        if not args.input:
            parser.error("--input is required unless --dry-run is specified.")

        from src.ingestion.geotiff import GeoTIFFHandler

        mask, meta = GeoTIFFHandler.read_raster(args.input)
        transform = meta.get("transform")

        print(f"Extracting spatial change events from: {args.input}...")
        events = builder.extract_events(change_mask=mask, transform=transform)

        out_file = builder.export_geojson(events, args.output)
        print(f"Extracted {len(events)} change events! Saved to GeoJSON: {out_file}")


if __name__ == "__main__":
    main()
