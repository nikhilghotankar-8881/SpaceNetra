"""
CLI Entry Point for SpaceNetra Similar-Site Satellite Discovery.

Usage:
    python find_similar.py --tile data/sample_tile.tif --top-k 5
    python find_similar.py --dry-run
"""

import argparse
from pathlib import Path
import numpy as np

from src.retrieval.similar_sites import SimilarSiteFinder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpaceNetra Similar-Site Satellite Tile Discovery CLI"
    )
    parser.add_argument(
        "--tile",
        type=str,
        default=None,
        help="Path to query satellite image patch / GeoTIFF file",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="outputs/faiss_index",
        help="Directory storing FAISS / Cosine vector index files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top similar satellite sites to retrieve",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.5,
        help="Minimum similarity score threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic similar-site discovery dry-run without file dependencies",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    finder = SimilarSiteFinder()

    if args.dry_run:
        print("Running Similar-Site Satellite Discovery DRY-RUN...")

        # Create 10 synthetic 256x256 RGB patch images
        patches = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        metadata = [
            {
                "site_id": f"SITE_{i+1:03d}",
                "location": f"Zone_{i+1}",
                "coordinates": f"{28.61 + i*0.05:.4f}°N, {77.21 + i*0.05:.4f}°E",
            }
            for i in range(10)
        ]

        # Populate index with synthetic patches
        embeddings = finder.embedder.embed_images(patches)
        finder.index.add_vectors(embeddings, metadata)

        # Query using the first patch as reference query tile
        query_patch = patches[0]
        print(f"Finding similar sites for reference query tile (top_k={args.top_k}, min_similarity={args.min_similarity})...")

        results = finder.find_similar_sites(
            query_image_or_vec=query_patch,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
        )

        print("\nRanked Similar Satellite Sites:")
        for res in results:
            rank = res["rank"]
            score = res["similarity_score"]
            meta = res["metadata"]
            print(f"  [{rank}] Similarity: {score:.4f} | Site ID: {meta['site_id']} | Coords: {meta['coordinates']}")

        print("\nSimilar-Site Satellite Discovery Dry-Run completed successfully!")

    else:
        if not args.tile:
            parser.error("--tile is required unless --dry-run is specified.")

        from src.ingestion.geotiff import GeoTIFFHandler

        query_tile, _ = GeoTIFFHandler.read_raster(args.tile)
        finder.index.load(args.index_dir)

        print(f"Searching index at '{args.index_dir}' for similar sites to '{args.tile}'...")
        results = finder.find_similar_sites(
            query_image_or_vec=query_tile,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
        )

        print("\nTop Matching Similar Sites:")
        for res in results:
            print(f"  Rank {res['rank']}: Score = {res['similarity_score']:.4f} | Site = {res['metadata']}")


if __name__ == "__main__":
    main()
