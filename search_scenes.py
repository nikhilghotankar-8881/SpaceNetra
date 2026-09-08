"""
CLI Entry Point for SpaceNetra Semantic Satellite Scene Vector Search.

Usage:
    python search_scenes.py --query "solar panels in desert"
    python search_scenes.py --dry-run
"""

import argparse
from pathlib import Path
import numpy as np

from src.retrieval.semantic_search import SemanticSearchEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpaceNetra Semantic Satellite Scene Retrieval CLI"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="solar panels in desert",
        help="Natural language text query to search satellite tile database",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="outputs/faiss_index",
        help="Directory storing FAISS / Cosine vector search index files",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matching scene tiles to retrieve",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic vector indexing & text search dry-run without file dependencies",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    engine = SemanticSearchEngine()

    if args.dry_run:
        print("Running Semantic Satellite Tile Search DRY-RUN...")

        # Create 10 synthetic 256x256 RGB patch images
        patches = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        metadata = [
            {"patch_id": f"patch_{i:03d}", "location": f"Zone_{i}", "file_path": f"data/tiles/tile_{i}.tif"}
            for i in range(10)
        ]

        # Index synthetic patches
        print(f"Indexing {len(patches)} synthetic satellite patches into vector index...")
        engine.index_patches(patches, metadata)

        # Execute text search query
        query = args.query
        print(f"Executing Text Search Query: '{query}' (top_k={args.top_k})...")
        results = engine.search_by_text(query, top_k=args.top_k)

        print("\nSemantic Search Query Results:")
        for res in results:
            rank = res["rank"]
            score = res["score"]
            meta = res["metadata"]
            print(f"  [{rank}] Score: {score:.4f} | Patch ID: {meta['patch_id']} | File: {meta['file_path']}")

        print("\nSemantic Satellite Tile Search Dry-Run completed successfully!")

    else:
        index_dir = Path(args.index_dir)
        if not index_dir.exists():
            print(f"Index directory '{args.index_dir}' not found. Initializing empty index...")

        engine.index.load(args.index_dir)
        print(f"Searching index at '{args.index_dir}' for query: '{args.query}'...")
        results = engine.search_by_text(args.query, top_k=args.top_k)

        print("\nTop Matching Satellite Tiles:")
        for res in results:
            print(f"  Rank {res['rank']}: Score = {res['score']:.4f} | Metadata = {res['metadata']}")


if __name__ == "__main__":
    main()
