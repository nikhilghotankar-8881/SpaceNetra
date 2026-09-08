"""
Unit Tests for Similar-Site Satellite Discovery Engine.
"""

from pathlib import Path
import numpy as np
import pytest
from find_similar import build_parser
from src.retrieval.similar_sites import SimilarSiteFinder


def test_similar_site_finder_find_similar_sites():
    finder = SimilarSiteFinder()

    patches = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]
    metadata = [{"site_id": f"site_{i}"} for i in range(5)]

    embeddings = finder.embedder.embed_images(patches)
    finder.index.add_vectors(embeddings, metadata)

    # Query with exact first patch -> similarity score should be 1.0
    results = finder.find_similar_sites(patches[0], top_k=3, min_similarity=0.1)

    assert len(results) >= 1
    assert results[0]["rank"] == 1
    assert results[0]["metadata"]["site_id"] == "site_0"
    assert np.isclose(results[0]["similarity_score"], 1.0, atol=1e-4)


def test_similar_site_finder_find_similar_by_fingerprint():
    finder = SimilarSiteFinder()

    query_fp = np.array([1.0, 0.5, 0.0, 1.0], dtype=np.float32)
    index_fps = np.array([
        [1.0, 0.5, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.8, 0.4, 0.1, 0.9],
    ], dtype=np.float32)
    metadata = [{"id": 1}, {"id": 2}, {"id": 3}]

    results = finder.find_similar_by_fingerprint(query_fp, index_fps, metadata, top_k=2)

    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[0]["metadata"]["id"] == 1
    assert np.isclose(results[0]["fingerprint_similarity"], 1.0, atol=1e-4)


def test_find_similar_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--dry-run", "--top-k", "4", "--min-similarity", "0.6"])

    assert args.dry_run is True
    assert args.top_k == 4
    assert args.min_similarity == 0.6
