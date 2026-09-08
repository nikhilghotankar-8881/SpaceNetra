"""
Unit Tests for Semantic Satellite Tile Search & Vector Indexing Engine.
"""

from pathlib import Path
import numpy as np
import pytest
from search_scenes import build_parser
from src.retrieval.semantic_search import (
    RemoteCLIPEmbedder,
    SemanticSearchEngine,
    VectorSearchIndex,
)


def test_remote_clip_embedder():
    embedder = RemoteCLIPEmbedder(embed_dim=512)

    # 3 synthetic patches
    patches = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
    img_embeds = embedder.embed_images(patches)

    assert img_embeds.shape == (3, 512)
    # Verify L2 normalization
    norms = np.linalg.norm(img_embeds, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Text queries
    text_embeds = embedder.embed_text(["solar panels", "flooded city"])
    assert text_embeds.shape == (2, 512)
    text_norms = np.linalg.norm(text_embeds, axis=1)
    assert np.allclose(text_norms, 1.0, atol=1e-4)


def test_vector_search_index_add_and_search():
    index = VectorSearchIndex(embed_dim=512)

    vectors = np.random.randn(10, 512).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    metadata = [{"id": i, "name": f"item_{i}"} for i in range(10)]

    index.add_vectors(vectors, metadata)
    assert len(index.vectors) == 10

    query = vectors[0]  # Query exact first item
    results = index.search(query, top_k=3)

    assert len(results) == 3
    assert results[0]["rank"] == 1
    assert results[0]["metadata"]["id"] == 0
    assert np.isclose(results[0]["score"], 1.0, atol=1e-4)


def test_vector_search_index_save_and_load(tmp_path: Path):
    index_dir = str(tmp_path / "faiss_index")
    index = VectorSearchIndex(embed_dim=512)

    vectors = np.random.randn(5, 512).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    metadata = [{"id": i} for i in range(5)]

    index.add_vectors(vectors, metadata)
    index.save(index_dir)

    # Load into new index instance
    new_index = VectorSearchIndex(embed_dim=512)
    new_index.load(index_dir)

    assert len(new_index.vectors) == 5
    assert len(new_index.metadata) == 5
    assert new_index.metadata[0]["id"] == 0


def test_semantic_search_engine():
    engine = SemanticSearchEngine()

    patches = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(4)]
    metadata = [{"tile": f"tile_{i}"} for i in range(4)]

    engine.index_patches(patches, metadata)

    text_results = engine.search_by_text("solar panels", top_k=2)
    assert len(text_results) == 2

    image_results = engine.search_by_image(patches[0], top_k=2)
    assert len(image_results) == 2


def test_search_scenes_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--dry-run", "--query", "deforestation patch", "--top-k", "3"])

    assert args.dry_run is True
    assert args.query == "deforestation patch"
    assert args.top_k == 3
