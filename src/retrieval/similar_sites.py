"""
Similar-Site Discovery & Pattern Retrieval Engine for SpaceNetra.

Finds satellite locations with similar visual characteristics or multi-temporal
change evolution signatures using RemoteCLIP embeddings and vector indexing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.retrieval.semantic_search import RemoteCLIPEmbedder, VectorSearchIndex


class SimilarSiteFinder:
    """
    Similar-Site & Pattern Retrieval Engine.
    """

    def __init__(
        self,
        embedder: Optional[RemoteCLIPEmbedder] = None,
        index: Optional[VectorSearchIndex] = None,
    ):
        self.embedder = embedder or RemoteCLIPEmbedder()
        self.index = index or VectorSearchIndex(embed_dim=self.embedder.embed_dim)

    def find_similar_sites(
        self,
        query_image_or_vec: Union[np.ndarray, List[float]],
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Finds satellite tiles with similar visual characteristics to the query.

        Args:
            query_image_or_vec: Query image patch array (H, W, C) or 1D feature vector.
            top_k: Number of top matching sites to retrieve.
            min_similarity: Minimum cosine similarity score threshold [0.0 - 1.0].

        Returns:
            Ranked list of similar site match dicts.
        """
        if isinstance(query_image_or_vec, np.ndarray):
            if query_image_or_vec.ndim in (2, 3):
                # Image patch -> embed
                query_vec = self.embedder.embed_images(query_image_or_vec)
            else:
                query_vec = query_image_or_vec
        else:
            query_vec = np.array(query_image_or_vec, dtype=np.float32)

        # Search index
        raw_results = self.index.search(query_vec, top_k=top_k)

        # Filter by minimum similarity score
        filtered_results = []
        for res in raw_results:
            if res["score"] >= min_similarity:
                filtered_results.append({
                    "rank": len(filtered_results) + 1,
                    "similarity_score": float(res["score"]),
                    "metadata": res["metadata"],
                })

        return filtered_results

    def find_similar_by_fingerprint(
        self,
        query_fingerprint: np.ndarray,
        index_fingerprints: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Matches multi-temporal change evolution signatures across time-series locations.
        """
        q_norm = query_fingerprint / (np.linalg.norm(query_fingerprint) + 1e-7)
        idx_norms = index_fingerprints / (np.linalg.norm(index_fingerprints, axis=1, keepdims=True) + 1e-7)

        scores = np.dot(idx_norms, q_norm)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "rank": rank + 1,
                "fingerprint_similarity": float(scores[idx]),
                "metadata": metadata_list[idx],
            })

        return results
