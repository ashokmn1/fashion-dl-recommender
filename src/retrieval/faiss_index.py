"""
FAISS-based Retrieval Index
=============================
Fast approximate nearest neighbor search for candidate item retrieval.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FashionFAISSIndex:
    """FAISS index for fast fashion item retrieval.

    Supports both brute-force (exact) and approximate (IVF-PQ) search
    depending on dataset size.

    Args:
        embedding_dim: Dimension of item embeddings.
        index_type: 'flat' for exact, 'ivf_pq' for approximate.
        n_centroids: Number of IVF centroids (for ivf_pq).
        n_probe: Number of clusters to search at query time.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "flat",
        n_centroids: int = 256,
        n_probe: int = 32,
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.n_centroids = n_centroids
        self.n_probe = n_probe

        self.index = None
        self.item_ids = []
        self.item_categories = []
        self.category_indices = {}  # category -> list of positions in index

        if not FAISS_AVAILABLE:
            print("WARNING: faiss not available. Using numpy brute-force fallback.")

    def build_index(
        self,
        embeddings: np.ndarray,
        item_ids: list[int],
        item_categories: list[str],
    ):
        """Build the FAISS index from item embeddings.

        Args:
            embeddings: [N, dim] item embedding matrix.
            item_ids: List of item IDs (same order as embeddings).
            item_categories: List of category strings (same order).
        """
        self.item_ids = item_ids
        self.item_categories = item_categories
        self.embeddings = embeddings.astype(np.float32)

        # Build per-category position mapping
        self.category_indices = {}
        for pos, cat in enumerate(item_categories):
            self.category_indices.setdefault(cat, []).append(pos)

        n_items = len(embeddings)

        if FAISS_AVAILABLE:
            # Normalize for cosine similarity
            faiss.normalize_L2(self.embeddings)

            if self.index_type == "ivf_pq" and n_items > 1000:
                # IVF-PQ for large datasets
                n_centroids = min(self.n_centroids, n_items // 4)
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFPQ(
                    quantizer, self.embedding_dim, n_centroids, 32, 8
                )
                self.index.train(self.embeddings)
                self.index.add(self.embeddings)
                self.index.nprobe = self.n_probe
            else:
                # Flat index for small datasets
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index.add(self.embeddings)

            print(f"FAISS index built: {n_items} items, type={self.index_type}")
        else:
            print(f"NumPy fallback index built: {n_items} items")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
        category_filter: Optional[str] = None,
        exclude_ids: Optional[set] = None,
    ) -> list[dict]:
        """Search for nearest neighbor items.

        Args:
            query_embedding: [1, dim] or [dim] query vector.
            top_k: Number of results to return.
            category_filter: Only return items from this category.
            exclude_ids: Item IDs to exclude from results.

        Returns:
            List of {item_id, score, category, rank} dicts.
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)
        exclude_ids = exclude_ids or set()

        if FAISS_AVAILABLE and self.index is not None:
            faiss.normalize_L2(query)
            # Search more than needed to account for filtering
            search_k = min(top_k * 5, self.index.ntotal)
            scores, indices = self.index.search(query, search_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # NumPy cosine similarity fallback
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            emb_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = (emb_norm @ query_norm.T).flatten()
            indices = np.argsort(-similarities)[:top_k * 5]
            scores = similarities[indices]

        # Filter and format results
        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            item_id = self.item_ids[idx]
            category = self.item_categories[idx]

            if item_id in exclude_ids:
                continue
            if category_filter and category != category_filter:
                continue

            results.append({
                "item_id": item_id,
                "score": float(score),
                "category": category,
                "rank": len(results) + 1,
            })

            if len(results) >= top_k:
                break

        return results

    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(path / "faiss.index"))

        np.save(path / "embeddings.npy", self.embeddings)
        np.save(path / "item_ids.npy", np.array(self.item_ids))

        import json
        with open(path / "metadata.json", "w") as f:
            json.dump({
                "item_categories": self.item_categories,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type,
            }, f)

    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)

        self.embeddings = np.load(path / "embeddings.npy")
        self.item_ids = np.load(path / "item_ids.npy").tolist()

        import json
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        self.item_categories = meta["item_categories"]
        self.embedding_dim = meta["embedding_dim"]

        if FAISS_AVAILABLE and (path / "faiss.index").exists():
            self.index = faiss.read_index(str(path / "faiss.index"))

        # Rebuild category index
        self.category_indices = {}
        for pos, cat in enumerate(self.item_categories):
            self.category_indices.setdefault(cat, []).append(pos)
