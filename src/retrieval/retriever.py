"""
retriever.py — Retrieves top-k historical customer/support pairs for a new message.

For a new customer message:
    1. Embed the message using the same model used to build the index.
    2. Search the FAISS index for the k most similar historical customer messages.
    3. Return the matched customer messages + their brand replies + similarity scores.

Similarity score: cosine similarity (0 to 1, higher = more similar).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_PROCESSED, EMBEDDING_MODEL, RETRIEVAL_TOP_K
from src.retrieval.index import load_index

INDEX_DIR = DATA_PROCESSED / "faiss_index"


class Retriever:
    """
    Wraps the FAISS index and provides a simple retrieve() interface.
    """

    def __init__(self, index_dir: Path = INDEX_DIR, model_name: str = EMBEDDING_MODEL):
        self.index_dir  = index_dir
        self.model_name = model_name
        self._index     = None
        self._metadata  = None
        self._encoder   = None

    def _load(self):
        """Lazy-load the index and encoder."""
        if self._index is None:
            if not self.index_dir.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {self.index_dir}\n"
                    "Run: python scripts/build_retrieval.py"
                )
            self._index, self._metadata, _ = load_index(self.index_dir)

        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
        """
        Retrieve top-k most similar historical support pairs.

        Args:
            query:  New customer message string.
            top_k:  Number of results to return.

        Returns:
            List of dicts, each containing:
                rank              — 1-indexed rank
                similarity        — cosine similarity score (0-1)
                customer_text     — historical customer message
                brand_text        — historical brand reply
                conversation_id   — conversation this came from
                created_at        — timestamp of the historical message
        """
        self._load()

        if not query or not query.strip():
            return []   # empty query → no results (handled gracefully)

        # Embed and normalise the query
        embedding = self._encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search — returns distances (inner product = cosine sim) and indices
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        scores, indices = self._index.search(embedding, k)
        scores  = scores[0]    # shape (k,)
        indices = indices[0]   # shape (k,)

        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0:   # FAISS returns -1 for empty slots
                continue
            meta = self._metadata[idx]
            results.append({
                "rank":            rank,
                "similarity":      round(float(score), 4),
                "customer_text":   meta.get("customer_text", ""),
                "brand_text":      meta.get("brand_text", ""),
                "conversation_id": meta.get("conversation_id", ""),
                "created_at":      str(meta.get("created_at", "")),
            })

        return results

    def get_evidence_score(self, results: list[dict]) -> float:
        """
        Compute a single evidence quality score from retrieval results.
        Returns the mean similarity of the top results, or 0.0 if empty.
        Used by the escalation decision layer.
        """
        if not results:
            return 0.0
        return round(float(np.mean([r["similarity"] for r in results])), 4)
