"""
test_retrieval.py — Unit tests for the FAISS retrieval system.
Run with: pytest tests/
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.index import build_index, save_index, load_index
from src.retrieval.retriever import Retriever


SAMPLE_TEXTS = [
    "my order has not arrived",
    "I want a refund for my purchase",
    "can't log into my account",
    "package was damaged on arrival",
    "how do I return this item",
]
SAMPLE_METADATA = [
    {"customer_text": t, "brand_text": f"Reply to: {t}",
     "conversation_id": str(i), "created_at": "2023-01-01"}
    for i, t in enumerate(SAMPLE_TEXTS)
]


def test_build_index_returns_correct_size():
    index, meta, embs = build_index(SAMPLE_TEXTS, SAMPLE_METADATA)
    assert index.ntotal == len(SAMPLE_TEXTS)
    assert len(meta) == len(SAMPLE_TEXTS)
    assert embs.shape[0] == len(SAMPLE_TEXTS)


def test_save_and_load_index():
    index, meta, embs = build_index(SAMPLE_TEXTS, SAMPLE_METADATA)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        save_index(index, meta, embs, save_dir)
        loaded_index, loaded_meta, loaded_embs = load_index(save_dir)
        assert loaded_index.ntotal == index.ntotal
        assert len(loaded_meta) == len(meta)


def test_retriever_returns_results():
    index, meta, embs = build_index(SAMPLE_TEXTS, SAMPLE_METADATA)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        save_index(index, meta, embs, save_dir)
        retriever = Retriever(index_dir=save_dir)
        results = retriever.retrieve("my package never arrived", top_k=3)
        assert len(results) <= 3
        assert all("similarity" in r for r in results)
        assert all("brand_text" in r for r in results)


def test_retriever_empty_query():
    """Empty query should return empty list, not crash."""
    index, meta, embs = build_index(SAMPLE_TEXTS, SAMPLE_METADATA)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        save_index(index, meta, embs, save_dir)
        retriever = Retriever(index_dir=save_dir)
        results = retriever.retrieve("", top_k=3)
        assert results == []


def test_retriever_similarity_in_range():
    index, meta, embs = build_index(SAMPLE_TEXTS, SAMPLE_METADATA)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        save_index(index, meta, embs, save_dir)
        retriever = Retriever(index_dir=save_dir)
        results = retriever.retrieve("order not delivered", top_k=3)
        for r in results:
            assert 0.0 <= r["similarity"] <= 1.0


def test_evidence_score_empty():
    index, meta, embs = build_index(SAMPLE_TEXTS, SAMPLE_METADATA)
    with tempfile.TemporaryDirectory() as tmpdir:
        retriever = Retriever(index_dir=Path(tmpdir) / "nonexistent")
        score = retriever.get_evidence_score([])
        assert score == 0.0
