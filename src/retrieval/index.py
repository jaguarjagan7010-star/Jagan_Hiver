"""
index.py — Builds and persists the FAISS vector index over historical support pairs.

The index stores embeddings of CUSTOMER messages from the training set only.
At query time, we embed the new customer message and find the most similar
historical customer messages, then return their brand replies.

IMPORTANT: The index is built from TRAIN data only.
Test data is never indexed — this prevents retrieval leakage.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_PROCESSED, EMBEDDING_MODEL


def build_index(
    texts: list[str],
    metadata: list[dict],
    model_name: str = EMBEDDING_MODEL,
) -> tuple:
    """
    Build a FAISS flat L2 index from a list of texts.

    Args:
        texts:     Customer message strings to index.
        metadata:  Parallel list of dicts with keys:
                   customer_text, brand_text, conversation_id, created_at
        model_name: Sentence transformer model name.

    Returns:
        (faiss_index, metadata_list, embeddings_array)
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name)

    print(f"Encoding {len(texts):,} texts...")
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise for cosine similarity via dot product
    )
    embeddings = embeddings.astype(np.float32)

    # FAISS IndexFlatIP = inner product on normalised vectors = cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Index built: {index.ntotal} vectors, dim={dim}")
    return index, metadata, embeddings


def save_index(index, metadata: list[dict], embeddings: np.ndarray, save_dir: Path):
    """Save FAISS index and metadata to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(save_dir / "faiss.index"))

    with open(save_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    np.save(save_dir / "embeddings.npy", embeddings)
    print(f"Saved index to {save_dir}")


def load_index(save_dir: Path) -> tuple:
    """Load FAISS index and metadata from disk."""
    index = faiss.read_index(str(save_dir / "faiss.index"))

    with open(save_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    embeddings = np.load(save_dir / "embeddings.npy")
    return index, metadata, embeddings
