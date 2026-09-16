"""
discovery.py — Clustering-aided intent discovery from real customer messages.

IMPORTANT: Clustering output is NOT ground truth.
It is used only as an aid to spot recurring themes.
Final intent definitions are documented in data/golden/intent_schema.json
and must be reviewed by a human.

Pipeline:
    customer messages
    -> TF-IDF vectors (fast, no model download needed for discovery)
    -> K-Means clustering
    -> top terms per cluster printed to file for human review
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RANDOM_SEED, RESULTS_DIR


def discover_intents(
    texts: list[str],
    n_clusters: int = 8,
    top_terms: int = 15,
) -> pd.DataFrame:
    """
    Run TF-IDF + K-Means to surface recurring themes.

    Args:
        texts:      List of cleaned customer message strings.
        n_clusters: Number of clusters to try (= candidate intents).
        top_terms:  Number of top TF-IDF terms to show per cluster.

    Returns:
        DataFrame with columns: cluster_id, size, top_terms
    """
    # TF-IDF: unigrams + bigrams, ignore very common/rare words
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    # K-Means clustering
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X)

    # Top terms per cluster (highest mean TF-IDF weight)
    rows = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_size = int(mask.sum())
        if cluster_size == 0:
            continue
        # Mean TF-IDF vector for this cluster
        center = X[mask].mean(axis=0)
        center_arr = np.asarray(center).flatten()
        top_idx = center_arr.argsort()[::-1][:top_terms]
        top_words = ", ".join(terms[top_idx])
        rows.append({
            "cluster_id": cluster_id,
            "size":       cluster_size,
            "pct":        round(cluster_size / len(texts) * 100, 1),
            "top_terms":  top_words,
        })

    result = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    return result, labels
