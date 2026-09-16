"""
classifier.py — Three intent classifiers in one file.

1. MajorityClassifier   — always predicts the most frequent training intent
2. TfidfLRClassifier    — TF-IDF features + Logistic Regression
3. EmbeddingClassifier  — Sentence Transformer embeddings + Logistic Regression

All classifiers share the same interface:
    .fit(texts, labels)
    .predict(texts)          -> list of intent strings
    .predict_proba(texts)    -> list of dicts {intent: probability}
    .predict_with_confidence(text) -> (intent, confidence, proba_dict)
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import EMBEDDING_MODEL, RANDOM_SEED


# ── 1. Majority Baseline ───────────────────────────────────────────────────────

class MajorityClassifier:
    """Always predicts the most frequent class seen during training."""

    def __init__(self):
        self.majority_class_ = None
        self.class_counts_   = None

    def fit(self, texts: list[str], labels: list[str]):
        counts = pd.Series(labels).value_counts()
        self.majority_class_ = counts.index[0]
        self.class_counts_   = counts.to_dict()
        self.classes_        = list(counts.index)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [self.majority_class_] * len(texts)

    def predict_proba(self, texts: list[str]) -> list[dict]:
        """Returns uniform probability for majority class = 1.0, others = 0.0."""
        proba = {c: 0.0 for c in self.classes_}
        proba[self.majority_class_] = 1.0
        return [proba] * len(texts)

    def predict_with_confidence(self, text: str) -> tuple[str, float, dict]:
        proba = {c: 0.0 for c in self.classes_}
        proba[self.majority_class_] = 1.0
        return self.majority_class_, 1.0, proba


# ── 2. TF-IDF + Logistic Regression ───────────────────────────────────────────

class TfidfLRClassifier:
    """
    TF-IDF vectorizer + Logistic Regression.

    Why TF-IDF + LR as a baseline?
    - Fast to train (seconds on CPU)
    - Interpretable: we can inspect which words drive each intent
    - Strong baseline for short text classification
    - No model download required
    """

    def __init__(self, C: float = 1.0, max_features: int = 10000):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,   # log(1+tf) — helps with long tweets
            )),
            ("lr", LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=RANDOM_SEED,
                class_weight="balanced",  # handles class imbalance
            )),
        ])
        self.classes_ = None

    def fit(self, texts: list[str], labels: list[str]):
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["lr"].classes_)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: list[str]) -> list[dict]:
        proba_matrix = self.pipeline.predict_proba(texts)
        return [
            dict(zip(self.classes_, row.tolist()))
            for row in proba_matrix
        ]

    def predict_with_confidence(self, text: str) -> tuple[str, float, dict]:
        proba_dict = self.predict_proba([text])[0]
        intent = max(proba_dict, key=proba_dict.get)
        confidence = proba_dict[intent]
        return intent, confidence, proba_dict

    def get_top_features(self, intent: str, n: int = 10) -> list[str]:
        """Return the top n TF-IDF features for a given intent (for explainability)."""
        lr = self.pipeline.named_steps["lr"]
        tfidf = self.pipeline.named_steps["tfidf"]
        classes = list(lr.classes_)
        if intent not in classes:
            return []
        idx = classes.index(intent)
        coef = lr.coef_[idx]
        top_idx = coef.argsort()[::-1][:n]
        feature_names = tfidf.get_feature_names_out()
        return [feature_names[i] for i in top_idx]


# ── 3. Sentence Embedding + Logistic Regression ────────────────────────────────

class EmbeddingClassifier:
    """
    Sentence Transformer embeddings + Logistic Regression.

    Why this architecture?
    - Sentence transformers capture semantic meaning, not just keywords.
    - LR on top of embeddings is fast, interpretable, and avoids overfitting
      on small datasets compared to fine-tuning the full transformer.
    - all-MiniLM-L6-v2 is ~80MB, fast on CPU, and well-tested.

    The embedding model is loaded lazily (only when fit() is called)
    so importing this module doesn't trigger a download.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, C: float = 1.0):
        self.model_name = model_name
        self.C          = C
        self.encoder_   = None   # SentenceTransformer, loaded lazily
        self.classifier_= None   # LogisticRegression
        self.classes_   = None

    def _load_encoder(self):
        if self.encoder_ is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self.encoder_ = SentenceTransformer(self.model_name)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings. Returns (n, dim) float32 array."""
        self._load_encoder()
        return self.encoder_.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def fit(self, texts: list[str], labels: list[str]):
        embeddings = self._embed(texts)
        self.classifier_ = LogisticRegression(
            C=self.C,
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )
        self.classifier_.fit(embeddings, labels)
        self.classes_ = list(self.classifier_.classes_)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        embeddings = self._embed(texts)
        return list(self.classifier_.predict(embeddings))

    def predict_proba(self, texts: list[str]) -> list[dict]:
        embeddings = self._embed(texts)
        proba_matrix = self.classifier_.predict_proba(embeddings)
        return [
            dict(zip(self.classes_, row.tolist()))
            for row in proba_matrix
        ]

    def predict_with_confidence(self, text: str) -> tuple[str, float, dict]:
        proba_dict = self.predict_proba([text])[0]
        intent = max(proba_dict, key=proba_dict.get)
        confidence = proba_dict[intent]
        return intent, confidence, proba_dict

    def save(self, path: str | Path):
        """Save the classifier (not the encoder — it's downloaded on demand)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier_,
                "classes":    self.classes_,
                "model_name": self.model_name,
                "C":          self.C,
            }, f)

    def load(self, path: str | Path):
        """Load a saved classifier."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.classifier_ = data["classifier"]
        self.classes_    = data["classes"]
        self.model_name  = data["model_name"]
        self.C           = data["C"]
        return self
