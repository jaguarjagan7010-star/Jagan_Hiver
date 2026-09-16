"""
evaluate.py — Runs the full evaluation harness and saves all results.

Evaluates:
1. Classifier metrics on test set
2. Retrieval similarity distribution
3. LLM judge scores on golden set
4. Human vs LLM agreement (if human labels exist)

Usage:
    python scripts/evaluate.py
"""

import sys
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_PROCESSED, DATA_GOLDEN, RESULTS_DIR, FIGURES_DIR,
    SELECTED_BRAND, LLM_BACKEND, LLM_JUDGE_SAMPLE
)
from src.evaluation.metrics import (
    compute_classifier_metrics, save_metrics,
    save_confusion_matrix, plot_confusion_matrix, save_comparison_table
)


def evaluate_classifier(log):
    """Re-evaluate all classifiers on the test set."""
    log.write("=== CLASSIFIER EVALUATION ===\n"); log.flush()

    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    X_test = test["clean_text"].tolist()
    y_test = test["intent"].tolist()

    all_metrics = []

    for name, path in [
        ("majority_baseline", None),
        ("tfidf_lr",          DATA_PROCESSED / "tfidf_lr_model.pkl"),
        ("embedding_lr",      DATA_PROCESSED / "embedding_classifier.pkl"),
    ]:
        metrics_path = RESULTS_DIR / f"{name}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            all_metrics.append(m)
            log.write(f"{name}: accuracy={m['accuracy']}  macro_f1={m['macro_f1']}\n")
        else:
            log.write(f"{name}: metrics not found (run training scripts first)\n")

    if all_metrics:
        save_comparison_table(all_metrics)
        log.write(f"Comparison table saved.\n")

    log.flush()


def evaluate_retrieval(log):
    """Analyse retrieval similarity distribution."""
    log.write("\n=== RETRIEVAL EVALUATION ===\n"); log.flush()

    index_dir = DATA_PROCESSED / "faiss_index"
    if not index_dir.exists():
        log.write("FAISS index not found. Run build_retrieval.py first.\n")
        return

    from src.retrieval.retriever import Retriever
    retriever = Retriever()

    # Sample test messages and retrieve
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    sample = test.sample(n=min(200, len(test)), random_state=42)

    similarities = []
    for _, row in sample.iterrows():
        results = retriever.retrieve(row["clean_text"], top_k=3)
        if results:
            similarities.extend([r["similarity"] for r in results])

    if similarities:
        sims = np.array(similarities)
        log.write(f"Retrieval similarity stats (n={len(sims)}):\n")
        log.write(f"  mean={sims.mean():.3f}  median={np.median(sims):.3f}\n")
        log.write(f"  min={sims.min():.3f}  max={sims.max():.3f}\n")
        log.write(f"  pct above 0.5: {(sims > 0.5).mean()*100:.1f}%\n")
        log.write(f"  pct above 0.3: {(sims > 0.3).mean()*100:.1f}%\n")

        # Save similarity distribution
        pd.DataFrame({"similarity": similarities}).to_csv(
            RESULTS_DIR / "retrieval_similarity_dist.csv", index=False
        )
    log.flush()


def evaluate_golden_set(log):
    """Run LLM judge on golden set if replies exist."""
    log.write("\n=== GOLDEN SET / LLM JUDGE EVALUATION ===\n"); log.flush()

    golden_path = DATA_GOLDEN / "golden_set.csv"
    if not golden_path.exists():
        log.write("Golden set not found. Run golden set creation first.\n")
        return

    golden = pd.read_csv(golden_path, encoding="utf-8")

    # Check if human labels exist
    has_human = golden["gold_intent"].notna() & (golden["gold_intent"] != "")
    n_labelled = has_human.sum()
    log.write(f"Golden set: {len(golden)} examples, {n_labelled} human-labelled\n")

    if n_labelled > 0:
        # Classifier accuracy on human-labelled subset
        labelled = golden[has_human].copy()
        if "auto_intent" in labelled.columns:
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(labelled["gold_intent"], labelled["auto_intent"])
            f1  = f1_score(labelled["gold_intent"], labelled["auto_intent"],
                           average="macro", zero_division=0)
            log.write(f"Auto-label accuracy vs human: {acc:.3f}\n")
            log.write(f"Auto-label macro F1 vs human: {f1:.3f}\n")

    log.write("Note: LLM judge requires generated replies. Run generate_replies.py first.\n")
    log.flush()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "evaluation_log.txt", "w", encoding="utf-8") as log:
        evaluate_classifier(log)
        evaluate_retrieval(log)
        evaluate_golden_set(log)
        log.write("\nPhase 14 evaluation complete.\n")

    print("Done. Check reports/results/evaluation_log.txt")


if __name__ == "__main__":
    main()
