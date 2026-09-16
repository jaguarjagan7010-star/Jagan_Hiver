"""
train_classifier.py — Trains the embedding-based intent classifier (Phase 9).
Compares against baselines and saves the final comparison table.

Usage:
    python scripts/train_classifier.py
"""

import sys
import json
import pickle
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_PROCESSED, RESULTS_DIR, RANDOM_SEED
from src.intents.classifier import EmbeddingClassifier
from src.evaluation.metrics import (
    compute_classifier_metrics, save_metrics,
    save_confusion_matrix, save_comparison_table, plot_confusion_matrix
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default=None, help="Brand (informational only)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "embedding_classifier_log.txt", "w", encoding="utf-8") as log:

        # ── Load splits ───────────────────────────────────────────────────────
        log.write("Loading splits...\n"); log.flush()
        train = pd.read_csv(DATA_PROCESSED / "train.csv")
        val   = pd.read_csv(DATA_PROCESSED / "validation.csv")
        test  = pd.read_csv(DATA_PROCESSED / "test.csv")

        X_train = train["clean_text"].tolist()
        y_train = train["intent"].tolist()
        X_val   = val["clean_text"].tolist()
        y_val   = val["intent"].tolist()
        X_test  = test["clean_text"].tolist()
        y_test  = test["intent"].tolist()

        log.write(f"Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}\n\n")
        log.flush()

        # ── Tune C on validation set ──────────────────────────────────────────
        log.write("Tuning C on validation set...\n"); log.flush()
        best_c, best_val_f1 = 1.0, 0.0

        for C in [0.1, 1.0, 10.0]:
            clf = EmbeddingClassifier(C=C)
            clf.fit(X_train, y_train)
            val_preds = clf.predict(X_val)
            val_f1 = compute_classifier_metrics(y_val, val_preds, "tmp")["macro_f1"]
            log.write(f"  C={C}  val_macro_f1={val_f1}\n"); log.flush()
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_c = C
                best_clf = clf

        log.write(f"Best C={best_c}  val_macro_f1={best_val_f1}\n\n"); log.flush()

        # ── Evaluate on test set (untouched until now) ────────────────────────
        log.write("Evaluating on test set...\n"); log.flush()
        y_pred = best_clf.predict(X_test)

        metrics = compute_classifier_metrics(y_test, y_pred, "embedding_lr")
        save_metrics(metrics, "embedding_lr")
        save_confusion_matrix(y_test, y_pred, "embedding_lr")
        plot_confusion_matrix(y_test, y_pred, "embedding_lr")

        log.write(f"Accuracy:    {metrics['accuracy']}\n")
        log.write(f"Macro F1:    {metrics['macro_f1']}\n")
        log.write(f"Weighted F1: {metrics['weighted_f1']}\n\n")
        log.write("Per-class F1:\n")
        for intent, vals in metrics["per_class"].items():
            log.write(f"  {intent:<25} F1={vals['f1']}  support={vals['support']}\n")
        log.flush()

        # ── Save model ────────────────────────────────────────────────────────
        model_path = DATA_PROCESSED / "embedding_classifier.pkl"
        best_clf.save(model_path)
        log.write(f"\nSaved model: {model_path}\n")

        # ── Update comparison table with all 3 models ─────────────────────────
        import json as _json
        all_metrics = []
        for name in ["majority_baseline", "tfidf_lr", "embedding_lr"]:
            p = RESULTS_DIR / f"{name}_metrics.json"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    all_metrics.append(_json.load(f))

        if all_metrics:
            comp_path = save_comparison_table(all_metrics)
            log.write(f"Updated comparison table: {comp_path}\n")

            # Write human-readable comparison
            log.write("\n=== MODEL COMPARISON (test set) ===\n")
            log.write(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}\n")
            log.write("-" * 60 + "\n")
            for m in all_metrics:
                log.write(
                    f"{m['model']:<25} {m['accuracy']:>10} {m['macro_f1']:>10} {m['weighted_f1']:>12}\n"
                )

        log.write("\nPhase 9 complete.\n")

    print("Done. Check reports/results/embedding_classifier_log.txt")


if __name__ == "__main__":
    main()
