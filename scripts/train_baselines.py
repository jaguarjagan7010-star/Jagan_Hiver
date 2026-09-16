"""
train_baselines.py — Trains and evaluates majority-class and TF-IDF+LR baselines.
Covers Phase 7 (majority) and Phase 8 (TF-IDF + LR).

Usage:
    python scripts/train_baselines.py
"""

import sys
import json
import pickle
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_RAW, RAW_DATA_FILENAME, DATA_PROCESSED, RESULTS_DIR,
    RANDOM_SEED, SELECTED_BRAND
)
from src.data.cleaner import clean_text
from src.data.conversations import get_brand_inbound
from src.intents.labeling import label_dataframe, split_conversations
from src.intents.classifier import MajorityClassifier, TfidfLRClassifier
from src.evaluation.metrics import (
    compute_classifier_metrics, save_metrics,
    save_confusion_matrix, save_comparison_table, plot_confusion_matrix
)


def load_and_prepare(brand: str, sample_n: int = 20000):
    """Load data, clean, label, split — all in one place."""
    with open(RESULTS_DIR / "prepare_log.txt", "w", encoding="utf-8") as log:
        log.write(f"Loading data for brand: {brand}\n"); log.flush()

        dtype_map = {"tweet_id": str, "author_id": str,
                     "response_tweet_id": str, "in_response_to_tweet_id": str}
        df = pd.read_csv(
            DATA_RAW / RAW_DATA_FILENAME,
            dtype=dtype_map, low_memory=False,
        )
        df["inbound"] = df["inbound"].map(
            lambda x: True if str(x).strip().lower() == "true" else False)
        for col in ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]:
            df[col] = df[col].astype(str).str.strip().replace("nan", None)

        log.write(f"Full dataset: {len(df):,} rows\n"); log.flush()

        # Get inbound messages for this brand
        inbound = get_brand_inbound(df, brand)
        log.write(f"Brand inbound messages: {len(inbound):,}\n"); log.flush()

        if len(inbound) == 0:
            raise ValueError(f"No inbound messages found for brand: {brand}")

        # Sample if needed
        if sample_n and len(inbound) > sample_n:
            inbound = inbound.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
            log.write(f"Sampled to {len(inbound):,}\n"); log.flush()

        # Clean text
        inbound["clean_text"] = inbound["text"].apply(clean_text)
        inbound = inbound[inbound["clean_text"].str.len() > 5].copy()

        # Add conversation_id (use in_response_to_tweet_id as proxy)
        inbound["conversation_id"] = inbound["in_response_to_tweet_id"].fillna(inbound["tweet_id"])

        # Label
        labelled = label_dataframe(inbound, text_col="clean_text")
        log.write(f"Labelled messages: {len(labelled):,}\n")
        log.write("Intent distribution:\n")
        log.write(labelled["intent"].value_counts().to_string() + "\n"); log.flush()

        # Split at conversation level
        train, val, test = split_conversations(labelled)
        log.write(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}\n"); log.flush()

        # Save splits
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        cols = ["tweet_id", "clean_text", "intent", "conversation_id"]
        train[cols].to_csv(DATA_PROCESSED / "train.csv", index=False, encoding="utf-8")
        val[cols].to_csv(DATA_PROCESSED / "validation.csv", index=False, encoding="utf-8")
        test[cols].to_csv(DATA_PROCESSED / "test.csv", index=False, encoding="utf-8")
        log.write("Saved train/validation/test CSVs\n"); log.flush()

    return train, val, test


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default=SELECTED_BRAND,
                        help="Brand to train on (default: AppleSupport)")
    args = parser.parse_args()
    brand = args.brand

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "baselines_log.txt", "w", encoding="utf-8") as log:

        # ── Load data ─────────────────────────────────────────────────────────
        log.write("=== LOADING DATA ===\n"); log.flush()

        # Check if splits already exist
        if (DATA_PROCESSED / "train.csv").exists():
            log.write("Loading existing splits...\n"); log.flush()
            train = pd.read_csv(DATA_PROCESSED / "train.csv")
            val   = pd.read_csv(DATA_PROCESSED / "validation.csv")
            test  = pd.read_csv(DATA_PROCESSED / "test.csv")
        else:
            train, val, test = load_and_prepare(brand, sample_n=20000)

        X_train = train["clean_text"].tolist()
        y_train = train["intent"].tolist()
        X_val   = val["clean_text"].tolist()
        y_val   = val["intent"].tolist()
        X_test  = test["clean_text"].tolist()
        y_test  = test["intent"].tolist()

        log.write(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n\n")
        log.flush()

        all_metrics = []

        # ── Phase 7: Majority Baseline ────────────────────────────────────────
        log.write("=== PHASE 7: MAJORITY BASELINE ===\n"); log.flush()
        majority = MajorityClassifier()
        majority.fit(X_train, y_train)
        y_pred_maj = majority.predict(X_test)

        metrics_maj = compute_classifier_metrics(y_test, y_pred_maj, "majority_baseline")
        save_metrics(metrics_maj, "majority_baseline")
        save_confusion_matrix(y_test, y_pred_maj, "majority_baseline")
        plot_confusion_matrix(y_test, y_pred_maj, "majority_baseline")
        all_metrics.append(metrics_maj)

        log.write(f"Majority class: {majority.majority_class_}\n")
        log.write(f"Accuracy:    {metrics_maj['accuracy']}\n")
        log.write(f"Macro F1:    {metrics_maj['macro_f1']}\n")
        log.write(f"Weighted F1: {metrics_maj['weighted_f1']}\n\n")
        log.flush()

        # ── Phase 8: TF-IDF + LR ─────────────────────────────────────────────
        log.write("=== PHASE 8: TF-IDF + LOGISTIC REGRESSION ===\n"); log.flush()

        # Tune C on validation set (only 3 values — no test set used)
        best_c, best_val_f1 = 1.0, 0.0
        for C in [0.1, 1.0, 10.0]:
            clf = TfidfLRClassifier(C=C)
            clf.fit(X_train, y_train)
            val_preds = clf.predict(X_val)
            val_f1 = compute_classifier_metrics(y_val, val_preds, "tmp")["macro_f1"]
            log.write(f"  C={C}  val_macro_f1={val_f1}\n"); log.flush()
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_c = C

        log.write(f"Best C={best_c} (val macro F1={best_val_f1})\n"); log.flush()

        # Train final model with best C, evaluate on TEST (untouched until now)
        tfidf_lr = TfidfLRClassifier(C=best_c)
        tfidf_lr.fit(X_train, y_train)
        y_pred_tfidf = tfidf_lr.predict(X_test)

        metrics_tfidf = compute_classifier_metrics(y_test, y_pred_tfidf, "tfidf_lr")
        save_metrics(metrics_tfidf, "tfidf_lr")
        save_confusion_matrix(y_test, y_pred_tfidf, "tfidf_lr")
        plot_confusion_matrix(y_test, y_pred_tfidf, "tfidf_lr")
        all_metrics.append(metrics_tfidf)

        log.write(f"Accuracy:    {metrics_tfidf['accuracy']}\n")
        log.write(f"Macro F1:    {metrics_tfidf['macro_f1']}\n")
        log.write(f"Weighted F1: {metrics_tfidf['weighted_f1']}\n\n")

        # Save top features per intent
        log.write("Top features per intent:\n")
        for intent in sorted(set(y_train)):
            feats = tfidf_lr.get_top_features(intent, n=8)
            log.write(f"  {intent}: {', '.join(feats)}\n")
        log.flush()

        # Save TF-IDF model
        model_path = DATA_PROCESSED / "tfidf_lr_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(tfidf_lr, f)
        log.write(f"\nSaved model: {model_path}\n")

        # ── Comparison table ──────────────────────────────────────────────────
        comp_path = save_comparison_table(all_metrics)
        log.write(f"Saved comparison: {comp_path}\n")
        log.write("\nPhases 7+8 complete.\n")

    print("Done. Check reports/results/baselines_log.txt")


if __name__ == "__main__":
    main()
