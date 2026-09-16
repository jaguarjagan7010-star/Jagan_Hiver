"""
build_retrieval.py — Builds the FAISS index over historical support pairs.

LEAKAGE AUDIT:
  - Only indexes pairs whose conversation_id is in the TRAIN split.
  - Explicitly excludes conversation_ids from test.csv AND golden set.
  - Logs the audit trail to retrieval_build_log.txt.

Usage:
    python scripts/build_retrieval.py
    python scripts/build_retrieval.py --brand AppleSupport
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_RAW, RAW_DATA_FILENAME, DATA_PROCESSED, DATA_GOLDEN,
    RESULTS_DIR, SELECTED_BRAND, EMBEDDING_MODEL,
)
from src.data.cleaner import clean_text
from src.data.conversations import reconstruct_pairs
from src.retrieval.index import build_index, save_index

INDEX_DIR = DATA_PROCESSED / "faiss_index"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", default=SELECTED_BRAND)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "retrieval_build_log.txt", "w", encoding="utf-8") as log:

        log.write(f"Brand: {args.brand}\n")
        log.write("Loading data...\n"); log.flush()

        dtype_map = {"tweet_id": str, "author_id": str,
                     "response_tweet_id": str, "in_response_to_tweet_id": str}
        df = pd.read_csv(DATA_RAW / RAW_DATA_FILENAME, dtype=dtype_map, low_memory=False)
        df["inbound"] = df["inbound"].map(
            lambda x: True if str(x).strip().lower() == "true" else False)
        for col in ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]:
            df[col] = df[col].astype(str).str.strip().replace("nan", None)
        log.write(f"Loaded {len(df):,} rows\n"); log.flush()

        # Reconstruct conversation pairs
        log.write("Reconstructing conversation pairs...\n"); log.flush()
        pairs = reconstruct_pairs(df, args.brand)
        log.write(f"Found {len(pairs):,} customer/brand pairs\n"); log.flush()

        if len(pairs) == 0:
            log.write("ERROR: No pairs found. Check brand name.\n")
            return

        # Clean texts
        pairs["customer_clean"] = pairs["customer_text"].apply(clean_text)
        pairs["brand_clean"]    = pairs["brand_text"].apply(clean_text)
        pairs = pairs[pairs["customer_clean"].str.len() > 5].copy()
        log.write(f"After cleaning: {len(pairs):,} pairs\n"); log.flush()

        # ── LEAKAGE AUDIT ─────────────────────────────────────────────────────
        log.write("\n=== LEAKAGE AUDIT ===\n")

        # Load train conversation IDs
        train_path = DATA_PROCESSED / "train.csv"
        if not train_path.exists():
            log.write("WARNING: train.csv not found. Run train_baselines.py first.\n")
            log.write("Indexing ALL pairs (leakage risk — run training first).\n")
            pairs_to_index = pairs
        else:
            train_df = pd.read_csv(train_path)
            train_conv_ids = set(train_df["conversation_id"].astype(str))
            log.write(f"Train conversation IDs: {len(train_conv_ids):,}\n")

            # Load test conversation IDs
            test_path = DATA_PROCESSED / "test.csv"
            test_conv_ids = set()
            if test_path.exists():
                test_df = pd.read_csv(test_path)
                test_conv_ids = set(test_df["conversation_id"].astype(str))
                log.write(f"Test conversation IDs (excluded): {len(test_conv_ids):,}\n")

            # Load validation conversation IDs
            val_path = DATA_PROCESSED / "validation.csv"
            val_conv_ids = set()
            if val_path.exists():
                val_df = pd.read_csv(val_path)
                val_conv_ids = set(val_df["conversation_id"].astype(str))
                log.write(f"Val conversation IDs (excluded): {len(val_conv_ids):,}\n")

            # Load golden set conversation IDs
            golden_conv_ids = set()
            golden_path = DATA_GOLDEN / "golden_200.json"
            if not golden_path.exists():
                golden_path = DATA_GOLDEN / "golden_set.csv"
            if golden_path.exists():
                if str(golden_path).endswith(".json"):
                    import json
                    with open(golden_path) as f:
                        golden_data = json.load(f)
                    golden_conv_ids = {
                        str(r.get("conversation_id", ""))
                        for r in golden_data if r.get("conversation_id")
                    }
                else:
                    golden_df = pd.read_csv(golden_path)
                    if "conversation_id" in golden_df.columns:
                        golden_conv_ids = set(golden_df["conversation_id"].astype(str))
                log.write(f"Golden set conversation IDs (excluded): {len(golden_conv_ids):,}\n")
            else:
                log.write("Golden set not found (will be excluded when created).\n")

            # Exclude test, val, and golden from index
            excluded = test_conv_ids | val_conv_ids | golden_conv_ids
            pairs["conv_str"] = pairs["conversation_id"].astype(str)

            # Only index train conversations
            pairs_to_index = pairs[
                pairs["conv_str"].isin(train_conv_ids) &
                ~pairs["conv_str"].isin(excluded)
            ].copy()

            log.write(f"\nPairs in train split:          {pairs['conv_str'].isin(train_conv_ids).sum():,}\n")
            log.write(f"Pairs excluded (test/val/gold): {pairs['conv_str'].isin(excluded).sum():,}\n")
            log.write(f"Pairs indexed (clean):          {len(pairs_to_index):,}\n")

            # Verify no leakage
            indexed_convs = set(pairs_to_index["conv_str"])
            overlap_test   = indexed_convs & test_conv_ids
            overlap_golden = indexed_convs & golden_conv_ids
            log.write(f"\nLeakage check:\n")
            log.write(f"  Index ∩ test:   {len(overlap_test)} (must be 0)\n")
            log.write(f"  Index ∩ golden: {len(overlap_golden)} (must be 0)\n")
            assert len(overlap_test) == 0,   "LEAKAGE: test conversations in index!"
            assert len(overlap_golden) == 0, "LEAKAGE: golden conversations in index!"
            log.write("  PASS: No leakage detected.\n")

        log.write("=== END LEAKAGE AUDIT ===\n\n"); log.flush()

        # Build metadata list
        metadata = []
        for _, row in pairs_to_index.iterrows():
            metadata.append({
                "customer_text":   row["customer_text"],
                "brand_text":      row["brand_text"],
                "conversation_id": str(row["conversation_id"]),
                "created_at":      str(row.get("created_at", "")),
            })

        texts = [m["customer_text"] for m in metadata]

        log.write(f"Building FAISS index over {len(texts):,} texts...\n"); log.flush()
        index, metadata, embeddings = build_index(texts, metadata, model_name=EMBEDDING_MODEL)
        save_index(index, metadata, embeddings, INDEX_DIR)

        log.write(f"Index saved to {INDEX_DIR}\n")
        log.write(f"Index size: {index.ntotal} vectors\n")
        log.write("build_retrieval complete.\n")

    print(f"Done. Index: {index.ntotal} vectors. Check reports/results/retrieval_build_log.txt")


if __name__ == "__main__":
    main()
