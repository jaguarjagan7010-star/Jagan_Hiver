"""
sample_data.py — Extract 500-1000 real AppleSupport tweets from twcs.csv.

Produces:
    data/raw/sample_data.csv   — 500-1000 inbound customer tweets to AppleSupport
    data/raw/sample_pairs.csv  — matched customer/reply pairs

Usage:
    python scripts/sample_data.py
    python scripts/sample_data.py --n 750 --seed 42

Reproducibility:
    Fixed seed=42 by default. Pass --seed to override.
    The same seed always produces the same sample.
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW, RAW_DATA_FILENAME, RANDOM_SEED, SELECTED_BRAND
from src.data.cleaner import clean_text
from src.data.conversations import reconstruct_pairs, get_brand_inbound


def main():
    parser = argparse.ArgumentParser(description="Sample AppleSupport tweets")
    parser.add_argument("--brand", default=SELECTED_BRAND,
                        help="Brand author_id to sample (default: AppleSupport)")
    parser.add_argument("--n",     type=int, default=750,
                        help="Target sample size (500-1000 recommended)")
    parser.add_argument("--seed",  type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    n = max(500, min(1000, args.n))   # clamp to 500-1000

    print(f"Loading {RAW_DATA_FILENAME}...")
    dtype_map = {
        "tweet_id": str, "author_id": str,
        "response_tweet_id": str, "in_response_to_tweet_id": str,
    }
    df = pd.read_csv(DATA_RAW / RAW_DATA_FILENAME, dtype=dtype_map, low_memory=False)
    df["inbound"] = df["inbound"].map(
        lambda x: True if str(x).strip().lower() == "true" else False
    )
    for col in ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]:
        df[col] = df[col].astype(str).str.strip().replace("nan", None)

    print(f"Full dataset: {len(df):,} rows")

    # -- Inbound messages directed at brand -----------------------------------
    inbound = get_brand_inbound(df, args.brand)
    print(f"Inbound messages to {args.brand}: {len(inbound):,}")

    if len(inbound) == 0:
        print(f"ERROR: No inbound messages found for brand '{args.brand}'.")
        print("Available brands (top 20):")
        top = df[~df["inbound"]]["author_id"].value_counts().head(20)
        print(top.to_string())
        sys.exit(1)

    # Clean text
    inbound = inbound.copy()
    inbound["clean_text"] = inbound["text"].apply(clean_text)
    inbound = inbound[inbound["clean_text"].str.len() > 5].copy()

    # Sample reproducibly
    rng = np.random.default_rng(args.seed)
    sample_n = min(n, len(inbound))
    idx = rng.choice(len(inbound), size=sample_n, replace=False)
    sample = inbound.iloc[sorted(idx)].reset_index(drop=True)

    # Save sample_data.csv
    out_cols = ["tweet_id", "author_id", "in_response_to_tweet_id",
                "created_at", "text", "clean_text", "inbound"]
    out_cols = [c for c in out_cols if c in sample.columns]
    out_path = DATA_RAW / "sample_data.csv"
    sample[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved {len(sample):,} inbound tweets -> {out_path}")

    # -- Conversation pairs ---------------------------------------------------
    print("\nReconstructing conversation pairs...")
    pairs = reconstruct_pairs(df, args.brand)
    print(f"Total pairs for {args.brand}: {len(pairs):,}")

    if len(pairs) > 0:
        pairs["customer_clean"] = pairs["customer_text"].apply(clean_text)
        pairs["brand_clean"]    = pairs["brand_text"].apply(clean_text)
        pairs = pairs[pairs["customer_clean"].str.len() > 5].copy()

        pair_n = min(n, len(pairs))
        pair_idx = rng.choice(len(pairs), size=pair_n, replace=False)
        sample_pairs = pairs.iloc[sorted(pair_idx)].reset_index(drop=True)

        pairs_path = DATA_RAW / "sample_pairs.csv"
        sample_pairs.to_csv(pairs_path, index=False, encoding="utf-8")
        print(f"Saved {len(sample_pairs):,} conversation pairs -> {pairs_path}")

    # -- Summary --------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Brand:          {args.brand}")
    print(f"Seed:           {args.seed}")
    print(f"Inbound sample: {len(sample):,} tweets")
    if len(pairs) > 0:
        print(f"Pair sample:    {len(sample_pairs):,} pairs")
    print(f"Output:         {out_path}")
    print(f"{'='*50}")
    print("\nNext: python scripts/train_baselines.py --brand AppleSupport")


if __name__ == "__main__":
    main()
