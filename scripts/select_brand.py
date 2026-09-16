"""
select_brand.py — Ranks all brands and selects one using objective criteria.

Usage:
    python scripts/select_brand.py
"""

import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW, RAW_DATA_FILENAME, RESULTS_DIR, RANDOM_SEED
from src.brand.selector import compute_usable_pairs, select_brand, score_brands


def load_full():
    dtype_map = {
        "tweet_id": str, "author_id": str,
        "response_tweet_id": str, "in_response_to_tweet_id": str,
    }
    df = pd.read_csv(
        DATA_RAW / RAW_DATA_FILENAME,
        dtype=dtype_map,
        parse_dates=["created_at"],
        low_memory=False,
    )
    df["inbound"] = df["inbound"].map(
        lambda x: True if str(x).strip().lower() == "true" else False
    )
    for col in ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]:
        df[col] = df[col].str.strip().replace("nan", None)
    return df


def build_brand_table(df):
    outbound = (
        df[~df["inbound"]]
        .groupby("author_id")["tweet_id"]
        .count()
        .rename("outbound_tweets")
    )
    tweet_to_brand = (
        df[~df["inbound"] & df["tweet_id"].notna()]
        [["tweet_id", "author_id"]]
        .drop_duplicates("tweet_id")
        .set_index("tweet_id")["author_id"]
    )
    inb = df[df["inbound"] & df["in_response_to_tweet_id"].notna()].copy()
    inb["brand"] = inb["in_response_to_tweet_id"].map(tweet_to_brand)
    inb = inb[inb["brand"].notna()]

    inbound_counts = inb.groupby("brand").agg(
        inbound_msgs=("tweet_id", "count"),
        conversations=("in_response_to_tweet_id", "nunique"),
    )
    brand_df = (
        outbound.to_frame()
        .join(inbound_counts, how="outer")
        .fillna(0).astype(int)
        .reset_index()
        .rename(columns={"author_id": "brand"})
        .sort_values("inbound_msgs", ascending=False)
        .reset_index(drop=True)
    )
    return brand_df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "select_brand_log.txt", "w", encoding="utf-8") as log:

        log.write("Loading full dataset...\n"); log.flush()
        df = load_full()
        log.write(f"Loaded {len(df):,} rows\n"); log.flush()

        log.write("Building brand table...\n"); log.flush()
        brand_df = build_brand_table(df)
        log.write(f"Found {len(brand_df)} brands\n"); log.flush()

        log.write("Computing usable pairs...\n"); log.flush()
        brand_df = compute_usable_pairs(df, brand_df)

        # Score and rank
        scored = score_brands(brand_df)
        best_brand, _ = select_brand(brand_df)

        # Save full ranked table
        scored.to_csv(RESULTS_DIR / "brand_ranking.csv", index=False, encoding="utf-8")

        # Save selection result
        top5 = scored.head(5)[["brand","inbound_msgs","conversations","usable_pairs","reply_rate","composite_score"]]
        log.write("\n=== TOP 5 BRANDS ===\n")
        log.write(top5.to_string(index=False))
        log.write(f"\n\nSELECTED BRAND: {best_brand}\n")

        # Save to JSON for other scripts to read
        result = {
            "selected_brand": best_brand,
            "stats": scored[scored["brand"] == best_brand].iloc[0].to_dict()
        }
        # Convert numpy types to native Python for JSON serialisation
        for k, v in result["stats"].items():
            if hasattr(v, "item"):
                result["stats"][k] = v.item()

        with open(RESULTS_DIR / "selected_brand.json", "w", encoding="utf-8") as jf:
            json.dump(result, jf, indent=2)

        log.write(f"\nSaved: {RESULTS_DIR / 'brand_ranking.csv'}\n")
        log.write(f"Saved: {RESULTS_DIR / 'selected_brand.json'}\n")
        log.write("Phase 3 complete.\n")

    print("Done. Check reports/results/select_brand_log.txt")


if __name__ == "__main__":
    main()
