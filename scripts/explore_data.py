"""
explore_data.py — Dataset summary statistics and charts.
All output written to files to avoid Windows stdout encoding issues.

Usage:
    python scripts/explore_data.py --sample-size 100000
"""

import sys
import json
import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW, RAW_DATA_FILENAME, FIGURES_DIR, RESULTS_DIR, RANDOM_SEED


def log(msg: str):
    """Print and flush immediately so we see progress."""
    print(msg, flush=True)


def load_data(sample_size):
    log(f"Loading data from {DATA_RAW / RAW_DATA_FILENAME} ...")
    t = time.time()
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
    log(f"  Loaded {len(df):,} rows in {time.time()-t:.1f}s")

    # Fix inbound column
    df["inbound"] = df["inbound"].map(
        lambda x: True if str(x).strip().lower() == "true" else False
    )
    # Clean ID columns
    for col in ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]:
        df[col] = df[col].str.strip().replace("nan", pd.NA)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
        log(f"  Sampled {len(df):,} rows")

    return df


def compute_summary(df):
    log("Computing summary stats...")
    text_len = df["text"].dropna().str.len()
    summary = {
        "total_rows":           len(df),
        "inbound_rows":         int(df["inbound"].sum()),
        "outbound_rows":        int((~df["inbound"]).sum()),
        "inbound_pct":          round(df["inbound"].mean() * 100, 2),
        "duplicate_rows":       int(df.duplicated().sum()),
        "duplicate_tweet_ids":  int(df["tweet_id"].duplicated().sum()),
        "missing_text":         int(df["text"].isna().sum()),
        "missing_response_id":  int(df["response_tweet_id"].isna().sum()),
        "unique_authors":       int(df["author_id"].nunique()),
        "date_min":             str(df["created_at"].min()),
        "date_max":             str(df["created_at"].max()),
        "text_len_mean":        round(float(text_len.mean()), 1),
        "text_len_median":      round(float(text_len.median()), 1),
        "text_len_min":         int(text_len.min()),
        "text_len_max":         int(text_len.max()),
        "text_len_p95":         round(float(text_len.quantile(0.95)), 1),
    }
    return summary


def compute_brand_table(df):
    log("Computing brand table (vectorized)...")
    t = time.time()

    # Outbound tweet counts per brand
    outbound = (
        df[~df["inbound"]]
        .groupby("author_id")["tweet_id"]
        .count()
        .rename("outbound_tweets")
    )

    # Map tweet_id -> brand (for outbound tweets only)
    tweet_to_brand = (
        df[~df["inbound"] & df["tweet_id"].notna()]
        [["tweet_id", "author_id"]]
        .drop_duplicates("tweet_id")
        .set_index("tweet_id")["author_id"]
    )

    # For each inbound tweet, find which brand it was directed at
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
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"author_id": "brand"})
        .sort_values("inbound_msgs", ascending=False)
        .reset_index(drop=True)
    )

    log(f"  Brand table done in {time.time()-t:.1f}s — {len(brand_df)} brands found")
    return brand_df


def save_charts(df, brand_df, summary):
    log("Saving charts...")
    # Import matplotlib only here — avoids any import-time hang
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # Chart 1: top brands
    top = brand_df.head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top["brand"], top["inbound_msgs"])
    ax.set_xlabel("Inbound Messages")
    ax.set_title("Top 20 Brands by Inbound Customer Messages")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "top_brands_inbound.png", dpi=150)
    plt.close(fig)

    # Chart 2: message length histogram
    lengths = df["text"].dropna().str.len()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths.clip(upper=300), bins=60, edgecolor="none", alpha=0.8)
    ax.axvline(lengths.median(), color="red", linestyle="--",
               label=f"Median={lengths.median():.0f}")
    ax.set_title("Tweet Text Length Distribution (chars, clipped at 300)")
    ax.set_xlabel("Characters")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "message_length_dist.png", dpi=150)
    plt.close(fig)

    # Chart 3: inbound vs outbound pie
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [summary["inbound_rows"], summary["outbound_rows"]],
        labels=["Inbound (customer)", "Outbound (brand)"],
        autopct="%1.1f%%", startangle=90,
    )
    ax.set_title("Inbound vs Outbound")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "inbound_outbound_ratio.png", dpi=150)
    plt.close(fig)

    log("  Charts saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(args.sample_size)
    summary = compute_summary(df)
    brand_df = compute_brand_table(df)

    # Save results
    summary_path = RESULTS_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved: {summary_path}")

    brand_path = RESULTS_DIR / "brand_table.csv"
    brand_df.to_csv(brand_path, index=False, encoding="utf-8")
    log(f"Saved: {brand_path}")

    # Print summary to a text file (avoids Windows encoding issues)
    report_path = RESULTS_DIR / "exploration_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== DATASET SUMMARY ===\n")
        for k, v in summary.items():
            f.write(f"  {k:<30} {v}\n")
        f.write("\n=== TOP 20 BRANDS ===\n")
        f.write(brand_df.head(20).to_string(index=False))
        f.write("\n")
    log(f"Saved: {report_path}")

    save_charts(df, brand_df, summary)

    log("\nPhase 2 complete. Check reports/results/ and reports/figures/")


if __name__ == "__main__":
    main()
