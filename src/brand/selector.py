"""
selector.py — Objective brand-selection procedure.

Ranks all brand accounts by multiple criteria and selects the best one
for building a support agent. Selection is data-driven, not by fame.

Criteria (all computed from actual data):
1. inbound_msgs       — volume of customer messages (more = richer training data)
2. conversations      — number of distinct conversation threads
3. usable_pairs       — customer messages that have a brand reply (for retrieval)
4. reply_rate         — fraction of inbound msgs that got a reply
5. avg_conv_length    — average turns per conversation (diversity proxy)

We select the brand with the best combined score across these criteria.
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_PROCESSED, RESULTS_DIR


def score_brands(brand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a composite score to the brand table.
    Each metric is min-max normalised to [0,1] then averaged.
    """
    df = brand_df.copy()

    # Only consider brands with meaningful volume (>=100 inbound msgs)
    df = df[df["inbound_msgs"] >= 100].copy()

    metrics = ["inbound_msgs", "conversations", "usable_pairs", "reply_rate"]

    for col in metrics:
        if col not in df.columns:
            continue
        mn, mx = df[col].min(), df[col].max()
        if mx > mn:
            df[f"{col}_norm"] = (df[col] - mn) / (mx - mn)
        else:
            df[f"{col}_norm"] = 0.0

    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    df["composite_score"] = df[norm_cols].mean(axis=1)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def select_brand(brand_df: pd.DataFrame) -> str:
    """
    Select the best brand using the composite score.
    Returns the brand author_id string.
    """
    scored = score_brands(brand_df)
    if scored.empty:
        raise ValueError("No brands with sufficient data found.")
    best = scored.iloc[0]["brand"]
    return best, scored


def compute_usable_pairs(df: pd.DataFrame, brand_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each brand, count how many inbound messages received a direct reply.
    A 'usable pair' = inbound tweet whose tweet_id appears in some
    outbound tweet's in_response_to_tweet_id.
    """
    # outbound tweets that are replies: in_response_to_tweet_id is set
    outbound_replies = df[
        ~df["inbound"] & df["in_response_to_tweet_id"].notna()
    ][["author_id", "in_response_to_tweet_id"]].copy()
    outbound_replies.columns = ["brand", "replied_to_tweet_id"]

    # Map tweet_id -> brand for outbound tweets
    tweet_to_brand = (
        df[~df["inbound"] & df["tweet_id"].notna()]
        [["tweet_id", "author_id"]]
        .drop_duplicates("tweet_id")
        .set_index("tweet_id")["author_id"]
    )

    # Inbound tweets with a known brand
    inb = df[df["inbound"] & df["in_response_to_tweet_id"].notna()].copy()
    inb["brand"] = inb["in_response_to_tweet_id"].map(tweet_to_brand)
    inb = inb[inb["brand"].notna()]

    # Which inbound tweet_ids got a reply?
    replied_ids = set(outbound_replies["replied_to_tweet_id"].dropna())
    inb["got_reply"] = inb["tweet_id"].isin(replied_ids)

    usable = inb.groupby("brand").agg(
        usable_pairs=("got_reply", "sum"),
        total_inbound=("tweet_id", "count"),
    ).reset_index()
    usable["reply_rate"] = (usable["usable_pairs"] / usable["total_inbound"]).round(3)

    # Merge into brand_df
    result = brand_df.merge(usable[["brand", "usable_pairs", "reply_rate"]], on="brand", how="left")
    result["usable_pairs"] = result["usable_pairs"].fillna(0).astype(int)
    result["reply_rate"]   = result["reply_rate"].fillna(0.0)
    return result
