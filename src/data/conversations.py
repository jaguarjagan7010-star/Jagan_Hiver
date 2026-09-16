"""
conversations.py — Reconstructs customer/support conversation pairs from raw tweets.

Assumptions documented:
1. A tweet with inbound=True is FROM a customer.
2. A tweet with inbound=False is FROM a brand/support account.
3. A conversation pair = one inbound customer tweet + the brand's direct reply.
4. We use in_response_to_tweet_id to link replies to their parent.
5. Multi-turn: we follow the chain up to MAX_TURNS deep.
6. Missing tweets (deleted/unavailable) break the chain — we stop there.
7. Duplicate tweet_ids: we keep the first occurrence.
8. Malformed IDs (non-numeric strings): treated as missing.

Output columns for pairs:
    conversation_id       — unique ID for the conversation thread
    customer_tweet_id     — tweet_id of the customer message
    customer_text         — raw text of the customer message
    brand_tweet_id        — tweet_id of the brand reply
    brand_text            — raw text of the brand reply
    created_at            — timestamp of the customer message
    turn                  — turn number within the conversation (1 = first)
"""

import sys
import re
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

MAX_TURNS = 5   # maximum conversation depth to follow


def is_valid_id(val) -> bool:
    """Check if a tweet ID is a valid numeric string."""
    if pd.isna(val) or val is None:
        return False
    return bool(re.match(r"^\d+$", str(val).strip()))


def reconstruct_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Build a DataFrame of (customer_message, brand_reply) pairs for a brand.

    Args:
        df:    Full cleaned DataFrame (all brands).
        brand: author_id of the selected brand.

    Returns:
        DataFrame of conversation pairs.
    """
    # Remove duplicate tweet_ids — keep first occurrence
    df = df.drop_duplicates(subset="tweet_id", keep="first").copy()

    # Build fast lookup: tweet_id -> row
    tweet_index = df.set_index("tweet_id")

    # Brand outbound tweets (replies sent by the brand)
    brand_replies = df[
        (~df["inbound"]) &
        (df["author_id"] == brand) &
        df["in_response_to_tweet_id"].apply(is_valid_id)
    ].copy()

    pairs = []
    seen_customer_ids = set()   # avoid duplicate pairs

    for _, reply_row in brand_replies.iterrows():
        parent_id = reply_row["in_response_to_tweet_id"]

        # The parent must exist in our dataset
        if parent_id not in tweet_index.index:
            continue

        parent_row = tweet_index.loc[parent_id]

        # Parent must be an inbound (customer) tweet
        if not parent_row["inbound"]:
            continue

        # Avoid processing the same customer tweet twice
        if parent_id in seen_customer_ids:
            continue
        seen_customer_ids.add(parent_id)

        # Determine conversation_id: trace back to the root tweet
        conv_id = _find_root(parent_id, tweet_index)

        pairs.append({
            "conversation_id":   conv_id,
            "customer_tweet_id": parent_id,
            "customer_text":     parent_row["text"],
            "brand_tweet_id":    reply_row["tweet_id"],
            "brand_text":        reply_row["text"],
            "created_at":        parent_row["created_at"],
            "turn":              1,   # simplified: all pairs are turn=1 for now
        })

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return pairs_df

    pairs_df = pairs_df.sort_values("created_at").reset_index(drop=True)
    return pairs_df


def _find_root(tweet_id: str, tweet_index: pd.DataFrame, depth: int = 0) -> str:
    """
    Walk up the reply chain to find the root tweet_id.
    Stops at MAX_TURNS depth or when the parent is missing.
    Returns the deepest reachable ancestor as the conversation_id.
    """
    if depth >= MAX_TURNS:
        return tweet_id

    try:
        row = tweet_index.loc[tweet_id]
        parent_id = row["in_response_to_tweet_id"]
        if is_valid_id(parent_id) and parent_id in tweet_index.index:
            return _find_root(parent_id, tweet_index, depth + 1)
    except (KeyError, TypeError):
        pass

    return tweet_id


def get_brand_inbound(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Return all inbound customer messages directed at the brand.
    Used for intent classification (includes messages without a reply).
    """
    df = df.drop_duplicates(subset="tweet_id", keep="first").copy()

    brand_tweet_ids = set(
        df.loc[(~df["inbound"]) & (df["author_id"] == brand), "tweet_id"].dropna()
    )

    inbound = df[
        df["inbound"] &
        df["in_response_to_tweet_id"].isin(brand_tweet_ids)
    ].copy()

    return inbound.reset_index(drop=True)
