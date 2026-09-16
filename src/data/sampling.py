"""
sampling.py — Reproducible dataset sampling utilities.
"""

import pandas as pd
from src.config import RANDOM_SEED


def sample_brand(df: pd.DataFrame, brand: str, n: int | None = None) -> pd.DataFrame:
    """
    Filter to a single brand's inbound messages, optionally sample n rows.

    Args:
        df:    Full cleaned DataFrame.
        brand: author_id of the brand's support account (outbound messages
               come FROM this account; inbound messages are TO this account,
               i.e. in_response_to_tweet_id points to a brand tweet).
        n:     If given, sample this many inbound messages.

    Returns:
        DataFrame of inbound customer messages directed at the brand.
    """
    # Inbound = True means the tweet is FROM a customer
    # We identify brand conversations by checking if the brand ever replied
    # (this is done in conversations.py; here we just filter inbound rows
    #  where the brand is the author of the parent tweet)
    brand_tweet_ids = set(df.loc[df["author_id"] == brand, "tweet_id"].dropna())
    mask = df["inbound"] & df["in_response_to_tweet_id"].isin(brand_tweet_ids)
    brand_inbound = df[mask].copy()

    if n is not None and n < len(brand_inbound):
        brand_inbound = brand_inbound.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    return brand_inbound
