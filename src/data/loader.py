"""
loader.py — Loads the raw Twitter customer-support CSV into a DataFrame.

Key design decisions:
- tweet_id and response IDs are kept as strings (they are 64-bit integers that
  lose precision if read as float64, which pandas does when NaNs are present).
- inbound is cast to bool.
- created_at is parsed as datetime.
- Sampling is done BEFORE any heavy processing to keep memory low.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_RAW, RAW_DATA_FILENAME, RANDOM_SEED


# Columns we actually need — ignore nothing, the dataset is small enough
USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# Force ID columns to string so we never lose precision on large tweet IDs
DTYPE_MAP = {
    "tweet_id":               str,
    "author_id":              str,
    "response_tweet_id":      str,
    "in_response_to_tweet_id": str,
}


def load_raw(sample_size: int | None = None) -> pd.DataFrame:
    """
    Load the raw dataset.

    Args:
        sample_size: If given, return a reproducible random sample of this
                     many rows. If None, load the full dataset.

    Returns:
        DataFrame with cleaned dtypes.
    """
    csv_path = DATA_RAW / RAW_DATA_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}\n"
            "Run: python scripts/download_data.py"
        )

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype=DTYPE_MAP,
        parse_dates=["created_at"],
        low_memory=False,
    )

    # Cast inbound: the CSV stores True/False as strings in some versions
    df["inbound"] = df["inbound"].map(
        lambda x: True if str(x).strip().lower() == "true" else False
    )

    # Strip whitespace from ID columns (sometimes trailing spaces appear)
    for col in ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]:
        df[col] = df[col].str.strip()

    # Replace the literal string "nan" that comes from dtype=str conversion
    for col in ["response_tweet_id", "in_response_to_tweet_id"]:
        df[col] = df[col].replace("nan", pd.NA)

    print(f"Loaded {len(df):,} rows.")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Sampled {len(df):,} rows (seed={RANDOM_SEED}).")

    return df
