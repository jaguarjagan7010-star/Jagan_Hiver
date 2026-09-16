"""
cleaner.py — Cleans and normalises raw tweet text.

What we clean:
- @mentions at the start of tweets (common in Twitter replies, not useful signal)
- URLs (not useful for intent classification)
- Excess whitespace
- We do NOT lowercase or remove punctuation here — the sentence transformer
  handles that internally, and TF-IDF benefits from mixed case for brand names.
"""

import re
import pandas as pd


# Regex patterns compiled once for speed
_MENTION_RE  = re.compile(r"@\w+")          # any @mention
_URL_RE      = re.compile(r"https?://\S+")  # http/https URLs
_WHITESPACE  = re.compile(r"\s+")           # multiple spaces/newlines


def clean_text(text: str) -> str:
    """
    Clean a single tweet string.

    Args:
        text: Raw tweet text.

    Returns:
        Cleaned text string. Returns empty string if input is null/empty.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Remove URLs first (before mentions, to avoid partial matches)
    text = _URL_RE.sub(" ", text)

    # Remove @mentions
    text = _MENTION_RE.sub(" ", text)

    # Collapse whitespace
    text = _WHITESPACE.sub(" ", text).strip()

    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text to the 'text' column and add a 'clean_text' column.
    Rows where clean_text is empty after cleaning are dropped.

    Args:
        df: DataFrame with a 'text' column.

    Returns:
        DataFrame with added 'clean_text' column, empty rows removed.
    """
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped:,} rows with empty text after cleaning.")

    return df
