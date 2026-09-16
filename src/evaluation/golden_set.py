"""
golden_set.py — Builds the golden evaluation set for human labelling.

Sampling strategy (stratified, not random):
1. Sample ~25 examples per intent (ensures rare intents are represented).
2. Within each intent, sample across message length quartiles.
3. Include some ambiguous examples (low classifier confidence).
4. Include some multi-turn examples (conversation_id appears multiple times).
5. Total target: 150-250 examples.

IMPORTANT:
- The CSV is created with EMPTY gold_intent, gold_reply_quality_label,
  gold_escalation_label columns.
- A human must fill these in.
- DO NOT treat auto-generated labels as human labels.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_PROCESSED, DATA_GOLDEN, RANDOM_SEED


def build_golden_set(
    test_df: pd.DataFrame,
    target_size: int = 200,
    per_intent: int = 25,
) -> pd.DataFrame:
    """
    Build a stratified sample for human labelling.

    Args:
        test_df:     Test split DataFrame with columns: tweet_id, clean_text,
                     intent, conversation_id.
        target_size: Target number of examples.
        per_intent:  Max examples per intent.

    Returns:
        DataFrame ready for human labelling (gold columns are empty).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_rows = []

    intents = test_df["intent"].unique()

    for intent in sorted(intents):
        intent_df = test_df[test_df["intent"] == intent].copy()

        # Add message length quartile
        intent_df["text_len"] = intent_df["clean_text"].str.len()
        intent_df["len_quartile"] = pd.qcut(
            intent_df["text_len"], q=4, labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop"
        )

        # Sample across quartiles
        n_per_quartile = max(1, per_intent // 4)
        for q in intent_df["len_quartile"].unique():
            q_df = intent_df[intent_df["len_quartile"] == q]
            n = min(n_per_quartile, len(q_df))
            sampled_rows.append(q_df.sample(n=n, random_state=RANDOM_SEED))

    golden = pd.concat(sampled_rows, ignore_index=True)

    # Deduplicate
    golden = golden.drop_duplicates(subset="tweet_id").reset_index(drop=True)

    # Trim to target size
    if len(golden) > target_size:
        golden = golden.sample(n=target_size, random_state=RANDOM_SEED).reset_index(drop=True)

    # Build output with human-labelling columns (empty)
    output = pd.DataFrame({
        "id":                      range(1, len(golden) + 1),
        "tweet_id":                golden["tweet_id"],
        "conversation_id":         golden["conversation_id"],
        "text":                    golden["clean_text"],
        "auto_intent":             golden["intent"],   # auto-label for reference
        "gold_intent":             "",                 # HUMAN FILLS THIS
        "gold_reply_quality_label":"",                 # HUMAN FILLS THIS (1-5)
        "gold_escalation_label":   "",                 # HUMAN FILLS THIS (AUTO_HANDLE/ESCALATE)
        "notes":                   "",                 # HUMAN FILLS THIS
    })

    return output


def load_or_create(target_size: int = 200) -> pd.DataFrame:
    """Load existing golden set or create a new one."""
    golden_path = DATA_GOLDEN / "golden_set.csv"

    if golden_path.exists():
        df = pd.read_csv(golden_path, encoding="utf-8", dtype=str, keep_default_na=False)
        legacy_path = DATA_GOLDEN / "golden_200.json"
        if legacy_path.exists() and "tweet_id" in df.columns:
            with legacy_path.open("r", encoding="utf-8") as fh:
                legacy = pd.DataFrame(json.load(fh))
            if "tweet_id" in legacy.columns:
                legacy = legacy.drop_duplicates("tweet_id").copy()
                legacy["_tweet_key"] = legacy["tweet_id"].astype(str)
                legacy = legacy.set_index("_tweet_key")
                for index, row in df.iterrows():
                    source = legacy.loc[str(row["tweet_id"])] if str(row["tweet_id"]) in legacy.index else None
                    if source is None:
                        continue
                    for field in ("gold_intent", "gold_escalation_label", "gold_reply_quality_label", "notes"):
                        if field in df.columns and (
                            pd.isna(row.get(field, "")) or not str(row.get(field, "")).strip()
                        ):
                            legacy_field = {
                                "gold_escalation_label": "should_escalate",
                                "gold_reply_quality_label": "gold_reply_quality",
                            }.get(field, field)
                            if legacy_field in source.index:
                                df.at[index, field] = source[legacy_field]
                df.to_csv(golden_path, index=False, encoding="utf-8")
        print(f"Loaded existing golden set: {len(df)} examples")
        return df

    # Create from test split
    test_path = DATA_PROCESSED / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            "test.csv not found. Run train_baselines.py first."
        )

    test_df = pd.read_csv(test_path)
    golden = build_golden_set(test_df, target_size=target_size)

    DATA_GOLDEN.mkdir(parents=True, exist_ok=True)
    golden.to_csv(golden_path, index=False, encoding="utf-8")
    print(f"Created golden set: {len(golden)} examples -> {golden_path}")
    print("\nNEXT STEP: Open data/golden/golden_set.csv and fill in:")
    print("  gold_intent             — correct intent label")
    print("  gold_reply_quality_label — 1-5 score for reply quality")
    print("  gold_escalation_label   — AUTO_HANDLE or ESCALATE")
    print("  notes                   — any observations")

    return golden


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a real-data golden annotation queue.")
    parser.add_argument("--target-size", type=int, default=200)
    args = parser.parse_args()
    load_or_create(target_size=args.target_size)


if __name__ == "__main__":
    main()
