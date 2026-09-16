"""
build_golden_set.py - Generate golden_200.json from real AppleSupport tweets.

Sampling strategy:
  - Stratified: >=15 examples per intent including 'other'
  - Within each intent: sample across message length quartiles
  - Total target: 200 examples
  - First 50 rows are flagged for double-labelling (kappa computation)
  - Seed=42 for reproducibility

Output:
  data/golden/golden_200.json     - template for human labelling
  data/golden/double_label_50.csv - first 50 rows for second annotator

IMPORTANT: gold_intent, should_escalate, gold_reply_quality are LEFT EMPTY.
A human must fill these in. Never auto-fill them.

Usage:
    python scripts/build_golden_set.py
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_PROCESSED, DATA_GOLDEN, RANDOM_SEED


def main():
    DATA_GOLDEN.mkdir(parents=True, exist_ok=True)

    test_path = DATA_PROCESSED / "test.csv"
    if not test_path.exists():
        print("ERROR: test.csv not found. Run train_baselines.py first.")
        sys.exit(1)

    test = pd.read_csv(test_path)
    print(f"Test set: {len(test):,} examples")
    print(f"Intent distribution:\n{test['intent'].value_counts().to_string()}\n")

    rng = np.random.default_rng(RANDOM_SEED)
    sampled_rows = []

    intents = sorted(test["intent"].unique())
    per_intent_target = max(15, 200 // len(intents))

    for intent in intents:
        intent_df = test[test["intent"] == intent].copy()
        intent_df["text_len"] = intent_df["clean_text"].str.len()

        try:
            intent_df["len_q"] = pd.qcut(
                intent_df["text_len"], q=4, labels=False, duplicates="drop"
            )
        except ValueError:
            intent_df["len_q"] = 0

        n_per_q = max(1, per_intent_target // 4)
        for q in intent_df["len_q"].unique():
            q_df = intent_df[intent_df["len_q"] == q]
            n = min(n_per_q, len(q_df))
            if n > 0:
                sampled_rows.append(q_df.sample(n=n, random_state=RANDOM_SEED))

    golden = pd.concat(sampled_rows, ignore_index=True)
    golden = golden.drop_duplicates(subset="tweet_id").reset_index(drop=True)

    if len(golden) > 200:
        golden = golden.sample(n=200, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Sampled {len(golden)} examples")
    print("Intent distribution in golden set:")
    print(golden["intent"].value_counts().to_string())

    records = []
    for i, row in golden.iterrows():
        records.append({
            "id":                 i + 1,
            "tweet_id":           str(row.get("tweet_id", "")),
            "conversation_id":    str(row.get("conversation_id", "")),
            "text":               str(row["clean_text"]),
            "auto_intent":        str(row["intent"]),
            "gold_intent":        "",
            "should_escalate":    "",
            "gold_reply_quality": "",
            "double_label":       "YES" if i < 50 else "NO",
            "notes":              "",
        })

    out_path = DATA_GOLDEN / "golden_200.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(records)} examples -> {out_path}")

    double_df = pd.DataFrame([r for r in records if r["double_label"] == "YES"])
    double_path = DATA_GOLDEN / "double_label_50.csv"
    double_df[["id", "text", "auto_intent", "gold_intent", "should_escalate",
               "gold_reply_quality", "notes"]].to_csv(double_path, index=False)
    print(f"Saved {len(double_df)} double-label examples -> {double_path}")

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Open data/golden/golden_200.json")
    print("2. Read data/golden/codebook.md for labelling instructions")
    print("3. Fill in gold_intent, should_escalate, gold_reply_quality for ALL rows")
    print("4. Have a second annotator label data/golden/double_label_50.csv")
    print("5. Run: python evaluate.py --golden_set golden_200.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
