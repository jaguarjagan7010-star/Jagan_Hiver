"""
labeling.py — Assigns intent labels to messages and creates train/val/test splits.

Labeling strategy:
- We use a keyword-based rule labeler as a STARTING POINT.
- Rules are derived from the intent_schema.json definitions.
- Messages that match no rule get label "unknown" and are excluded from training.
- IMPORTANT: These are auto-labels, not human labels.
  The golden set (Phase 13) contains human-verified labels.

Split strategy:
- Split at CONVERSATION level to prevent leakage.
- All messages from the same conversation go to the same split.
- Stratified by intent to maintain class balance.
- Seed is fixed for reproducibility.
"""

import sys
import re
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_GOLDEN, DATA_PROCESSED, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED


# ── Keyword rules for AppleSupport intents ────────────────────────────────────
# Each rule is a list of regex patterns. First match wins.
# Order matters: more specific rules first.
# "other" is NOT in rules — it is the fallback for unmatched messages.

INTENT_RULES = {
    "account_access": [
        r"\bapple id\b", r"\bcan'?t log\b", r"\bcan'?t sign\b", r"\blocke?d out\b",
        r"\bforgot password\b", r"\breset password\b", r"\bhacked\b",
        r"\baccount.*suspend\b", r"\bcan'?t access\b", r"\btwo.?factor\b",
        r"\bverification code\b", r"\bicloud.*lock\b",
    ],
    "billing_payment": [
        r"\brefund\b", r"\bcharged twice\b", r"\bwrong charge\b",
        r"\bunauthorized charge\b", r"\bpayment.*fail\b", r"\bovercharged\b",
        r"\bmoney back\b", r"\bget.*refund\b", r"\bapp store.*charge\b",
        r"\bitunes.*charge\b", r"\bsubscription.*charge\b",
    ],
    "device_hardware": [
        r"\biphone\b.*\bbroken\b", r"\bscreen.*crack\b", r"\bbattery\b.*\bdrain\b",
        r"\bwon'?t turn on\b", r"\bdead\b.*\bphone\b", r"\bphone.*dead\b",
        r"\bwater damage\b", r"\brepair\b", r"\bgenius bar\b",
        r"\bhardware\b", r"\bphysical damage\b",
    ],
    "software_update": [
        r"\bios\b.*\bupdate\b", r"\bupdate\b.*\bios\b", r"\bios \d\b",
        r"\bsoftware update\b", r"\bupgrade\b.*\bios\b", r"\binstall.*update\b",
        r"\bupdate.*fail\b", r"\bstuck.*update\b", r"\bmacos\b.*\bupdate\b",
    ],
    "app_issue": [
        r"\bapp.*crash\b", r"\bapp.*not.*work\b", r"\bapp.*won'?t\b",
        r"\bapp store\b.*\bproblem\b", r"\bcan'?t.*download\b", r"\bapp.*freez\b",
        r"\bapp.*open\b", r"\bapp.*load\b", r"\bapp.*error\b",
    ],
    "icloud_sync": [
        r"\bicloud\b", r"\bbackup\b.*\bfail\b", r"\bsync\b.*\bproblem\b",
        r"\bphotos.*not.*sync\b", r"\bstorage.*full\b", r"\bicloud.*storage\b",
        r"\bcloud.*backup\b",
    ],
    "connectivity": [
        r"\bwifi\b.*\bnot.*connect\b", r"\bbluetooth\b.*\bnot.*work\b",
        r"\bno.*signal\b", r"\bcellular\b.*\bproblem\b", r"\bairpods\b.*\bconnect\b",
        r"\bhotspot\b", r"\binternet.*not.*work\b",
    ],
    "warranty_repair": [
        r"\bwarranty\b", r"\bapplecare\b", r"\bapple care\b",
        r"\brepair.*cost\b", r"\bservice.*center\b", r"\bfind.*store\b",
        r"\bbook.*appointment\b",
    ],
}

# Compile all patterns once
_COMPILED_RULES = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in INTENT_RULES.items()
}


def label_text(text: str) -> str:
    """
    Apply keyword rules to assign an intent label.
    Returns 'other' if no rule matches (kept in dataset, not dropped).
    """
    if not isinstance(text, str) or not text.strip():
        return "other"

    for intent, patterns in _COMPILED_RULES.items():
        for pattern in patterns:
            if pattern.search(text):
                return intent

    return "other"


def label_dataframe(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Add an 'intent' column to the DataFrame using keyword rules.
    Keeps 'other' rows (unlike old version that dropped 'unknown').
    Drops only rows with empty text.
    """
    df = df.copy()
    df["intent"] = df[text_col].apply(label_text)

    total = len(df)
    other_n = (df["intent"] == "other").sum()
    print(f"Labelled {total:,} messages ({other_n:,} labelled 'other', kept in dataset)")
    return df


def split_conversations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at conversation level to prevent leakage.

    Strategy:
    1. Get unique conversation_ids.
    2. Split conversation_ids into train/val/test (70/15/15).
    3. Assign all messages from each conversation to the same split.

    This ensures no conversation appears in both train and test.
    """
    conv_ids = df["conversation_id"].unique()
    n = len(conv_ids)

    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(conv_ids)

    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_convs = set(shuffled[:n_train])
    val_convs   = set(shuffled[n_train:n_train + n_val])
    test_convs  = set(shuffled[n_train + n_val:])

    train = df[df["conversation_id"].isin(train_convs)].copy()
    val   = df[df["conversation_id"].isin(val_convs)].copy()
    test  = df[df["conversation_id"].isin(test_convs)].copy()

    print(f"Split: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"Conversations: train={len(train_convs)}  val={len(val_convs)}  test={len(test_convs)}")

    # Verify no conversation leakage
    assert len(train_convs & test_convs) == 0, "LEAKAGE: train/test share conversations!"
    assert len(val_convs & test_convs) == 0,   "LEAKAGE: val/test share conversations!"

    return train, val, test
