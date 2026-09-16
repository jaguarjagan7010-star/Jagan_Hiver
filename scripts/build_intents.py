"""
build_intents.py — Runs clustering-aided intent discovery and saves results.
Also validates the intent_schema.json against real data.

Usage:
    python scripts/build_intents.py
"""

import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW, RAW_DATA_FILENAME, DATA_GOLDEN, RESULTS_DIR, RANDOM_SEED
from src.data.cleaner import clean_text
from src.intents.discovery import discover_intents


def load_brand_inbound(brand="AppleSupport", sample_n=5000):
    dtype_map = {"tweet_id": str, "author_id": str,
                 "response_tweet_id": str, "in_response_to_tweet_id": str}
    df = pd.read_csv(DATA_RAW / RAW_DATA_FILENAME, dtype=dtype_map, low_memory=False)
    df["inbound"] = df["inbound"].map(
        lambda x: True if str(x).strip().lower() == "true" else False)
    for col in ["tweet_id", "author_id", "response_tweet_id", "in_response_to_tweet_id"]:
        df[col] = df[col].astype(str).str.strip().replace("nan", None)

    brand_tweet_ids = set(
        df.loc[(~df["inbound"]) & (df["author_id"] == brand), "tweet_id"].dropna()
    )
    inbound = df[df["inbound"] & df["in_response_to_tweet_id"].isin(brand_tweet_ids)].copy()
    inbound["clean_text"] = inbound["text"].apply(clean_text)
    inbound = inbound[inbound["clean_text"].str.len() > 5]

    if sample_n and len(inbound) > sample_n:
        inbound = inbound.sample(n=sample_n, random_state=RANDOM_SEED)

    return inbound.reset_index(drop=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "intent_discovery_log.txt", "w", encoding="utf-8") as log:
        log.write("Loading brand inbound messages...\n"); log.flush()
        df = load_brand_inbound(brand="AppleSupport", sample_n=5000)
        log.write(f"Loaded {len(df)} messages\n"); log.flush()

        log.write("Running clustering...\n"); log.flush()
        cluster_df, labels = discover_intents(
            texts=df["clean_text"].tolist(),
            n_clusters=8,
            top_terms=15,
        )

        cluster_path = RESULTS_DIR / "cluster_discovery.csv"
        cluster_df.to_csv(cluster_path, index=False, encoding="utf-8")

        log.write("\n=== CLUSTER DISCOVERY RESULTS ===\n")
        log.write("(Use these to validate/refine intent_schema.json)\n\n")
        log.write(cluster_df.to_string(index=False))
        log.write("\n\nNote: Cluster labels are NOT ground truth intents.\n")
        log.write("Review data/golden/intent_schema.json for final definitions.\n")

        # Validate intent schema exists
        schema_path = DATA_GOLDEN / "intent_schema.json"
        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as f:
                schema = json.load(f)
            intents = [k for k in schema.keys() if not k.startswith("_")]
            log.write(f"\nIntent schema loaded: {len(intents)} intents defined\n")
            for intent in intents:
                log.write(f"  - {intent}\n")
        else:
            log.write("\nWARNING: intent_schema.json not found!\n")

        log.write("\nPhase 5 complete.\n")

    print("Done. Check reports/results/intent_discovery_log.txt")


if __name__ == "__main__":
    main()
