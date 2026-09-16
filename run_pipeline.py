"""
run_pipeline.py — Complete pipeline runner for the Hiver Support Agent.

Required commands:
    python run_pipeline.py --brand AppleSupport --sample
    python run_pipeline.py --brand AppleSupport --message "My iPhone won't turn on"
    python run_pipeline.py --brand AppleSupport --demo-only --backend template

Usage:
    python run_pipeline.py --brand AppleSupport --sample
        Samples data, trains all models, builds retrieval index, runs demo.

    python run_pipeline.py --brand AppleSupport --message "My iPhone won't turn on"
        Runs the pipeline on a single message.

    python run_pipeline.py --brand AppleSupport --demo-only --backend ollama
        Skips training, runs demo messages only.
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import (
    DATA_RAW, RAW_DATA_FILENAME, DATA_PROCESSED, RESULTS_DIR,
    SELECTED_BRAND, RANDOM_SEED, LLM_BACKEND,
)

ROOT = Path(__file__).resolve().parent

# Apple-specific demo messages
APPLE_DEMO_MESSAGES = [
    "My iPhone won't turn on after the latest iOS update",
    "I was charged twice on my App Store purchase",
    "I can't log into my Apple ID, it says my account is locked",
    "My AirPods won't connect to my iPhone",
    "How do I check my AppleCare warranty status?",
    "iCloud backup keeps failing with not enough storage",
    "The screen on my MacBook is cracked, how do I get it repaired?",
    "App keeps crashing every time I open it",
    "still waiting",                          # ambiguous short message
    "This is absolutely terrible service",    # general feedback / other
]


def run_sample(brand: str):
    """Step 0: Sample data for the brand."""
    print(f"\n{'='*60}")
    print(f"Step 0: Sampling data for {brand}")
    print(f"{'='*60}")
    ret = subprocess.run(
        [sys.executable, "scripts/sample_data.py", "--brand", brand],
        cwd=str(ROOT),
    )
    if ret.returncode != 0:
        print("Sampling failed. Check the error above.")
        sys.exit(1)


def run_training(brand: str):
    """Steps 1-3: Train baselines, embedding classifier, build retrieval."""
    steps = [
        (["scripts/train_baselines.py", "--brand", brand],
         "Training baselines (Majority + TF-IDF)"),
        (["scripts/train_classifier.py", "--brand", brand],
         "Training embedding classifier"),
        (["scripts/build_retrieval.py", "--brand", brand],
         "Building FAISS retrieval index"),
    ]

    for script_args, desc in steps:
        print(f"\n{'='*60}")
        print(f"Step: {desc}")
        print(f"{'='*60}")
        ret = subprocess.run(
            [sys.executable] + script_args,
            cwd=str(ROOT),
        )
        if ret.returncode != 0:
            print(f"Step failed: {script_args[0]}")
            sys.exit(1)
        print(f"Done: {desc}")


def run_demo_messages(brand: str, backend: str):
    """Run the pipeline on Apple-specific demo messages."""
    from src.pipeline import run_pipeline, print_result

    results = []
    for msg in APPLE_DEMO_MESSAGES:
        print(f"\nProcessing: {msg[:60]}")
        try:
            result = run_pipeline(msg, brand=brand, backend=backend)
            print_result(result)
            results.append({
                "message":       msg,
                "intent":        result.get("intent"),
                "confidence":    result.get("confidence"),
                "evidence_score":result.get("evidence_score"),
                "composite_risk":result.get("decision_reason", ""),
                "decision":      result.get("decision"),
                "gate_triggered":result.get("decision_reason", ""),
                "reply":         result.get("reply"),
                "backend":       result.get("backend"),
            })
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"message": msg, "error": str(e)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "demo_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nDemo results saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Hiver AI Support Agent — AppleSupport")
    parser.add_argument("--brand",    default=SELECTED_BRAND,
                        help="Brand to run (default: AppleSupport)")
    parser.add_argument("--sample",   action="store_true",
                        help="Sample data before training")
    parser.add_argument("--message",  type=str, default=None,
                        help="Run pipeline on a single message")
    parser.add_argument("--backend",  default=LLM_BACKEND,
                        choices=["ollama", "huggingface", "template"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--demo-only", action="store_true",
                        help="Skip training, run demo messages only")
    parser.add_argument("--sample-size", type=int, default=750,
                        help="Number of tweets to sample (500-1000)")
    args = parser.parse_args()

    # Single message mode
    if args.message:
        from src.pipeline import run_pipeline, print_result
        result = run_pipeline(args.message, brand=args.brand, backend=args.backend)
        print_result(result)
        return

    if not args.demo_only:
        if args.sample:
            run_sample(args.brand)
        print(f"\nRunning full training pipeline for brand: {args.brand}")
        run_training(args.brand)

    print(f"\nRunning demo messages for brand: {args.brand}...")
    run_demo_messages(brand=args.brand, backend=args.backend)
    print("\nPipeline complete.")
    print(f"Results: {RESULTS_DIR / 'demo_results.csv'}")


if __name__ == "__main__":
    main()
