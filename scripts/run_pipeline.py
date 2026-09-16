"""
run_pipeline.py — One command that runs the complete demonstration pipeline.

Usage:
    python scripts/run_pipeline.py --sample-size 10000
    python scripts/run_pipeline.py --message "My order hasn't arrived"
"""

import sys
import json
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_RAW, RAW_DATA_FILENAME, DATA_PROCESSED, RESULTS_DIR,
    SELECTED_BRAND, RANDOM_SEED, LLM_BACKEND
)


def run_demo_messages(backend: str):
    """Run the pipeline on a set of demo messages and save results."""
    from src.pipeline import run_pipeline, print_result

    demo_messages = [
        "My order hasn't arrived yet and it's been 2 weeks",
        "I was charged twice for the same item",
        "I can't log into my account",
        "How do I return this item?",
        "My Alexa stopped working",
        "I want to cancel my Prime membership",
        "still waiting",   # ambiguous short message
        "This is absolutely terrible service",
    ]

    results = []
    for msg in demo_messages:
        print(f"\nProcessing: {msg[:50]}...")
        try:
            result = run_pipeline(msg, backend=backend)
            print_result(result)
            results.append({
                "message":    msg,
                "intent":     result.get("intent"),
                "confidence": result.get("confidence"),
                "decision":   result.get("decision"),
                "reply":      result.get("reply"),
            })
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"message": msg, "error": str(e)})

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "demo_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nDemo results saved: {out_path}")


def run_full_pipeline(sample_size: int):
    """Run the complete data preparation and training pipeline."""
    import subprocess

    steps = [
        (f"python scripts/train_baselines.py", "Training baselines (Phases 7+8)"),
        (f"python scripts/train_classifier.py", "Training embedding classifier (Phase 9)"),
        (f"python scripts/build_retrieval.py",  "Building retrieval index (Phase 10)"),
    ]

    for cmd, desc in steps:
        print(f"\n{'='*60}")
        print(f"Step: {desc}")
        print(f"{'='*60}")
        ret = subprocess.run(cmd, shell=True, cwd=str(Path(__file__).parent.parent))
        if ret.returncode != 0:
            print(f"Step failed: {cmd}")
            print("Fix the error above and re-run.")
            sys.exit(1)
        print(f"Done: {desc}")


def main():
    parser = argparse.ArgumentParser(description="Hiver AI Support Agent Pipeline")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Number of rows to sample from dataset")
    parser.add_argument("--message", type=str, default=None,
                        help="Run pipeline on a single message")
    parser.add_argument("--backend", type=str, default=LLM_BACKEND,
                        choices=["ollama", "huggingface", "template"],
                        help="LLM backend to use")
    parser.add_argument("--demo-only", action="store_true",
                        help="Skip training, just run demo messages")
    args = parser.parse_args()

    if args.message:
        # Single message mode
        from src.pipeline import run_pipeline, print_result
        result = run_pipeline(args.message, backend=args.backend)
        print_result(result)
        return

    if not args.demo_only:
        print("Running full pipeline...")
        run_full_pipeline(args.sample_size)

    print("\nRunning demo messages...")
    run_demo_messages(backend=args.backend)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
