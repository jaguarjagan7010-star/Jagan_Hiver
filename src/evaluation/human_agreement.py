"""
human_agreement.py — Computes agreement between human and LLM judge scores.

Agreement metric: Cohen's Kappa (for ordinal scores 1-5).
Also reports: Pearson correlation, mean absolute error, exact agreement rate.

IMPORTANT: If sample size is small (<30), results are not statistically reliable.
We report this limitation explicitly.
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RESULTS_DIR
from src.generation.prompts import JUDGE_CRITERIA

DIMENSIONS = JUDGE_CRITERIA


def compute_agreement(
    human_scores: pd.DataFrame,
    llm_scores: pd.DataFrame,
    id_col: str = "id",
) -> dict:
    """
    Compute agreement statistics between human and LLM judge scores.

    Args:
        human_scores: DataFrame with columns: id, correctness, relevance, ...
        llm_scores:   DataFrame with columns: id, judge_correctness, ...

    Returns:
        Dict of agreement statistics per dimension + overall.
    """
    # Merge on id
    merged = human_scores.merge(llm_scores, on=id_col, how="inner")
    n = len(merged)

    results = {
        "n_samples": n,
        "warning": "Results not statistically reliable (n<30)" if n < 30 else None,
        "dimensions": {},
    }

    all_human, all_llm = [], []

    for dim in DIMENSIONS:
        human_col = dim                  # human CSV uses plain names
        llm_col   = f"judge_{dim}"       # LLM judge uses judge_ prefix

        if human_col not in merged.columns or llm_col not in merged.columns:
            continue

        h = merged[human_col].dropna().astype(int)
        l = merged[llm_col].dropna().astype(int)

        # Align indices
        common_idx = h.index.intersection(l.index)
        h = h.loc[common_idx].tolist()
        l = l.loc[common_idx].tolist()

        if len(h) < 2:
            continue

        all_human.extend(h)
        all_llm.extend(l)

        try:
            kappa = cohen_kappa_score(h, l, weights="quadratic")
        except Exception:
            kappa = float("nan")

        corr = float(np.corrcoef(h, l)[0, 1]) if len(h) > 1 else float("nan")
        mae  = float(np.mean(np.abs(np.array(h) - np.array(l))))
        exact = float(np.mean(np.array(h) == np.array(l)))

        results["dimensions"][dim] = {
            "n":              len(h),
            "kappa":          round(kappa, 3),
            "pearson_r":      round(corr, 3),
            "mae":            round(mae, 3),
            "exact_agreement":round(exact, 3),
        }

    # Overall
    if all_human:
        try:
            overall_kappa = cohen_kappa_score(all_human, all_llm, weights="quadratic")
        except Exception:
            overall_kappa = float("nan")
        results["overall"] = {
            "kappa":          round(overall_kappa, 3),
            "pearson_r":      round(float(np.corrcoef(all_human, all_llm)[0, 1]), 3),
            "mae":            round(float(np.mean(np.abs(np.array(all_human) - np.array(all_llm)))), 3),
            "exact_agreement":round(float(np.mean(np.array(all_human) == np.array(all_llm))), 3),
        }

    return results


def create_human_eval_template(eval_df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Create a CSV template for human evaluators to fill in.
    Samples n rows from the evaluation DataFrame.
    """
    sample = eval_df.sample(n=min(n, len(eval_df)), random_state=42)
    template = pd.DataFrame({
        "id": sample["id"].values if "id" in sample.columns else range(len(sample)),
        "text": sample["text"].values,
        "intent": (
            sample["intent"].values if "intent" in sample.columns
            else sample["auto_intent"].values if "auto_intent" in sample.columns
            else [""] * len(sample)
        ),
        "generated_reply": (
            sample["generated_reply"].values if "generated_reply" in sample.columns
            else [""] * len(sample)
        ),
        "notes": [""] * len(sample),
    })
    for dimension in DIMENSIONS:
        template[dimension] = ""
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blank human judge calibration template.")
    parser.add_argument("--source", default="data/golden/golden_set.csv")
    parser.add_argument("--output", default="data/golden/human_scores_50.csv")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    source = Path(args.source)
    if source.suffix.lower() == ".json":
        eval_df = pd.DataFrame(json.loads(source.read_text(encoding="utf-8")))
    else:
        eval_df = pd.read_csv(source, dtype=str, keep_default_na=False)
    if "text" not in eval_df.columns:
        raise ValueError("Calibration source must contain a text column.")

    template = create_human_eval_template(eval_df, n=args.n)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)
    print(f"Created blank human calibration template: {output}")
    print("Fill the seven score columns with integers from 1 to 5 after judge outputs exist.")


if __name__ == "__main__":
    main()
