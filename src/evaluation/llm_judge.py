"""
llm_judge.py — LLM-as-a-judge rubric for evaluating generated responses.

Rubric dimensions (each scored 1-5):
1. correctness        — Is the reply factually correct given the evidence?
2. relevance          — Does the reply address the customer's actual question?
3. grounding          — Is the reply grounded in the historical examples?
4. helpfulness        — Would this reply actually help the customer?
5. brand_tone         — Is the tone professional and on-brand?
6. no_hallucination   — Does the reply avoid inventing facts? (5=no hallucination)
7. escalation_appropriate — Is the escalation decision appropriate?

Score interpretation:
    5 = Excellent
    4 = Good
    3 = Acceptable
    2 = Poor
    1 = Very poor / wrong
"""

import sys
import json
import re
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import LLM_BACKEND, RESULTS_DIR
from src.generation.prompts import build_judge_prompt, JUDGE_CRITERIA
from src.generation.generator import _call_ollama, _call_huggingface

DIMENSIONS = JUDGE_CRITERIA


def judge_response(
    customer_message: str,
    generated_reply: str,
    historical_examples: list[dict],
    intent: str,
    decision: str = "",
    backend: str = LLM_BACKEND,
) -> dict:
    """
    Ask the LLM to score a generated reply.
    Judge runs at temperature=0 for reproducibility.
    """
    prompt = build_judge_prompt(
        customer_message=customer_message,
        generated_reply=generated_reply,
        historical_examples=historical_examples,
        intent=intent,
        decision=decision,
    )

    try:
        if backend == "ollama":
            # temperature=0 enforced for reproducibility
            raw = _call_ollama(prompt, temperature=0.0)
        elif backend == "huggingface":
            raw = _call_huggingface(prompt)
        else:
            return _fallback_scores()
        return _parse_judge_output(raw)
    except Exception as e:
        print(f"Judge failed: {e}")
        return _fallback_scores()


def _parse_judge_output(raw: str) -> dict:
    """
    Parse the JSON output from the LLM judge.
    Handles cases where the LLM wraps JSON in markdown code blocks.
    """
    # Strip markdown code blocks if present
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)

    # Find the JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return _fallback_scores()

    try:
        data = json.loads(match.group())
        scores = {}
        for dim in DIMENSIONS:
            if dim in data:
                scores[dim] = {
                    "score":  int(data[dim].get("score", 3)),
                    "reason": str(data[dim].get("reason", "")),
                }
            else:
                scores[dim] = {"score": 3, "reason": "not scored"}
        scores["mean_score"] = round(
            sum(v["score"] for v in scores.values() if isinstance(v, dict) and "score" in v)
            / len(DIMENSIONS), 2
        )
        return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        return _fallback_scores()


def _fallback_scores() -> dict:
    """Return neutral scores when the judge fails."""
    scores = {dim: {"score": 3, "reason": "judge unavailable"} for dim in DIMENSIONS}
    scores["mean_score"] = 3.0
    return scores


def run_batch_judge(
    eval_df: pd.DataFrame,
    reply_col: str = "generated_reply",
    evidence_col: str = "evidence",
    backend: str = LLM_BACKEND,
    max_samples: int = 50,
) -> pd.DataFrame:
    """
    Run the LLM judge on a batch of generated replies.

    Args:
        eval_df:     DataFrame with columns: text, intent, generated_reply, evidence.
        reply_col:   Column name for generated replies.
        evidence_col:Column name for evidence (list of dicts, stored as JSON string).
        backend:     LLM backend to use.
        max_samples: Maximum number of samples to judge (cost/time control).

    Returns:
        DataFrame with judge scores added.
    """
    sample = eval_df.head(max_samples).copy()
    results = []

    for i, row in sample.iterrows():
        print(f"Judging {i+1}/{len(sample)}...", end="\r", flush=True)

        # Parse evidence from JSON string if needed
        evidence = row.get(evidence_col, [])
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except Exception:
                evidence = []

        scores = judge_response(
            customer_message=row["text"],
            generated_reply=row.get(reply_col, ""),
            historical_examples=evidence,
            intent=row.get("intent", "unknown"),
            backend=backend,
        )

        row_result = {"id": row.get("id", i)}
        for dim in DIMENSIONS:
            row_result[f"judge_{dim}"] = scores.get(dim, {}).get("score", 3)
            row_result[f"judge_{dim}_reason"] = scores.get(dim, {}).get("reason", "")
        row_result["judge_mean"] = scores.get("mean_score", 3.0)
        results.append(row_result)

    print()
    return pd.DataFrame(results)
