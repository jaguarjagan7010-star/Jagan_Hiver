"""
escalation.py — Evidence Gate: multi-signal AUTO_HANDLE vs ESCALATE decision.

Decision logic (in order):
  1. HIGH-RISK INTENT  → always ESCALATE (account_access, billing_payment)
  2. LOW CONFIDENCE    → ESCALATE if confidence < CLASSIFIER_THRESHOLD
  3. NO EVIDENCE       → ESCALATE if no historical examples retrieved
  4. WEAK EVIDENCE     → ESCALATE if evidence_score < RETRIEVAL_THRESHOLD
  5. RISK COMPOSITE    → ESCALATE if combined_risk_score > 0.5
  6. Otherwise         → AUTO_HANDLE

Combined risk score = 0.5*(1-confidence) + 0.5*(1-evidence_score)
This is the unique Evidence Gate: it combines classifier uncertainty and
retrieval weakness into a single risk signal before making the final call.

Thresholds are set in config.py and tuned on validation data only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CLASSIFIER_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    HIGH_RISK_INTENTS,
)

RISK_COMPOSITE_THRESHOLD = 0.50   # combined risk above this → escalate


def _composite_risk(confidence: float, evidence_score: float) -> float:
    """Weighted combination of classifier uncertainty and retrieval weakness."""
    return round(0.5 * (1.0 - confidence) + 0.5 * (1.0 - evidence_score), 4)


def decide(
    intent: str,
    confidence: float,
    evidence_score: float,
    n_evidence: int,
    classifier_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
    retrieval_threshold:  float = RETRIEVAL_SIMILARITY_THRESHOLD,
) -> dict:
    """
    Evidence Gate decision.

    Returns:
        {
            "decision":              "AUTO_HANDLE" or "ESCALATE",
            "reason":                str,
            "classifier_confidence": float,
            "evidence_score":        float,
            "composite_risk":        float,
            "intent":                str,
            "gate_triggered":        str,   # which gate fired
        }
    """
    risk = _composite_risk(confidence, evidence_score)
    result = {
        "classifier_confidence": round(confidence, 4),
        "evidence_score":        round(evidence_score, 4),
        "composite_risk":        risk,
        "intent":                intent,
    }

    # Gate 1: High-risk intent — always escalate
    if intent in HIGH_RISK_INTENTS:
        result["decision"]      = "ESCALATE"
        result["gate_triggered"] = "high_risk_intent"
        result["reason"] = (
            f"Intent '{intent}' is high-risk (financial/security). "
            "Always requires human review."
        )
        return result

    # Gate 2: Low classifier confidence
    if confidence < classifier_threshold:
        result["decision"]      = "ESCALATE"
        result["gate_triggered"] = "low_confidence"
        result["reason"] = (
            f"Classifier confidence {confidence:.2f} < threshold {classifier_threshold}. "
            "Intent prediction is uncertain."
        )
        return result

    # Gate 3: No historical evidence
    if n_evidence == 0:
        result["decision"]      = "ESCALATE"
        result["gate_triggered"] = "no_evidence"
        result["reason"] = (
            "No historical support examples retrieved. "
            "Cannot generate a grounded response."
        )
        return result

    # Gate 4: Weak retrieval evidence
    if evidence_score < retrieval_threshold:
        result["decision"]      = "ESCALATE"
        result["gate_triggered"] = "weak_evidence"
        result["reason"] = (
            f"Retrieval evidence score {evidence_score:.2f} < threshold {retrieval_threshold}. "
            "Historical examples are not sufficiently similar."
        )
        return result

    # Gate 5: Composite risk (unique Evidence Gate signal)
    if risk > RISK_COMPOSITE_THRESHOLD:
        result["decision"]      = "ESCALATE"
        result["gate_triggered"] = "composite_risk"
        result["reason"] = (
            f"Composite risk score {risk:.2f} > {RISK_COMPOSITE_THRESHOLD} "
            f"(confidence={confidence:.2f}, evidence={evidence_score:.2f}). "
            "Combined uncertainty too high."
        )
        return result

    # All gates passed → AUTO_HANDLE
    result["decision"]      = "AUTO_HANDLE"
    result["gate_triggered"] = "none"
    result["reason"] = (
        f"All gates passed: confidence={confidence:.2f}, "
        f"evidence_score={evidence_score:.2f}, "
        f"composite_risk={risk:.2f}, "
        f"{n_evidence} examples retrieved."
    )
    return result


def tune_thresholds(
    val_confidences: list,
    val_evidence_scores: list,
    val_true_labels: list,
    val_pred_labels: list,
) -> dict:
    """
    Grid-search confidence and retrieval thresholds on validation set.
    Returns the best thresholds found.
    """
    from sklearn.metrics import f1_score
    import numpy as np

    best = {"classifier_threshold": 0.60, "retrieval_threshold": 0.30, "val_f1": 0.0}

    for c_thresh in np.arange(0.3, 0.9, 0.1):
        for r_thresh in np.arange(0.1, 0.6, 0.1):
            decisions = []
            for conf, ev_score, pred in zip(val_confidences, val_evidence_scores, val_pred_labels):
                d = decide(pred, conf, ev_score, n_evidence=1,
                           classifier_threshold=c_thresh,
                           retrieval_threshold=r_thresh)
                decisions.append(d["decision"])

            auto_mask = [d == "AUTO_HANDLE" for d in decisions]
            if sum(auto_mask) < 10:
                continue

            y_true_auto = [l for l, m in zip(val_true_labels, auto_mask) if m]
            y_pred_auto = [l for l, m in zip(val_pred_labels, auto_mask) if m]
            f1 = f1_score(y_true_auto, y_pred_auto, average="macro", zero_division=0)

            if f1 > best["val_f1"]:
                best = {
                    "classifier_threshold": round(float(c_thresh), 2),
                    "retrieval_threshold":  round(float(r_thresh), 2),
                    "val_f1":               round(f1, 4),
                    "auto_handle_rate":     round(sum(auto_mask) / len(decisions), 3),
                }

    return best
