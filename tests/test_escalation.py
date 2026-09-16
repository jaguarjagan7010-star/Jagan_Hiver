"""
test_escalation.py — Unit tests for the Evidence Gate escalation logic.
Run with: pytest tests/
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decision.escalation import decide, HIGH_RISK_INTENTS


# ── AUTO_HANDLE cases ──────────────────────────────────────────────────────────

def test_auto_handle_high_confidence_good_evidence():
    result = decide(
        intent="app_issue",
        confidence=0.85,
        evidence_score=0.70,
        n_evidence=3,
        classifier_threshold=0.60,
        retrieval_threshold=0.30,
    )
    assert result["decision"] == "AUTO_HANDLE"


def test_auto_handle_at_exact_thresholds():
    # confidence=0.60, evidence=0.30 → risk=0.5*(0.4)+0.5*(0.7)=0.55 > 0.50 → ESCALATE via gate 5
    # Use higher values to actually pass all gates
    result = decide(
        intent="connectivity",
        confidence=0.75,
        evidence_score=0.60,
        n_evidence=2,
        classifier_threshold=0.60,
        retrieval_threshold=0.30,
    )
    assert result["decision"] == "AUTO_HANDLE"


# ── ESCALATE cases ─────────────────────────────────────────────────────────────

def test_escalate_low_confidence():
    result = decide(
        intent="app_issue",
        confidence=0.40,
        evidence_score=0.80,
        n_evidence=3,
        classifier_threshold=0.60,
        retrieval_threshold=0.30,
    )
    assert result["decision"] == "ESCALATE"
    assert result["gate_triggered"] == "low_confidence"


def test_escalate_no_evidence():
    result = decide(
        intent="app_issue",
        confidence=0.90,
        evidence_score=0.0,
        n_evidence=0,
        classifier_threshold=0.60,
        retrieval_threshold=0.30,
    )
    assert result["decision"] == "ESCALATE"
    assert result["gate_triggered"] == "no_evidence"


def test_escalate_weak_evidence_score():
    result = decide(
        intent="app_issue",
        confidence=0.90,
        evidence_score=0.10,
        n_evidence=3,
        classifier_threshold=0.60,
        retrieval_threshold=0.30,
    )
    assert result["decision"] == "ESCALATE"
    assert result["gate_triggered"] == "weak_evidence"


def test_escalate_composite_risk():
    # confidence=0.65, evidence=0.35 → risk=0.5*0.35+0.5*0.65=0.50 → NOT > 0.50, passes
    # Use values that push risk above 0.50
    result = decide(
        intent="software_update",
        confidence=0.62,
        evidence_score=0.32,
        n_evidence=2,
        classifier_threshold=0.60,
        retrieval_threshold=0.30,
    )
    # risk = 0.5*(1-0.62) + 0.5*(1-0.32) = 0.19 + 0.34 = 0.53 > 0.50 → ESCALATE
    assert result["decision"] == "ESCALATE"
    assert result["gate_triggered"] == "composite_risk"


def test_always_escalate_billing():
    result = decide(
        intent="billing_payment",
        confidence=0.99,
        evidence_score=0.99,
        n_evidence=5,
    )
    assert result["decision"] == "ESCALATE"
    assert result["gate_triggered"] == "high_risk_intent"


def test_always_escalate_account_access():
    result = decide(
        intent="account_access",
        confidence=0.99,
        evidence_score=0.99,
        n_evidence=5,
    )
    assert result["decision"] == "ESCALATE"
    assert result["gate_triggered"] == "high_risk_intent"


# ── Output schema tests ────────────────────────────────────────────────────────

def test_result_has_required_keys():
    result = decide("app_issue", 0.8, 0.6, 3)
    for key in ("decision", "reason", "classifier_confidence",
                "evidence_score", "composite_risk", "intent", "gate_triggered"):
        assert key in result


def test_decision_is_valid_value():
    for intent in ["app_issue", "billing_payment", "connectivity"]:
        result = decide(intent, 0.8, 0.6, 3)
        assert result["decision"] in ("AUTO_HANDLE", "ESCALATE")


def test_high_risk_intents_set():
    assert "billing_payment" in HIGH_RISK_INTENTS
    assert "account_access" in HIGH_RISK_INTENTS


def test_composite_risk_range():
    result = decide("app_issue", 0.8, 0.6, 3)
    assert 0.0 <= result["composite_risk"] <= 1.0
