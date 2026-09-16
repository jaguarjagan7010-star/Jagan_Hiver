"""
test_classifier.py — Unit tests for intent classifiers.
Run with: pytest tests/
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.intents.classifier import MajorityClassifier, TfidfLRClassifier


TRAIN_TEXTS = [
    "my iPhone won't turn on",
    "screen is cracked",
    "battery drains too fast",
    "I want a refund for my App Store purchase",
    "charged twice on my account",
    "unauthorized charge on iTunes",
    "can't log into my Apple ID",
    "forgot my password",
    "account is locked out",
]
TRAIN_LABELS = [
    "device_hardware", "device_hardware", "device_hardware",
    "billing_payment", "billing_payment", "billing_payment",
    "account_access", "account_access", "account_access",
]

TEST_TEXTS = ["my phone screen broke", "I need a refund please", "reset my Apple ID"]


# ── MajorityClassifier ─────────────────────────────────────────────────────────

def test_majority_predicts_most_frequent():
    clf = MajorityClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    preds = clf.predict(TEST_TEXTS)
    assert all(p == clf.majority_class_ for p in preds)


def test_majority_proba_sums_to_one():
    clf = MajorityClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    probas = clf.predict_proba(TEST_TEXTS)
    for p in probas:
        assert abs(sum(p.values()) - 1.0) < 1e-6


def test_majority_confidence_is_one():
    clf = MajorityClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    _, conf, _ = clf.predict_with_confidence("any message")
    assert conf == 1.0


def test_majority_empty_input():
    clf = MajorityClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    preds = clf.predict([])
    assert preds == []


# ── TfidfLRClassifier ──────────────────────────────────────────────────────────

def test_tfidf_lr_fits_and_predicts():
    clf = TfidfLRClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    preds = clf.predict(TEST_TEXTS)
    assert len(preds) == len(TEST_TEXTS)
    assert all(p in TRAIN_LABELS for p in preds)


def test_tfidf_lr_proba_valid():
    clf = TfidfLRClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    probas = clf.predict_proba(TEST_TEXTS)
    for p in probas:
        assert abs(sum(p.values()) - 1.0) < 1e-4
        assert all(0.0 <= v <= 1.0 for v in p.values())


def test_tfidf_lr_confidence_in_range():
    clf = TfidfLRClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    _, conf, _ = clf.predict_with_confidence("my iPhone screen is broken")
    assert 0.0 <= conf <= 1.0


def test_tfidf_lr_returns_all_classes():
    clf = TfidfLRClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    _, _, proba = clf.predict_with_confidence("test message")
    assert set(proba.keys()) == set(TRAIN_LABELS)


def test_tfidf_lr_malformed_input():
    clf = TfidfLRClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)
    preds = clf.predict(["", "   ", "!!!"])
    assert len(preds) == 3
