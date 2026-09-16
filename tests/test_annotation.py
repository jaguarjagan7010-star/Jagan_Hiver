import json
from pathlib import Path

import pytest

from src.evaluation.annotation import (
    load_intent_schema,
    validate_annotation,
    validate_annotation_file,
    initialize_double_subset,
)


def test_intent_schema_has_nine_intents():
    schema = load_intent_schema(Path("data/golden/intent_schema.json"))
    intents = sorted([k for k in schema.keys() if not k.startswith("_")])
    assert intents == sorted([
        "account_access",
        "billing_payment",
        "device_hardware",
        "software_update",
        "app_issue",
        "icloud_sync",
        "connectivity",
        "warranty_repair",
        "other",
    ])


def test_validate_annotation_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_annotation({
            "id": 1,
            "text": "hello",
            "conversation_id": "abc",
            "gold_intent": "not_real",
            "gold_escalation": "AUTO_HANDLE",
            "escalation_reason": "Routine issue",
            "notes": "",
        })

    with pytest.raises(ValueError):
        validate_annotation({
            "id": 1,
            "text": "hello",
            "conversation_id": "abc",
            "gold_intent": "account_access",
            "gold_escalation": "MAYBE",
            "escalation_reason": "Routine issue",
            "notes": "",
        })


def test_initialize_double_subset_creates_required_fields(tmp_path):
    source = tmp_path / "golden.json"
    source.write_text(json.dumps([
        {"id": 1, "text": "hello", "conversation_id": "c1", "gold_intent": "", "should_escalate": "", "notes": ""},
        {"id": 2, "text": "world", "conversation_id": "c2", "gold_intent": "", "should_escalate": "", "notes": ""},
    ]), encoding="utf-8")
    out = tmp_path / "double_subset.csv"
    initialize_double_subset(source, out, n=2)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "human_a_intent" in text
    assert "human_b_intent" in text
    assert "human_a_escalation" in text
    assert "human_b_escalation" in text


def test_validate_annotation_file_detects_duplicates_and_counts_labels(tmp_path):
    source = tmp_path / "golden.json"
    source.write_text(json.dumps([
        {
            "id": 1,
            "tweet_id": "t1",
            "text": "hello",
            "conversation_id": "c1",
            "gold_intent": "other",
            "should_escalate": "NO",
        },
        {
            "id": 2,
            "tweet_id": "t2",
            "text": "world",
            "conversation_id": "c2",
            "gold_intent": "",
            "should_escalate": "",
        },
    ]), encoding="utf-8")

    summary = validate_annotation_file(source)
    assert summary == {"rows": 2, "labelled": 1, "unlabelled": 1}

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps([
        {"id": 1, "tweet_id": "same", "text": "a", "conversation_id": "c1"},
        {"id": 2, "tweet_id": "same", "text": "b", "conversation_id": "c2"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate tweet_id"):
        validate_annotation_file(duplicate)
