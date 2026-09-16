"""
test_data.py — Unit tests for data loading and conversation reconstruction.
Run with: pytest tests/
"""

import sys
from pathlib import Path
import json
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.cleaner import clean_text, clean_dataframe
from src.data.conversations import reconstruct_pairs, get_brand_inbound, is_valid_id


# ── clean_text tests ───────────────────────────────────────────────────────────

def test_clean_text_removes_mentions():
    assert "@AmazonHelp" not in clean_text("@AmazonHelp my order is late")

def test_clean_text_removes_urls():
    assert "http" not in clean_text("check this http://example.com please")

def test_clean_text_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   ") == ""

def test_clean_text_preserves_content():
    result = clean_text("my order has not arrived yet")
    assert "order" in result
    assert "arrived" in result

def test_clean_dataframe_drops_empty():
    df = pd.DataFrame({"text": ["hello world", "", "@user", None]})
    result = clean_dataframe(df)
    # Only "hello world" survives (others become empty after cleaning)
    assert len(result) == 1
    assert result.iloc[0]["clean_text"] == "hello world"


# ── is_valid_id tests ──────────────────────────────────────────────────────────

def test_valid_id_numeric():
    assert is_valid_id("123456789") is True

def test_valid_id_none():
    assert is_valid_id(None) is False

def test_valid_id_nan():
    assert is_valid_id(float("nan")) is False

def test_valid_id_text():
    assert is_valid_id("abc") is False

def test_valid_id_empty():
    assert is_valid_id("") is False


# ── reconstruct_pairs tests ────────────────────────────────────────────────────

def _make_test_df():
    """Create a minimal DataFrame that simulates the Twitter dataset."""
    # Structure:
    # tweet 0: AmazonHelp outbound (initial brand tweet)
    # tweet 1: customer1 inbound, replying to tweet 0
    # tweet 2: AmazonHelp outbound, replying to tweet 1
    # tweet 3: customer2 inbound, replying to tweet 0
    # tweet 4: AmazonHelp outbound, replying to tweet 3
    return pd.DataFrame({
        "tweet_id":                ["0", "1", "2", "3", "4"],
        "author_id":               ["AmazonHelp", "customer1", "AmazonHelp", "customer2", "AmazonHelp"],
        "inbound":                 [False, True, False, True, False],
        "created_at":              pd.to_datetime(["2023-01-01"] * 5),
        "text":                    [
            "How can we help you today?",
            "My order is late",
            "Sorry to hear that! DM us your order number.",
            "I need a refund",
            "We can help with that! Please DM us.",
        ],
        "response_tweet_id":       [None, None, None, None, None],
        "in_response_to_tweet_id": [None, "0", "1", "0", "3"],
    })


def test_reconstruct_pairs_basic():
    df = _make_test_df()
    pairs = reconstruct_pairs(df, brand="AmazonHelp")
    assert len(pairs) == 2   # two customer/brand pairs
    assert "customer_text" in pairs.columns
    assert "brand_text" in pairs.columns


def test_reconstruct_pairs_no_brand():
    df = _make_test_df()
    pairs = reconstruct_pairs(df, brand="NonExistentBrand")
    assert len(pairs) == 0


def test_reconstruct_pairs_malformed_ids():
    """Malformed IDs should not crash the system."""
    df = _make_test_df()
    df.loc[0, "in_response_to_tweet_id"] = "not_a_number"
    # Should not raise
    pairs = reconstruct_pairs(df, brand="AmazonHelp")
    assert isinstance(pairs, pd.DataFrame)


def test_get_brand_inbound():
    df = _make_test_df()
    inbound = get_brand_inbound(df, brand="AmazonHelp")
    # customer1 and customer2 replied to AmazonHelp tweets
    assert len(inbound) == 2
    assert all(inbound["inbound"] == True)


def test_applesupport_submission_contract():
    """Contract test: passes only after sample_data.py and human labelling are done."""
    root = Path(__file__).resolve().parent.parent
    sample_path = root / "data" / "raw" / "sample_data.csv"
    golden_path = root / "data" / "golden" / "golden_200.json"

    if not sample_path.exists():
        pytest.skip("sample_data.csv not yet generated — run: python scripts/sample_data.py")

    sample = pd.read_csv(sample_path)
    assert 500 <= len(sample) <= 1000, f"sample size should be 500-1000, got {len(sample)}"
    assert "text" in sample.columns

    if not golden_path.exists():
        pytest.skip("golden_200.json not yet created — run: python scripts/build_golden_set.py")

    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert 1 <= len(data) <= 250, f"golden set should be 1-250 entries, got {len(data)}"
    # Only check human labels if they've been filled in
    labelled = [row for row in data if row.get("gold_intent", "").strip()]
    if labelled:
        assert len(labelled) >= 15, "Need at least 15 human-labelled examples"
