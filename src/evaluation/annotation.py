"""Utilities for human labelling of the golden AppleSupport set.

This module keeps the workflow intentionally small and local: it validates
intents against the canonical AppleSupport schema, supports resume-safe manual
annotation, and creates a double-label subset without fabricating labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_FIELDS = ("id", "text", "conversation_id")


def load_intent_schema(schema_path: str | Path | None = None) -> dict:
    """Load the canonical AppleSupport intent schema from disk."""
    if schema_path is None:
        schema_path = Path(__file__).resolve().parent.parent.parent / "data" / "golden" / "intent_schema.json"

    with Path(schema_path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError("Intent schema must be a JSON object.")

    return data


def get_valid_intents(schema_path: str | Path | None = None) -> list[str]:
    schema = load_intent_schema(schema_path)
    return [key for key in schema.keys() if not key.startswith("_")]


def _canonical_escalation(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    mapping = {
        "YES": "YES",
        "NO": "NO",
        "Y": "YES",
        "N": "NO",
        "AUTO_HANDLE": "AUTO_HANDLE",
        "ESCALATE": "ESCALATE",
        "HANDLE": "AUTO_HANDLE",
        "REVIEW": "ESCALATE",
    }
    return mapping.get(text, text)


def validate_annotation(record: dict, schema_path: str | Path | None = None) -> dict:
    """Validate a single annotation row and normalise common field aliases.

    Accepts either `gold_escalation` or `should_escalate` in the record and
    allows blank fields for partially completed work.
    """
    if not isinstance(record, dict):
        raise ValueError("Annotation record must be a dictionary.")

    missing_fields = [
        field for field in REQUIRED_FIELDS
        if field not in record or str(record[field]).strip() == ""
    ]
    if missing_fields:
        raise ValueError(
            f"Annotation record is missing required fields: {', '.join(missing_fields)}."
        )

    valid_intents = set(get_valid_intents(schema_path))
    allowed_escalations = {"YES", "NO", "AUTO_HANDLE", "ESCALATE"}

    record = dict(record)

    # Normalise common alias names used throughout the repo.
    if "gold_escalation" not in record and "should_escalate" in record:
        record["gold_escalation"] = record["should_escalate"]
    if "gold_escalation" not in record and "gold_escalation_label" in record:
        record["gold_escalation"] = record["gold_escalation_label"]
    if "gold_intent" not in record and "intent" in record:
        record["gold_intent"] = record["intent"]

    gold_intent = str(record.get("gold_intent", "")).strip()
    if gold_intent and gold_intent not in valid_intents:
        raise ValueError(f"gold_intent '{gold_intent}' is not valid for this schema.")

    escalation = record.get("gold_escalation", record.get("should_escalate", ""))
    escalation = _canonical_escalation(escalation)
    if escalation and escalation not in allowed_escalations:
        raise ValueError(f"Escalation '{escalation}' must be YES, NO, AUTO_HANDLE, or ESCALATE.")

    record["gold_intent"] = gold_intent
    record["gold_escalation"] = escalation
    if "should_escalate" in record:
        record["should_escalate"] = escalation
    elif "gold_escalation_label" in record:
        record["should_escalate"] = escalation

    return record


def validate_annotation_file(
    source_path: str | Path,
    schema_path: str | Path | None = None,
    require_complete: bool = False,
) -> dict:
    """Validate a JSON or CSV annotation file without changing it.

    Empty label fields are allowed for an annotation queue unless
    ``require_complete`` is true. Duplicate row IDs and source tweet IDs are
    always rejected because they make annotation counts ambiguous.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)
    elif path.suffix.lower() == ".csv":
        records = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    else:
        raise ValueError("Annotation file must be JSON or CSV.")

    if not isinstance(records, list):
        raise ValueError("Annotation file must contain a list of records.")

    errors = []
    seen_ids = set()
    seen_tweet_ids = set()
    labelled = 0
    for index, record in enumerate(records, start=1):
        try:
            normalized = validate_annotation(record, schema_path=schema_path)
            row_id = str(normalized["id"])
            if row_id in seen_ids:
                raise ValueError(f"duplicate id '{row_id}'")
            seen_ids.add(row_id)

            tweet_id = str(normalized.get("tweet_id", "")).strip()
            if tweet_id:
                if tweet_id in seen_tweet_ids:
                    raise ValueError(f"duplicate tweet_id '{tweet_id}'")
                seen_tweet_ids.add(tweet_id)

            has_intent = bool(normalized["gold_intent"])
            has_escalation = bool(normalized["gold_escalation"])
            if require_complete and not (has_intent and has_escalation):
                raise ValueError("gold_intent and escalation label are required.")
            if has_intent and has_escalation:
                labelled += 1
        except (TypeError, ValueError) as exc:
            errors.append(f"row {index}: {exc}")

    if errors:
        raise ValueError("Annotation validation failed:\n" + "\n".join(errors))

    return {"rows": len(records), "labelled": labelled, "unlabelled": len(records) - labelled}


def initialize_double_subset(
    source_path: str | Path,
    output_path: str | Path,
    n: int = 50,
) -> pd.DataFrame:
    """Create a double-label subset that preserves Human A/B annotations separately."""
    source = Path(source_path)
    records = json.loads(source.read_text(encoding="utf-8")) if source.exists() else []

    subset = []
    for row in records[:n]:
        subset.append({
            "id": row.get("id", ""),
            "text": row.get("text", ""),
            "auto_intent": row.get("auto_intent", ""),
            "human_a_intent": "",
            "human_b_intent": "",
            "human_a_escalation": "",
            "human_b_escalation": "",
            "human_a_notes": "",
            "human_b_notes": "",
            "notes": row.get("notes", ""),
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(subset)
    df.to_csv(out, index=False)
    return df


def label_row(record: dict, schema_path: str | Path | None = None) -> dict:
    """A minimal validation-and-normalisation helper for a single user annotation."""
    return validate_annotation(record, schema_path=schema_path)


def _prompt_choice(prompt: str, choices: Iterable[str]) -> str:
    choices = list(choices)
    while True:
        value = input(f"{prompt} [{'/'.join(choices)}]: ").strip()
        if not value:
            print("Value is required.")
            continue
        upper = value.upper()
        if upper in {choice.upper() for choice in choices}:
            return next(choice for choice in choices if choice.upper() == upper)
        print(f"Invalid choice. Please choose from: {', '.join(choices)}")


def annotate_file(source_path: str | Path, schema_path: str | Path | None = None) -> list[dict]:
    """Interactive local annotation loop for the golden JSON file.

    The tool intentionally does not auto-fill labels. It validates each choice as
    it is entered and saves progress as it goes.
    """
    path = Path(source_path)
    schema = load_intent_schema(schema_path)
    valid_intents = [key for key in schema.keys() if not key.startswith("_")]

    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".csv":
        rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    else:
        raise ValueError("Annotation source must be JSON or CSV.")
    for idx, row in enumerate(rows, start=1):
        row = validate_annotation(row, schema_path=schema_path)
        rows[idx - 1] = row
        if str(row.get("gold_intent", "")).strip() and str(row.get("should_escalate", "")).strip():
            print(f"[{idx}/{len(rows)}] Skipping already labelled item {row.get('id')}")
            continue

        print(f"\n[{idx}/{len(rows)}] ID {row.get('id')}")
        print(row.get("text", ""))
        intent = _prompt_choice("Intent", valid_intents)
        escalation = _prompt_choice("Escalate?", ["YES", "NO"])

        row["gold_intent"] = intent
        row["should_escalate"] = escalation
        row["gold_escalation"] = escalation
        row["notes"] = row.get("notes", "")

        validate_annotation(row, schema_path=schema_path)

        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved annotation for item {row.get('id')}.")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Human annotation helper for the AppleSupport golden set.")
    parser.add_argument("--source", default="data/golden/golden_200.json", help="Path to the JSON golden file to annotate.")
    parser.add_argument("--schema", default="data/golden/intent_schema.json", help="Path to the JSON intent schema.")
    parser.add_argument("--double-subset", nargs="?", default=None, help="Optional path to create a double-label CSV subset.")
    parser.add_argument("--n", type=int, default=50, help="Number of rows to include in the double-label subset.")
    parser.add_argument("--validate", action="store_true", help="Validate the source file without editing it.")
    parser.add_argument("--require-complete", action="store_true", help="Require every row to have intent and escalation labels.")
    args = parser.parse_args()

    source_path = Path(args.source)
    schema_path = Path(args.schema)
    if not source_path.exists():
        raise FileNotFoundError(f"Golden file not found: {source_path}")

    if args.validate:
        summary = validate_annotation_file(
            source_path,
            schema_path=schema_path,
            require_complete=args.require_complete,
        )
        print(
            f"Validated {summary['rows']} rows "
            f"({summary['labelled']} labelled, {summary['unlabelled']} unlabelled)."
        )
        return

    if args.double_subset:
        initialize_double_subset(source_path, args.double_subset, n=args.n)
        print(f"Created double-label subset: {args.double_subset}")

    annotate_file(source_path, schema_path=schema_path)


if __name__ == "__main__":
    main()