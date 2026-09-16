"""
pipeline.py — End-to-end pipeline: classify -> retrieve -> generate -> decide.

Run interactively:
    python -m src.pipeline

Run on a single message programmatically:
    from src.pipeline import run_pipeline
    result = run_pipeline("My order hasn't arrived yet")
"""

import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_PROCESSED, SELECTED_BRAND, LLM_BACKEND,
    CLASSIFIER_CONFIDENCE_THRESHOLD, RETRIEVAL_SIMILARITY_THRESHOLD,
)


def load_classifier():
    """Load the best available classifier (embedding > tfidf > majority)."""
    emb_path   = DATA_PROCESSED / "embedding_classifier.pkl"
    tfidf_path = DATA_PROCESSED / "tfidf_lr_model.pkl"

    if emb_path.exists():
        from src.intents.classifier import EmbeddingClassifier
        clf = EmbeddingClassifier()
        clf.load(emb_path)
        return clf, "embedding_lr"

    if tfidf_path.exists():
        with open(tfidf_path, "rb") as f:
            clf = pickle.load(f)
        return clf, "tfidf_lr"

    raise FileNotFoundError(
        "No trained classifier found.\n"
        "Run: python scripts/train_baselines.py\n"
        "Then: python scripts/train_classifier.py"
    )


def load_retriever():
    """Load the FAISS retriever."""
    from src.retrieval.retriever import Retriever
    return Retriever()


def run_pipeline(
    customer_message: str,
    brand: str = SELECTED_BRAND,
    backend: str = LLM_BACKEND,
    top_k: int = 5,
) -> dict:
    """
    Run the full pipeline for a single customer message.

    Returns a dict with all intermediate and final outputs.
    """
    from src.data.cleaner import clean_text
    from src.generation.generator import generate_reply
    from src.decision.escalation import decide
    from src.retrieval.retriever import Retriever

    # Step 1: Clean the input
    clean_msg = clean_text(customer_message)
    if not clean_msg:
        return {
            "error": "Empty message after cleaning.",
            "decision": "ESCALATE",
            "reason": "Empty or unreadable message.",
        }

    # Step 2: Classify intent
    clf, clf_name = load_classifier()
    intent, confidence, proba = clf.predict_with_confidence(clean_msg)

    # Step 3: Retrieve historical evidence
    retriever = load_retriever()
    evidence  = retriever.retrieve(clean_msg, top_k=top_k)
    ev_score  = retriever.get_evidence_score(evidence)

    # Step 4: Generate reply
    gen_result = generate_reply(
        brand=brand,
        customer_message=clean_msg,
        intent=intent,
        confidence=confidence,
        historical_examples=evidence,
        backend=backend,
    )

    # Step 5: Decide
    decision = decide(
        intent=intent,
        confidence=confidence,
        evidence_score=ev_score,
        n_evidence=len(evidence),
    )

    return {
        "brand":              brand,
        "customer_message":   customer_message,
        "clean_message":      clean_msg,
        "intent":             intent,
        "confidence":         confidence,
        "intent_proba":       proba,
        "evidence":           evidence,
        "evidence_score":     ev_score,
        "reply":              gen_result["reply"],
        "grounded":           gen_result["grounded"],
        "backend":            gen_result["backend"],
        "decision":           decision["decision"],
        "decision_reason":    decision["reason"],
        "gate_triggered":     decision.get("gate_triggered", ""),
        "composite_risk":     decision.get("composite_risk", 0.0),
        "classifier_used":    clf_name,
    }


def print_result(result: dict):
    """Pretty-print the pipeline result to stdout."""
    sep = "=" * 60
    print(sep)
    print(f"Brand:      {result.get('brand', '')}")
    print(f"Intent:     {result.get('intent', '')}  (confidence: {result.get('confidence', 0):.2f})")
    print()

    evidence = result.get("evidence", [])
    if evidence:
        print(f"Historical evidence ({len(evidence)} examples):")
        for ex in evidence:
            print(f"  [{ex['rank']}] sim={ex['similarity']:.3f}")
            print(f"      Customer: {ex['customer_text'][:80]}...")
            print(f"      Support:  {ex['brand_text'][:80]}...")
    else:
        print("Historical evidence: None found.")
    print()

    print(f"Draft reply:")
    print(f"  {result.get('reply', '')}")
    print()

    decision = result.get("decision", "ESCALATE")
    reason   = result.get("decision_reason", "")
    gate     = result.get("gate_triggered", "")
    risk     = result.get("composite_risk", 0.0)
    print(f"Decision:   {decision}")
    if gate:
        print(f"Gate:       {gate}")
    print(f"Risk score: {risk:.3f}")
    print(f"Reason:     {reason}")
    print(sep)


def interactive_cli():
    """Interactive command-line interface."""
    print("=" * 60)
    print(f"Hiver AI Support Agent — Brand: {SELECTED_BRAND}")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        print()
        try:
            msg = input("Customer message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if msg.lower() in ("quit", "exit", "q"):
            break

        if not msg:
            continue

        try:
            result = run_pipeline(msg)
            print_result(result)
        except FileNotFoundError as e:
            print(f"Setup error: {e}")
            break
        except Exception as e:
            print(f"Pipeline error: {e}")


if __name__ == "__main__":
    interactive_cli()
