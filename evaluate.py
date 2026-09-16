"""
evaluate.py — Full evaluation harness for the Hiver Support Agent.

Required command:
    python evaluate.py --golden_set golden_200.json

Also supports:
    python evaluate.py                          (uses golden_set.csv if exists)
    python evaluate.py --golden_set golden_200.json --backend ollama

Evaluates:
  1. Classifier metrics on test set (accuracy, macro F1, 95% bootstrap CI)
  2. Escalation precision/recall on golden set (requires should_escalate column)
  3. LLM judge scores on 50 examples (temperature=0, 7 criteria)
  4. Human vs LLM agreement (Cohen's kappa) on 50 double-labelled examples
  5. Top-5 failure analysis
"""

import sys
import json
import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import (
    DATA_PROCESSED, DATA_GOLDEN, RESULTS_DIR, FIGURES_DIR,
    SELECTED_BRAND, LLM_BACKEND, LLM_JUDGE_SAMPLE, BOOTSTRAP_N,
)
from src.evaluation.metrics import (
    compute_classifier_metrics, save_metrics,
    save_confusion_matrix, plot_confusion_matrix,
    save_comparison_table, compute_escalation_metrics,
)
from src.intents.classifier import MajorityClassifier


def evaluate_classifier(log):
    """Re-evaluate all classifiers on the test set with bootstrap CI."""
    log.write("=== CLASSIFIER EVALUATION ===\n"); log.flush()

    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    if not train_path.exists() or not test_path.exists():
        log.write("train.csv or test.csv not found. Run train_baselines.py first.\n")
        return

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_test = test["clean_text"].tolist()
    y_test = test["intent"].tolist()
    X_train = train["clean_text"].tolist()
    y_train = train["intent"].tolist()

    all_metrics = []
    for name, path in [
        ("majority_baseline", None),
        ("tfidf_lr",          DATA_PROCESSED / "tfidf_lr_model.pkl"),
        ("embedding_lr",      DATA_PROCESSED / "embedding_classifier.pkl"),
    ]:
        try:
            if name == "majority_baseline":
                classifier = MajorityClassifier().fit(X_train, y_train)
            else:
                if not path.exists():
                    log.write(f"{name}: model not found at {path}\n")
                    continue
                if name == "embedding_lr":
                    try:
                        import torch  # noqa: F401
                    except (ImportError, OSError, RuntimeError) as exc:
                        log.write(
                            f"{name}: unavailable ({type(exc).__name__}: {exc})\n"
                        )
                        continue
                    from src.intents.classifier import EmbeddingClassifier
                    classifier = EmbeddingClassifier()
                    classifier.load(path)
                else:
                    with open(path, "rb") as f:
                        classifier = pickle.load(f)

            y_pred = classifier.predict(X_test)
            m = compute_classifier_metrics(y_test, y_pred, name)
            save_metrics(m, name)
            save_confusion_matrix(y_test, y_pred, name)
            plot_confusion_matrix(y_test, y_pred, name)
            all_metrics.append(m)
            ci = m["macro_f1_ci_95"]
            log.write(
                f"{name}: accuracy={m['accuracy']}  "
                f"macro_f1={m['macro_f1']}  "
                f"95%CI=[{ci[0]},{ci[1]}]\n"
            )
        except (ImportError, OSError, RuntimeError) as exc:
            log.write(f"{name}: unavailable ({type(exc).__name__}: {exc})\n")

    if all_metrics:
        save_comparison_table(all_metrics)
        log.write("Comparison table saved.\n")
    log.flush()


def evaluate_retrieval(log):
    """Analyse retrieval similarity distribution."""
    log.write("\n=== RETRIEVAL EVALUATION ===\n"); log.flush()

    index_dir = DATA_PROCESSED / "faiss_index"
    if not index_dir.exists():
        log.write("FAISS index not found. Run build_retrieval.py first.\n")
        return

    try:
        import sentence_transformers  # noqa: F401
    except (ImportError, OSError, RuntimeError) as exc:
        log.write(f"Retrieval evaluation unavailable ({type(exc).__name__}: {exc})\n")
        log.flush()
        return

    from src.retrieval.retriever import Retriever
    retriever = Retriever()

    test_path = DATA_PROCESSED / "test.csv"
    if not test_path.exists():
        log.write("test.csv not found.\n")
        return

    test = pd.read_csv(test_path)
    sample = test.sample(n=min(200, len(test)), random_state=42)

    similarities = []
    try:
        for _, row in sample.iterrows():
            results = retriever.retrieve(row["clean_text"], top_k=5)
            if results:
                similarities.extend([r["similarity"] for r in results])
    except (ImportError, OSError, RuntimeError) as exc:
        log.write(f"Retrieval evaluation unavailable ({type(exc).__name__}: {exc})\n")
        log.flush()
        return

    if similarities:
        sims = np.array(similarities)
        log.write(f"Retrieval similarity (n={len(sims)}):\n")
        log.write(f"  mean={sims.mean():.3f}  median={np.median(sims):.3f}\n")
        log.write(f"  min={sims.min():.3f}  max={sims.max():.3f}\n")
        log.write(f"  pct>0.5: {(sims>0.5).mean()*100:.1f}%  pct>0.3: {(sims>0.3).mean()*100:.1f}%\n")
        pd.DataFrame({"similarity": similarities}).to_csv(
            RESULTS_DIR / "retrieval_similarity_dist.csv", index=False)
    log.flush()


def evaluate_golden(golden_path: Path, log, backend: str):
    """
    Evaluate on the human-labelled golden set.
    Requires columns: text, gold_intent, should_escalate.
    Never fabricates labels — skips metrics if labels are missing.
    """
    log.write("\n=== GOLDEN SET EVALUATION ===\n"); log.flush()

    if not golden_path.exists():
        log.write(f"Golden set not found: {golden_path}\n")
        log.write("Create and label data/golden/golden_200.json first.\n")
        return

    # Load golden set (JSON or CSV)
    if str(golden_path).endswith(".json"):
        with open(golden_path, encoding="utf-8") as f:
            data = json.load(f)
        golden = pd.DataFrame(data)
    else:
        golden = pd.read_csv(golden_path, encoding="utf-8")

    log.write(f"Golden set: {len(golden)} examples\n")

    # ── Intent accuracy vs human labels ──────────────────────────────────────
    has_gold_intent = (
        "gold_intent" in golden.columns and
        golden["gold_intent"].notna() &
        (golden["gold_intent"].astype(str).str.strip() != "")
    )
    n_labelled = has_gold_intent.sum() if hasattr(has_gold_intent, "sum") else 0

    log.write(f"Human-labelled rows: {n_labelled}\n")

    if n_labelled > 0:
        labelled = golden[has_gold_intent].copy()
        if "auto_intent" in labelled.columns:
            from sklearn.metrics import accuracy_score, f1_score
            y_true = labelled["gold_intent"].astype(str).tolist()
            y_pred = labelled["auto_intent"].astype(str).tolist()
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
            log.write(f"Auto-label accuracy vs human: {acc:.3f}\n")
            log.write(f"Auto-label macro F1 vs human: {f1:.3f}\n")

    # ── Escalation precision/recall ───────────────────────────────────────────
    has_escalation = (
        "should_escalate" in golden.columns and
        golden["should_escalate"].notna() &
        (golden["should_escalate"].astype(str).str.strip() != "")
    )
    n_esc = has_escalation.sum() if hasattr(has_escalation, "sum") else 0

    if n_esc > 0:
        log.write(f"\nEscalation evaluation (n={n_esc}):\n")
        esc_df = golden[has_escalation].copy()

        # Run pipeline to get predicted decisions
        from src.pipeline import run_pipeline

        try:
            import sentence_transformers  # noqa: F401
        except (ImportError, OSError, RuntimeError) as exc:
            log.write(
                f"Escalation metrics not calculated: retrieval dependency unavailable "
                f"({type(exc).__name__}: {exc})\n"
            )
            return

        y_true_esc, y_pred_esc = [], []
        failed_rows = 0
        for _, row in esc_df.iterrows():
            text = str(row.get("text", row.get("clean_text", "")))
            true_esc = str(row["should_escalate"]).strip().upper()
            # Normalise to ESCALATE / AUTO_HANDLE
            if true_esc in ("YES", "TRUE", "1", "ESCALATE"):
                true_esc = "ESCALATE"
            else:
                true_esc = "AUTO_HANDLE"

            try:
                result = run_pipeline(text, brand=SELECTED_BRAND, backend="template")
                pred_esc = result.get("decision", "ESCALATE")
            except (ImportError, OSError, RuntimeError) as exc:
                failed_rows = len(esc_df)
                log.write(
                    f"  Escalation prediction unavailable for row {row.get('id', '')}; "
                    f"skipping the remaining {len(esc_df) - len(y_true_esc)} rows: "
                    f"{type(exc).__name__}: {exc}\n"
                )
                break

            y_true_esc.append(true_esc)
            y_pred_esc.append(pred_esc)

        if failed_rows:
            log.write(
                f"Escalation metrics not calculated: {failed_rows} prediction(s) failed.\n"
            )
        else:
            esc_metrics = compute_escalation_metrics(y_true_esc, y_pred_esc)
            log.write(f"  Escalation precision: {esc_metrics['escalation_precision']:.3f}\n")
            log.write(f"  Escalation recall:    {esc_metrics['escalation_recall']:.3f}\n")
            log.write(f"  Escalation F1:        {esc_metrics['escalation_f1']:.3f}\n")

            with open(RESULTS_DIR / "escalation_metrics.json", "w") as f:
                json.dump(esc_metrics, f, indent=2)
    else:
        log.write("No should_escalate labels found. Add them to the golden set.\n")

    # ── LLM Judge ─────────────────────────────────────────────────────────────
    if backend in ("ollama", "huggingface"):
        log.write(f"\nRunning LLM judge (temperature=0, backend={backend})...\n")
        _run_llm_judge(golden, log, backend)
    else:
        log.write("\nLLM judge skipped (backend=template). Use --backend ollama to enable.\n")

    log.flush()


def _run_llm_judge(golden: pd.DataFrame, log, backend: str):
    """Run LLM judge on up to LLM_JUDGE_SAMPLE examples."""
    from src.evaluation.llm_judge import judge_response, DIMENSIONS
    from src.pipeline import run_pipeline

    sample = golden.sample(n=min(LLM_JUDGE_SAMPLE, len(golden)), random_state=42)
    judge_results = []

    for i, (_, row) in enumerate(sample.iterrows()):
        text = str(row.get("text", row.get("clean_text", "")))
        print(f"  Judging {i+1}/{len(sample)}...", end="\r", flush=True)

        try:
            result = run_pipeline(text, brand=SELECTED_BRAND, backend=backend)
            scores = judge_response(
                customer_message=text,
                generated_reply=result.get("reply", ""),
                historical_examples=result.get("evidence", []),
                intent=result.get("intent", ""),
                decision=result.get("decision", ""),
                backend=backend,
            )
            row_result = {"id": row.get("id", i)}
            for dim in DIMENSIONS:
                row_result[f"judge_{dim}"] = scores.get(dim, {}).get("score", 3)
            row_result["judge_mean"] = scores.get("mean_score", 3.0)
            judge_results.append(row_result)
        except Exception as e:
            log.write(f"\n  Judge error on row {i}: {e}\n")

    print()
    if judge_results:
        judge_df = pd.DataFrame(judge_results)
        judge_df.to_csv(RESULTS_DIR / "llm_judge_scores.csv", index=False)
        mean_scores = {
            dim: round(judge_df[f"judge_{dim}"].mean(), 3)
            for dim in DIMENSIONS if f"judge_{dim}" in judge_df.columns
        }
        log.write(f"LLM judge mean scores (n={len(judge_df)}):\n")
        for dim, score in mean_scores.items():
            log.write(f"  {dim}: {score}\n")
        log.write(f"  overall_mean: {judge_df['judge_mean'].mean():.3f}\n")

        # ── Human vs LLM agreement ─────────────────────────────────────────
        human_scores_path = DATA_GOLDEN / "human_scores_50.csv"
        if human_scores_path.exists():
            log.write("\nComputing human vs LLM agreement (Cohen's kappa)...\n")
            from src.evaluation.human_agreement import compute_agreement
            human_df = pd.read_csv(human_scores_path)
            agreement = compute_agreement(human_df, judge_df)
            log.write(f"  Overall kappa: {agreement.get('overall', {}).get('kappa', 'N/A')}\n")
            with open(RESULTS_DIR / "human_llm_agreement.json", "w") as f:
                json.dump(agreement, f, indent=2)
        else:
            log.write(f"\nHuman scores not found at {human_scores_path}.\n")
            log.write("Fill in data/golden/human_scores_50.csv to compute kappa.\n")


def failure_analysis(log):
    """Top-5 failure analysis on test set."""
    log.write("\n=== TOP-5 FAILURE ANALYSIS ===\n"); log.flush()

    test_path = DATA_PROCESSED / "test.csv"
    model_path = DATA_PROCESSED / "tfidf_lr_model.pkl"
    if not test_path.exists() or not model_path.exists():
        log.write("test.csv or tfidf_lr_model.pkl not found.\n")
        return

    test = pd.read_csv(test_path)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    X_test = test["clean_text"].tolist()
    y_true = test["intent"].tolist()
    y_pred = clf.predict(X_test)

    # Find misclassified examples
    errors = [
        {"text": x, "true": t, "pred": p}
        for x, t, p in zip(X_test, y_true, y_pred) if t != p
    ]

    if not errors:
        log.write("No errors found on test set.\n")
        return

    # Count error pairs
    from collections import Counter
    error_pairs = Counter((e["true"], e["pred"]) for e in errors)
    top5 = error_pairs.most_common(5)

    log.write(f"Total errors: {len(errors)} / {len(y_true)} ({len(errors)/len(y_true)*100:.1f}%)\n\n")
    log.write("Top-5 confusion pairs (true → predicted):\n")
    for (true, pred), count in top5:
        log.write(f"  {true} → {pred}: {count} errors\n")
        # Show 2 examples
        examples = [e for e in errors if e["true"] == true and e["pred"] == pred][:2]
        for ex in examples:
            log.write(f"    e.g.: \"{ex['text'][:80]}\"\n")

    # Save failure analysis
    pd.DataFrame(errors[:100]).to_csv(RESULTS_DIR / "failure_analysis.csv", index=False)
    log.write(f"\nFull failure list saved to failure_analysis.csv\n")
    log.flush()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hiver Support Agent")
    parser.add_argument("--golden_set", default=None,
                        help="Path to golden set file (e.g. golden_200.json)")
    parser.add_argument("--backend", default="template",
                        choices=["ollama", "huggingface", "template"],
                        help="LLM backend for judge (default: template)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve golden set path
    if args.golden_set:
        golden_path = Path(args.golden_set)
        if not golden_path.is_absolute():
            # Try relative to data/golden/ first, then cwd
            candidate = DATA_GOLDEN / args.golden_set
            if candidate.exists():
                golden_path = candidate
    else:
        golden_path = DATA_GOLDEN / "golden_set.csv"
        if not golden_path.exists():
            golden_path = DATA_GOLDEN / "golden_200.json"

    with open(RESULTS_DIR / "evaluation_log.txt", "w", encoding="utf-8") as log:
        log.write(f"Golden set: {golden_path}\n")
        log.write(f"Backend: {args.backend}\n\n")

        evaluate_classifier(log)
        evaluate_retrieval(log)
        evaluate_golden(golden_path, log, backend=args.backend)
        failure_analysis(log)

        log.write("\nEvaluation complete.\n")

    print(f"\nDone. Check {RESULTS_DIR / 'evaluation_log.txt'}")


if __name__ == "__main__":
    main()
