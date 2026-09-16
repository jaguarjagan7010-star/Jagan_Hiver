"""
metrics.py — Automated evaluation metrics for the intent classifier.

Computes and saves:
- Accuracy
- Macro F1, Weighted F1
- Per-intent Precision, Recall, F1
- Confusion matrix
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import RESULTS_DIR, BOOTSTRAP_N, RANDOM_SEED


def compute_classifier_metrics(
    y_true: list,
    y_pred: list,
    model_name: str = "model",
) -> dict:
    """
    Compute full classification metrics with 95% bootstrap CI.
    """
    labels = sorted(set(y_true) | set(y_pred))

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    acc      = round(accuracy_score(y_true, y_pred), 4)
    macro_f1 = round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4)
    w_f1     = round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)

    # 95% bootstrap CI for accuracy and macro F1
    acc_ci, f1_ci = _bootstrap_ci(y_true, y_pred)

    metrics = {
        "model":           model_name,
        "n_samples":       len(y_true),
        "accuracy":        acc,
        "accuracy_ci_95":  acc_ci,
        "macro_f1":        macro_f1,
        "macro_f1_ci_95":  f1_ci,
        "weighted_f1":     w_f1,
        "macro_precision": round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "macro_recall":    round(recall_score(y_true, y_pred, average="macro",       zero_division=0), 4),
        "per_class":       {},
    }

    for label in labels:
        if label in report:
            metrics["per_class"][label] = {
                "precision": round(report[label]["precision"], 4),
                "recall":    round(report[label]["recall"],    4),
                "f1":        round(report[label]["f1-score"],  4),
                "support":   int(report[label]["support"]),
            }

    return metrics


def _bootstrap_ci(
    y_true: list,
    y_pred: list,
    n_iter: int = BOOTSTRAP_N,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Compute 95% bootstrap confidence intervals for accuracy and macro F1.
    Returns (acc_ci, f1_ci) where each is [lower, upper].
    """
    rng = np.random.default_rng(seed)
    labels = {label: index for index, label in enumerate(sorted(set(y_true) | set(y_pred)))}
    y_true = np.array([labels[label] for label in y_true], dtype=np.intp)
    y_pred = np.array([labels[label] for label in y_pred], dtype=np.intp)
    n = len(y_true)
    n_classes = len(labels)

    accs, f1s = [], []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sampled_true = y_true[idx]
        sampled_pred = y_pred[idx]
        accs.append(float(np.mean(sampled_true == sampled_pred)))

        true_counts = np.bincount(sampled_true, minlength=n_classes)
        pred_counts = np.bincount(sampled_pred, minlength=n_classes)
        pair_counts = np.bincount(
            sampled_true * n_classes + sampled_pred,
            minlength=n_classes * n_classes,
        ).reshape(n_classes, n_classes)
        denominators = true_counts + pred_counts
        class_f1 = np.divide(
            2 * np.diag(pair_counts),
            denominators,
            out=np.zeros(n_classes, dtype=float),
            where=denominators != 0,
        )
        f1s.append(float(np.mean(class_f1)))

    acc_ci = [round(float(np.percentile(accs, 2.5)), 4),
              round(float(np.percentile(accs, 97.5)), 4)]
    f1_ci  = [round(float(np.percentile(f1s, 2.5)), 4),
              round(float(np.percentile(f1s, 97.5)), 4)]
    return acc_ci, f1_ci


def compute_escalation_metrics(
    y_true_decisions: list,
    y_pred_decisions: list,
) -> dict:
    """
    Compute escalation precision and recall.
    Positive class = ESCALATE.
    """
    tp = sum(t == "ESCALATE" and p == "ESCALATE" for t, p in zip(y_true_decisions, y_pred_decisions))
    fp = sum(t == "AUTO_HANDLE" and p == "ESCALATE" for t, p in zip(y_true_decisions, y_pred_decisions))
    fn = sum(t == "ESCALATE" and p == "AUTO_HANDLE" for t, p in zip(y_true_decisions, y_pred_decisions))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "escalation_precision": round(precision, 4),
        "escalation_recall":    round(recall, 4),
        "escalation_f1":        round(f1, 4),
        "n_true_escalate":      sum(t == "ESCALATE" for t in y_true_decisions),
        "n_pred_escalate":      sum(p == "ESCALATE" for p in y_pred_decisions),
    }


def save_metrics(metrics: dict, model_name: str):
    """Save metrics dict to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def save_confusion_matrix(y_true, y_pred, model_name: str):
    """Save confusion matrix as CSV."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    path = RESULTS_DIR / f"{model_name}_confusion_matrix.csv"
    cm_df.to_csv(path, encoding="utf-8")
    return path


def save_comparison_table(results: list[dict]):
    """
    Save a side-by-side comparison of all models.
    results: list of metrics dicts from compute_classifier_metrics()
    """
    rows = []
    for m in results:
        rows.append({
            "model":           m["model"],
            "accuracy":        m["accuracy"],
            "macro_f1":        m["macro_f1"],
            "weighted_f1":     m["weighted_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall":    m["macro_recall"],
        })
    df = pd.DataFrame(rows)
    path = RESULTS_DIR / "comparison.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Save confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.config import FIGURES_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
