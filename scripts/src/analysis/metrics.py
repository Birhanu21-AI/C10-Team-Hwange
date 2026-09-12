"""Shared metric computation for baselines (Phase 7) and probes (Phase 9)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None
) -> dict[str, Any]:
    """Standard binary classification metrics.

    ROC-AUC and PR-AUC are omitted rather than faked when only one class is
    present in y_true, which happens in small per-category slices.
    """
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    # Impact Assessment Card 8.1 names false-positive and false-negative rates as
    # first-class measures: a false positive here means suppressing legitimate speech.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["false_positive_rate"] = round(float(fp / (fp + tn)), 4) if (fp + tn) else None
    metrics["false_negative_rate"] = round(float(fn / (fn + tp)), 4) if (fn + tp) else None
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["auc_note"] = "single class present in this slice"
    return metrics


def aggregate_seeds(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean +/- standard deviation across seeds (Phase 14 reporting convention)."""
    numeric_keys = [
        key
        for key in runs[0]
        if isinstance(runs[0][key], (int, float)) and key not in {"n"}
    ]
    summary: dict[str, Any] = {"n_seeds": len(runs), "n": runs[0].get("n")}
    for key in numeric_keys:
        values = [run[key] for run in runs if run.get(key) is not None]
        if values:
            summary[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
            }
    return summary
