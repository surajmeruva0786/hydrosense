"""Classification metrics shared by training-time validation and final evaluation.

All functions operate on numpy arrays so they are usable from both the
PyTorch training loop and the standalone evaluation script without a
framework dependency.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 2) -> float:
    """Fraction of samples where the true label is among the top-k predicted classes."""
    if y_prob.shape[1] < k:
        k = y_prob.shape[1]
    top_k_preds = np.argsort(-y_prob, axis=1)[:, :k]
    hits = (top_k_preds == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def macro_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    """One-vs-rest macro-averaged ROC-AUC. Returns NaN if a class is entirely absent."""
    y_true_onehot = np.eye(num_classes)[y_true]
    try:
        return float(roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        return float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute the full HydroSense evaluation metric suite for one fold / run.

    Returns a JSON-serialisable dict: overall accuracy, macro/weighted F1,
    per-class precision/recall/F1, top-2 accuracy, macro ROC-AUC, and the
    raw confusion matrix (counts).
    """
    num_classes = len(class_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": top_k_accuracy(y_true, y_prob, k=2),
        "macro_roc_auc": macro_roc_auc(y_true, y_prob, num_classes),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist(),
        "class_names": class_names,
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Mean +/- std across cross-validation folds for the scalar metrics.

    Mirrors the "mean ± standard deviation across folds" reporting convention
    used throughout the README results tables.
    """
    scalar_keys = ["accuracy", "macro_f1", "weighted_f1", "top2_accuracy", "macro_roc_auc"]
    agg = {}
    for key in scalar_keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        agg[key] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    return agg
