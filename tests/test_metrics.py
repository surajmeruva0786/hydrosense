from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import aggregate_fold_metrics, compute_metrics, macro_roc_auc, top_k_accuracy


def test_top_k_accuracy_perfect_top1():
    y_true = np.array([0, 1, 2])
    y_prob = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.0, 0.0, 1.0]])
    assert top_k_accuracy(y_true, y_prob, k=1) == 1.0


def test_top_k_accuracy_top2_catches_second_place():
    y_true = np.array([1])
    y_prob = np.array([[0.6, 0.4, 0.0]])  # true class is 2nd highest
    assert top_k_accuracy(y_true, y_prob, k=1) == 0.0
    assert top_k_accuracy(y_true, y_prob, k=2) == 1.0


def test_compute_metrics_perfect_predictions():
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    y_prob = np.eye(3)[y_true]

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["top2_accuracy"] == 1.0
    assert np.array(metrics["confusion_matrix"]).trace() == len(y_true)


def test_compute_metrics_handles_missing_class_gracefully():
    class_names = ["A", "B", "C"]
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    assert metrics["per_class"]["C"]["support"] == 0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_macro_roc_auc_returns_nan_when_class_absent():
    y_true = np.array([0, 0, 0])
    y_prob = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
    result = macro_roc_auc(y_true, y_prob, num_classes=3)
    assert np.isnan(result)


def test_aggregate_fold_metrics_mean_std():
    fold_metrics = [
        {
            "accuracy": 0.8,
            "macro_f1": 0.7,
            "weighted_f1": 0.75,
            "top2_accuracy": 0.9,
            "macro_roc_auc": 0.85,
        },
        {
            "accuracy": 0.9,
            "macro_f1": 0.8,
            "weighted_f1": 0.85,
            "top2_accuracy": 0.95,
            "macro_roc_auc": 0.9,
        },
    ]
    agg = aggregate_fold_metrics(fold_metrics)
    assert agg["accuracy"]["mean"] == pytest.approx(0.85)
    assert agg["macro_f1"]["mean"] == pytest.approx(0.75)
