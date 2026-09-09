"""Binary classification metrics and validation-only threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validated_binary_arrays(
    y_true: ArrayLike, probabilities: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and probabilities must be same-length one-dimensional arrays")
    if len(labels) == 0:
        raise ValueError("labels and probabilities cannot be empty")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary values 0 or 1")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return labels.astype(int), scores


def binary_classification_metrics(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the metrics reported for the binary classifier."""

    labels, scores = _validated_binary_arrays(y_true, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity_denominator = true_negative + false_positive
    specificity = (
        true_negative / specificity_denominator if specificity_denominator else float("nan")
    )
    auc = float("nan")
    if len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, scores))
    balanced_accuracy = float("nan")
    if len(np.unique(labels)) == 2:
        balanced_accuracy = float((specificity + recall_score(labels, predictions)) / 2)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": auc,
        "specificity": float(specificity),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix,
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "true_negative": int(true_negative),
        "false_negative": int(false_negative),
    }


def select_threshold(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    objective: str = "balanced_accuracy",
) -> tuple[float, dict[str, Any]]:
    """Choose a decision threshold using validation labels only."""

    labels, scores = _validated_binary_arrays(y_true, probabilities)
    if len(np.unique(labels)) != 2:
        raise ValueError("threshold selection requires both classes in validation data")
    if objective not in {"balanced_accuracy", "min_class_recall"}:
        raise ValueError("objective must be 'balanced_accuracy' or 'min_class_recall'")
    unique_scores = np.unique(scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], unique_scores, midpoints]))

    ranked: list[tuple[float, float, float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = binary_classification_metrics(labels, scores, float(threshold))
        if objective == "balanced_accuracy":
            value = metrics["balanced_accuracy"]
        else:
            value = min(metrics["specificity"], metrics["recall"])
        ranked.append((-float(value), abs(float(threshold) - 0.5), float(threshold), metrics))
    _, _, threshold, best_metrics = min(ranked, key=lambda item: item[:3])
    return threshold, best_metrics
