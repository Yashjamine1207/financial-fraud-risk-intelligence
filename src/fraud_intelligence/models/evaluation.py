"""Reusable model evaluation metrics for Phase 4.

This module evaluates fraud-risk probabilities on the chronological validation
split. It supports threshold-based reference metrics and investigation-capacity
metrics without using the locked final test split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_evaluation_inputs(
    y_true: np.ndarray,
    fraud_probabilities: np.ndarray,
    review_capacity: int,
    reference_threshold: float,
) -> None:
    """Validate labels, probabilities, and evaluation-policy inputs."""
    if y_true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array.")

    if fraud_probabilities.ndim != 1:
        raise ValueError("fraud_probabilities must be a one-dimensional array.")

    if len(y_true) != len(fraud_probabilities):
        raise ValueError("y_true and fraud_probabilities must contain the same number of rows.")

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    unique_labels = set(np.unique(y_true))
    if not unique_labels.issubset({0, 1}):
        raise ValueError("y_true must contain only binary fraud labels: 0 and 1.")

    if len(unique_labels) < 2:
        raise ValueError("Evaluation data must contain both fraud and non-fraud labels.")

    if not np.isfinite(fraud_probabilities).all():
        raise ValueError("fraud_probabilities contains missing or non-finite values.")

    if ((fraud_probabilities < 0.0) | (fraud_probabilities > 1.0)).any():
        raise ValueError("fraud_probabilities must be between 0.0 and 1.0.")

    if review_capacity <= 0:
        raise ValueError("review_capacity must be greater than zero.")

    if review_capacity > len(y_true):
        raise ValueError("review_capacity cannot exceed the number of evaluation rows.")

    if not 0.0 <= reference_threshold <= 1.0:
        raise ValueError("reference_threshold must be between 0.0 and 1.0.")


def calculate_precision_recall_at_k(
    y_true: np.ndarray,
    fraud_probabilities: np.ndarray,
    review_capacity: int,
) -> dict[str, float | int]:
    """Calculate ranking quality at the available manual-review capacity.

    The top-k transactions are selected by descending fraud probability.
    When scores tie, NumPy's stable sorting preserves their original row order.

    Parameters
    ----------
    y_true:
        Binary fraud labels for the evaluation period.
    fraud_probabilities:
        Predicted fraud probabilities for the evaluation period.
    review_capacity:
        Maximum number of transactions analysts can review.

    Returns
    -------
    dict[str, float | int]
        Precision@k, Recall@k, captured fraud count, total fraud count, and k.
    """
    sorted_indices = np.argsort(-fraud_probabilities, kind="stable")
    selected_indices = sorted_indices[:review_capacity]
    selected_labels = y_true[selected_indices]

    captured_fraud_count = int(selected_labels.sum())
    total_fraud_count = int(y_true.sum())

    return {
        "review_capacity": review_capacity,
        "captured_fraud_count_at_k": captured_fraud_count,
        "total_fraud_count": total_fraud_count,
        "precision_at_k": float(selected_labels.mean()),
        "recall_at_k": float(captured_fraud_count / total_fraud_count),
    }


def evaluate_binary_classifier(
    y_true: Any,
    fraud_probabilities: Any,
    review_capacity: int,
    reference_threshold: float,
    prediction_latency_ms_per_row: float,
) -> dict[str, float | int]:
    """Calculate Phase 4 validation metrics for a binary fraud classifier.

    This function does not tune a threshold. The supplied reference threshold
    is reported only as a diagnostic. Model selection should prioritise ranking
    metrics at the documented review capacity.

    Parameters
    ----------
    y_true:
        Binary fraud labels from the chronological validation period.
    fraud_probabilities:
        Fraud probabilities predicted for the validation period.
    review_capacity:
        Daily number of transactions available for manual investigation.
    reference_threshold:
        Fixed diagnostic threshold, not a final operating threshold.
    prediction_latency_ms_per_row:
        Average prediction latency per validation row in milliseconds.

    Returns
    -------
    dict[str, float | int]
        JSON-serialisable validation metrics for reporting and MLflow logging.
    """
    y_true_array = np.asarray(y_true, dtype=np.int8)
    fraud_probability_array = np.asarray(fraud_probabilities, dtype=float)

    _validate_evaluation_inputs(
        y_true=y_true_array,
        fraud_probabilities=fraud_probability_array,
        review_capacity=review_capacity,
        reference_threshold=reference_threshold,
    )

    reference_predictions = (fraud_probability_array >= reference_threshold).astype(np.int8)

    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        y_true_array,
        reference_predictions,
        labels=[0, 1],
    ).ravel()

    ranking_metrics = calculate_precision_recall_at_k(
        y_true=y_true_array,
        fraud_probabilities=fraud_probability_array,
        review_capacity=review_capacity,
    )

    return {
        "pr_auc": float(average_precision_score(y_true_array, fraud_probability_array)),
        "roc_auc": float(roc_auc_score(y_true_array, fraud_probability_array)),
        "precision_at_reference_threshold": float(
            precision_score(
                y_true_array,
                reference_predictions,
                zero_division=0,
            )
        ),
        "recall_at_reference_threshold": float(
            recall_score(
                y_true_array,
                reference_predictions,
                zero_division=0,
            )
        ),
        "f1_at_reference_threshold": float(
            f1_score(
                y_true_array,
                reference_predictions,
                zero_division=0,
            )
        ),
        "reference_threshold": float(reference_threshold),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "prediction_latency_ms_per_row": float(prediction_latency_ms_per_row),
        **ranking_metrics,
    }
