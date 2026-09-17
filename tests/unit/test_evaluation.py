"""Tests for Phase 4 binary fraud-model evaluation metrics."""

import numpy as np
import pytest

from fraud_intelligence.models.evaluation import (
    calculate_precision_recall_at_k,
    evaluate_binary_classifier,
)


def test_calculate_precision_recall_at_k_returns_expected_ranking_metrics() -> None:
    """The top-k ranking metrics must use the highest fraud probabilities."""
    y_true = np.array([0, 1, 0, 1])
    fraud_probabilities = np.array([0.10, 0.90, 0.20, 0.80])

    metrics = calculate_precision_recall_at_k(
        y_true=y_true,
        fraud_probabilities=fraud_probabilities,
        review_capacity=2,
    )

    assert metrics["review_capacity"] == 2
    assert metrics["captured_fraud_count_at_k"] == 2
    assert metrics["total_fraud_count"] == 2
    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0


def test_evaluate_binary_classifier_returns_all_phase4_metrics() -> None:
    """The full evaluator must return ranking, threshold, and latency metrics."""
    y_true = np.array([0, 1, 0, 1])
    fraud_probabilities = np.array([0.10, 0.90, 0.20, 0.80])

    metrics = evaluate_binary_classifier(
        y_true=y_true,
        fraud_probabilities=fraud_probabilities,
        review_capacity=2,
        reference_threshold=0.50,
        prediction_latency_ms_per_row=0.25,
    )

    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision_at_reference_threshold"] == 1.0
    assert metrics["recall_at_reference_threshold"] == 1.0
    assert metrics["f1_at_reference_threshold"] == 1.0
    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["true_positives"] == 2
    assert metrics["prediction_latency_ms_per_row"] == 0.25


def test_evaluate_binary_classifier_rejects_invalid_review_capacity() -> None:
    """Review capacity greater than evaluation rows must fail clearly."""
    with pytest.raises(ValueError, match="cannot exceed"):
        evaluate_binary_classifier(
            y_true=np.array([0, 1]),
            fraud_probabilities=np.array([0.10, 0.90]),
            review_capacity=3,
            reference_threshold=0.50,
            prediction_latency_ms_per_row=0.25,
        )
