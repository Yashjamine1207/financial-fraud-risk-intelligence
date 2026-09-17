"""Unit tests for Phase 5 fraud-decision policy and threshold optimisation."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_intelligence.decisioning.cost_model import CostMatrix
from fraud_intelligence.decisioning.threshold_policy import (
    DecisionPolicyConfig,
    apply_decision_policy,
    evaluate_policy_at_capacity,
    select_top_k_review_indices,
    validate_binary_labels,
)


@pytest.fixture
def cost_matrix() -> CostMatrix:
    """Return the documented Phase 5 illustrative cost assumptions."""
    return CostMatrix(
        false_negative_cost=500.0,
        manual_review_cost=5.0,
        false_positive_escalation_cost=10.0,
        fraud_prevention_value=500.0,
    )


@pytest.fixture
def policy_config() -> DecisionPolicyConfig:
    """Return a minimal valid policy configuration."""
    return DecisionPolicyConfig(
        currency="GBP",
        daily_review_capacity=1000,
        min_review_probability=0.05,
        max_review_probability=0.95,
        action_labels=("approve", "review", "block"),
    )


def test_decision_policy_config_accepts_valid_values(
    policy_config: DecisionPolicyConfig,
) -> None:
    """A well-formed policy configuration should be accepted."""
    assert policy_config.currency == "GBP"
    assert policy_config.daily_review_capacity == 1000
    assert policy_config.min_review_probability == 0.05
    assert policy_config.max_review_probability == 0.95
    assert policy_config.action_labels == ("approve", "review", "block")


@pytest.mark.parametrize(
    "kwargs,error_substring",
    [
        (
            {
                "currency": "GBP",
                "daily_review_capacity": 0,
                "min_review_probability": 0.05,
                "max_review_probability": 0.95,
                "action_labels": ("approve", "review", "block"),
            },
            "at least 1",
        ),
        (
            {
                "currency": "GBP",
                "daily_review_capacity": 1000,
                "min_review_probability": 0.95,
                "max_review_probability": 0.05,
                "action_labels": ("approve", "review", "block"),
            },
            "strictly less than",
        ),
        (
            {
                "currency": "GBP",
                "daily_review_capacity": 1000,
                "min_review_probability": -0.01,
                "max_review_probability": 0.95,
                "action_labels": ("approve", "review", "block"),
            },
            "within",
        ),
        (
            {
                "currency": "GBP",
                "daily_review_capacity": 1000,
                "min_review_probability": 0.05,
                "max_review_probability": 0.95,
                "action_labels": ("approve", "review"),
            },
            "exactly three",
        ),
    ],
)
def test_decision_policy_config_rejects_invalid_values(
    kwargs: dict,
    error_substring: str,
) -> None:
    """Invalid policy configurations must fail with a clear error."""
    with pytest.raises(ValueError, match=error_substring):
        DecisionPolicyConfig(**kwargs)


def test_validate_binary_labels() -> None:
    """Binary labels must be one-dimensional and contain only 0 and 1."""
    valid = np.array([0, 1, 0, 1], dtype=np.int64)
    result = validate_binary_labels(valid)
    np.testing.assert_array_equal(result, valid)

    with pytest.raises(ValueError):
        validate_binary_labels(np.array([0, 2], dtype=np.int64))

    with pytest.raises(ValueError):
        validate_binary_labels(np.array([], dtype=np.int64))

    with pytest.raises(ValueError):
        validate_binary_labels(np.array([[0, 1]], dtype=np.int64))


def test_select_top_k_review_indices_respects_capacity(
    cost_matrix: CostMatrix,
) -> None:
    """The number of selected reviews must not exceed capacity."""
    probabilities = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
    labels = np.array([0, 0, 1, 1, 1], dtype=np.int64)

    indices = select_top_k_review_indices(
        fraud_probabilities=probabilities,
        labels=labels,
        review_capacity=2,
        min_probability=0.0,
        max_probability=1.0,
    )

    assert indices.size == 2
    assert set(indices) == {3, 4}  # 0.70 and 0.90


def test_select_top_k_review_indices_respects_probability_bounds(
    cost_matrix: CostMatrix,
) -> None:
    """Transactions outside probability bounds must not be selected."""
    probabilities = np.array([0.01, 0.20, 0.60, 0.98])

    indices = select_top_k_review_indices(
        fraud_probabilities=probabilities,
        labels=None,
        review_capacity=10,
        min_probability=0.10,
        max_probability=0.95,
    )

    assert indices.size == 2
    assert set(indices) == {1, 2}  # 0.20 and 0.60


def test_select_top_k_review_indices_ranks_by_priority(
    cost_matrix: CostMatrix,
) -> None:
    """Higher probabilities should be selected before lower ones."""
    probabilities = np.array([0.15, 0.80, 0.40, 0.60, 0.25])

    indices = select_top_k_review_indices(
        fraud_probabilities=probabilities,
        labels=None,
        review_capacity=3,
        min_probability=0.0,
        max_probability=1.0,
    )

    assert indices.size == 3
    assert set(indices) == {1, 3, 2}  # 0.80, 0.60, 0.40


def test_apply_decision_policy_assigns_actions_correctly(
    cost_matrix: CostMatrix,
) -> None:
    """Reviewed transactions should receive the review action code."""
    probabilities = np.array([0.10, 0.50, 0.70, 0.90])
    labels = np.array([0, 1, 1, 1], dtype=np.int64)
    review_indices = np.array([1, 2], dtype=np.int64)

    actions, confusion, _costs = apply_decision_policy(
        fraud_probabilities=probabilities,
        labels=labels,
        review_indices=review_indices,
        cost_matrix=cost_matrix,
        action_labels=("approve", "review", "block"),
    )

    assert actions.shape == probabilities.shape
    assert set(actions) == {0, 1}  # approve and review
    assert confusion["true_positives"] == 2
    assert confusion["false_negatives"] == 1
    assert confusion["false_positives"] == 0
    assert confusion["true_negatives"] == 1


def test_evaluate_policy_at_capacity(
    cost_matrix: CostMatrix,
    policy_config: DecisionPolicyConfig,
) -> None:
    """Policy evaluation should return expected fraud capture and costs."""
    probabilities = np.array(
        [0.05, 0.20, 0.40, 0.60, 0.80, 0.95],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)

    evaluation = evaluate_policy_at_capacity(
        fraud_probabilities=probabilities,
        labels=labels,
        review_capacity=3,
        min_probability=0.10,
        max_probability=0.99,
        cost_matrix=cost_matrix,
        action_labels=policy_config.action_labels,
        policy_version="test-policy-v1",
    )

    assert evaluation.selected_review_count == 3
    assert evaluation.review_capacity == 3
    assert evaluation.captured_fraud_count == 3
    assert evaluation.total_fraud_count == 4
    assert evaluation.fraud_capture_rate == pytest.approx(0.75)
    assert evaluation.total_expected_review_cost == pytest.approx(15.0)
    assert evaluation.total_expected_prevention_value == pytest.approx(1500.0)


def test_higher_capacity_captures_more_fraud(
    cost_matrix: CostMatrix,
) -> None:
    """Larger review capacity should capture at least as much fraud."""
    probabilities = np.array(
        [0.10, 0.30, 0.50, 0.70, 0.90],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1, 1], dtype=np.int64)

    eval_capacity_2 = evaluate_policy_at_capacity(
        fraud_probabilities=probabilities,
        labels=labels,
        review_capacity=2,
        min_probability=0.0,
        max_probability=1.0,
        cost_matrix=cost_matrix,
        action_labels=("approve", "review", "block"),
        policy_version="test",
    )

    eval_capacity_4 = evaluate_policy_at_capacity(
        fraud_probabilities=probabilities,
        labels=labels,
        review_capacity=4,
        min_probability=0.0,
        max_probability=1.0,
        cost_matrix=cost_matrix,
        action_labels=("approve", "review", "block"),
        policy_version="test",
    )

    assert eval_capacity_4.captured_fraud_count >= eval_capacity_2.captured_fraud_count