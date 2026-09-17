"""Cost-sensitive fraud-decision policy and threshold optimisation.

This module selects approve/review/block thresholds using:
- Calibrated fraud probabilities.
- Explicit financial cost assumptions.
- A configured daily analyst review capacity.

It supports two policy modes:
1. Top-k review: select the k highest-risk transactions for review.
2. Threshold-based review: review all transactions above a calibrated threshold.

This module does not:
- Train or calibrate a fraud model.
- Fit cost assumptions.
- Load data.
- Access the locked final test split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fraud_intelligence.decisioning.cost_model import (
    CostMatrix,
    validate_probabilities,
)

_FLOAT_ARRAY = NDArray[np.float64]
_INT_ARRAY = NDArray[np.int64]


@dataclass(frozen=True)
class DecisionPolicyConfig:
    """Versioned configuration for the fraud decision policy.

    Parameters
    ----------
    currency:
        Currency code used for cost assumptions, e.g. "GBP".
    daily_review_capacity:
        Maximum number of transactions that can be manually reviewed.
    min_review_probability:
        Minimum calibrated fraud probability that may be selected for review.
        Transactions below this floor are never reviewed regardless of capacity.
    max_review_probability:
        Maximum calibrated fraud probability that may be reviewed.
        Transactions above this ceiling are blocked or escalated directly.
    action_labels:
        Labels used for the three actions: approve, review, block.
    """

    currency: str
    daily_review_capacity: int
    min_review_probability: float
    max_review_probability: float
    action_labels: tuple[str, str, str]

    def __post_init__(self) -> None:
        """Validate capacity, probability bounds, and action labels."""
        if self.daily_review_capacity < 1:
            raise ValueError("daily_review_capacity must be at least 1.")

        if not 0.0 <= self.min_review_probability < self.max_review_probability <= 1.0:
            raise ValueError(
                "min_review_probability must be strictly less than "
                "max_review_probability, and both must be within [0.0, 1.0]."
            )

        if len(self.action_labels) != 3:
            raise ValueError("action_labels must contain exactly three labels.")


@dataclass(frozen=True)
class PolicyEvaluation:
    """Evaluation of a fraud-decision policy on one dataset."""

    policy_version: str
    selected_review_count: int
    review_capacity: int
    capacity_utilisation: float
    captured_fraud_count: int
    total_fraud_count: int
    fraud_capture_rate: float
    false_positive_review_count: int
    total_expected_review_cost: float
    total_expected_prevention_value: float
    net_expected_value: float


def validate_binary_labels(
    labels: NDArray[np.integer],
) -> _INT_ARRAY:
    """Validate and return a one-dimensional binary label array."""
    target = np.asarray(labels, dtype=np.int64)

    if target.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array.")

    if target.size == 0:
        raise ValueError("Labels must not be empty.")

    if not np.isin(target, [0, 1]).all():
        raise ValueError("Labels must contain only binary values: 0 and 1.")

    return target


def select_top_k_review_indices(
    fraud_probabilities: _FLOAT_ARRAY,
    labels: _INT_ARRAY | None,
    review_capacity: int,
    min_probability: float,
    max_probability: float,
) -> _INT_ARRAY:
    """Select indices for top-k review under capacity and probability bounds.

    Transactions are ranked by review priority value, which is proportional to
    calibrated fraud probability under the Phase 5 cost assumptions.

    Parameters
    ----------
    fraud_probabilities:
        Calibrated fraud probabilities for all transactions.
    labels:
        Optional true fraud labels for evaluation. If None, fraud-capture
        statistics are not calculated.
    review_capacity:
        Maximum number of transactions to select for review.
    min_probability:
        Minimum fraud probability required to be eligible for review.
    max_probability:
        Maximum fraud probability eligible for review; above this, transactions
        are blocked rather than reviewed.

    Returns
    -------
    review_indices:
        Sorted array of transaction indices selected for review.
    """
    probabilities = validate_probabilities(fraud_probabilities)

    if review_capacity < 1:
        raise ValueError("review_capacity must be at least 1.")

    if min_probability < 0.0 or max_probability > 1.0:
        raise ValueError(
            "Probability bounds must be within the range [0.0, 1.0]."
        )

    if min_probability >= max_probability:
        raise ValueError(
            "min_probability must be strictly less than max_probability."
        )

    eligible_mask = (
        (probabilities >= min_probability)
        & (probabilities <= max_probability)
    )

    eligible_indices = np.flatnonzero(eligible_mask)

    if eligible_indices.size == 0:
        return np.array([], dtype=np.int64)

    eligible_probabilities = probabilities[eligible_indices]

    priority_values = eligible_probabilities

    descending_order = np.argsort(priority_values)[::-1]
    top_k_count = min(review_capacity, eligible_indices.size)
    selected_eligible_indices = eligible_indices[descending_order[:top_k_count]]

    return np.sort(selected_eligible_indices)


def apply_decision_policy(
    fraud_probabilities: _FLOAT_ARRAY,
    labels: _INT_ARRAY,
    review_indices: _INT_ARRAY,
    cost_matrix: CostMatrix,
    action_labels: tuple[str, str, str],
) -> tuple[_INT_ARRAY, dict[str, int], dict[str, float]]:
    """Assign approve/review/block actions and calculate expected costs.

    This function assumes:
    - Reviewed transactions incur the manual-review cost.
    - Fraud is prevented for reviewed fraud cases.
    - Approved fraud cases incur the false-negative cost.
    - Blocked transactions incur the false-positive escalation cost when
      legitimate, and prevent fraud when fraudulent.

    Returns
    -------
    actions:
        Integer action codes per transaction.
    confusion:
        Counts of true positives, false positives, false negatives, true negatives.
    costs:
        Expected total review cost, prevention value, and net value.
    """
    probabilities = validate_probabilities(fraud_probabilities)
    target = validate_binary_labels(labels)

    if probabilities.shape != target.shape:
        raise ValueError(
            "fraud_probabilities and labels must have the same length."
        )

    if (
        review_indices.size > 0
        and (
            review_indices.min() < 0
            or review_indices.max() >= probabilities.size
        )
    ):
        raise ValueError(
            "review_indices contains out-of-range indices."
        )

    actions = np.zeros(probabilities.shape, dtype=np.int64)

    review_code = 1
    block_code = 2

    review_mask = np.zeros(probabilities.shape, dtype=bool)
    review_mask[review_indices] = True

    actions[review_mask] = review_code

    high_risk_mask = probabilities > 0.95
    actions[high_risk_mask & ~review_mask] = block_code

    confusion = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 0,
    }

    for index in range(probabilities.size):
        is_fraud = bool(target[index])
        is_reviewed = bool(review_mask[index])
        is_blocked = bool(actions[index] == block_code)

        if is_fraud and is_reviewed:
            confusion["true_positives"] += 1
        elif not is_fraud and is_reviewed:
            confusion["false_positives"] += 1
        elif is_fraud and not is_reviewed and not is_blocked:
            confusion["false_negatives"] += 1
        elif not is_fraud and not is_reviewed and not is_blocked:
            confusion["true_negatives"] += 1

    review_cost = float(review_indices.size) * cost_matrix.manual_review_cost

    prevented_fraud = int(target[review_indices].sum())
    prevention_value = float(
        prevented_fraud * cost_matrix.fraud_prevention_value
    )

    net_value = prevention_value - review_cost

    costs = {
        "total_expected_review_cost": review_cost,
        "total_expected_prevention_value": prevention_value,
        "net_expected_value": net_value,
    }

    return actions, confusion, costs


def evaluate_policy_at_capacity(
    fraud_probabilities: _FLOAT_ARRAY,
    labels: _INT_ARRAY,
    review_capacity: int,
    min_probability: float,
    max_probability: float,
    cost_matrix: CostMatrix,
    action_labels: tuple[str, str, str],
    policy_version: str,
) -> PolicyEvaluation:
    """Evaluate a top-k review policy at the configured review capacity."""
    review_indices = select_top_k_review_indices(
        fraud_probabilities=fraud_probabilities,
        labels=labels,
        review_capacity=review_capacity,
        min_probability=min_probability,
        max_probability=max_probability,
    )

    _, confusion, costs = apply_decision_policy(
        fraud_probabilities=fraud_probabilities,
        labels=labels,
        review_indices=review_indices,
        cost_matrix=cost_matrix,
        action_labels=action_labels,
    )

    total_fraud = int(labels.sum())
    captured_fraud = int(confusion["true_positives"])

    fraud_capture_rate = (
        float(captured_fraud) / float(total_fraud)
        if total_fraud > 0
        else 0.0
    )

    capacity_utilisation = (
        float(review_indices.size) / float(review_capacity)
        if review_capacity > 0
        else 0.0
    )

    return PolicyEvaluation(
        policy_version=policy_version,
        selected_review_count=int(review_indices.size),
        review_capacity=review_capacity,
        capacity_utilisation=capacity_utilisation,
        captured_fraud_count=captured_fraud,
        total_fraud_count=total_fraud,
        fraud_capture_rate=fraud_capture_rate,
        false_positive_review_count=int(confusion["false_positives"]),
        total_expected_review_cost=costs["total_expected_review_cost"],
        total_expected_prevention_value=costs["total_expected_prevention_value"],
        net_expected_value=costs["net_expected_value"],
    )