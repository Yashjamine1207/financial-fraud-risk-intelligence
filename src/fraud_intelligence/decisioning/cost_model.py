"""Cost-sensitive decision calculations for fraud-risk actions.

The functions in this module convert a calibrated fraud probability into
expected financial cost and expected prevention value for three actions:

- approve: allow the transaction without intervention.
- review: send the transaction to a manual investigation queue.
- block: block or escalate the transaction immediately.

All monetary assumptions are illustrative portfolio-project assumptions.
They must be supplied through configuration, never hard-coded in workflows.

This module calculates expected values from probabilities. It does not:
- Train or calibrate a fraud model.
- Load data.
- Select operational thresholds.
- Access the locked final test split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_FLOAT_ARRAY = NDArray[np.float64]


@dataclass(frozen=True)
class CostMatrix:
    """Versioned financial assumptions for fraud-decision evaluation.

    Parameters
    ----------
    false_negative_cost:
        Expected loss when fraud is approved and therefore missed.
    manual_review_cost:
        Direct analyst-review cost for a transaction placed in the review queue.
    false_positive_escalation_cost:
        Expected customer, operational, and escalation cost when a legitimate
        transaction is blocked or escalated.
    fraud_prevention_value:
        Estimated financial value recovered or prevented when an intervention
        correctly stops a fraud transaction.
    """

    false_negative_cost: float
    manual_review_cost: float
    false_positive_escalation_cost: float
    fraud_prevention_value: float

    def __post_init__(self) -> None:
        """Validate that all cost assumptions are finite and non-negative."""
        values = {
            "false_negative_cost": self.false_negative_cost,
            "manual_review_cost": self.manual_review_cost,
            "false_positive_escalation_cost": self.false_positive_escalation_cost,
            "fraud_prevention_value": self.fraud_prevention_value,
        }

        invalid_values = [
            name
            for name, value in values.items()
            if not np.isfinite(value) or value < 0.0
        ]

        if invalid_values:
            raise ValueError(
                "Cost assumptions must be finite and non-negative. "
                f"Invalid values: {', '.join(invalid_values)}."
            )


@dataclass(frozen=True)
class ExpectedActionCosts:
    """Expected cost and expected prevention value for one transaction."""

    fraud_probability: float
    approve_expected_cost: float
    review_expected_cost: float
    block_expected_cost: float
    review_expected_prevention_value: float
    block_expected_prevention_value: float


def validate_probabilities(
    fraud_probabilities: NDArray[np.floating],
) -> _FLOAT_ARRAY:
    """Validate and return a one-dimensional fraud-probability array."""
    probabilities = np.asarray(fraud_probabilities, dtype=np.float64)

    if probabilities.ndim != 1:
        raise ValueError("Fraud probabilities must be a one-dimensional array.")

    if probabilities.size == 0:
        raise ValueError("Fraud probabilities must not be empty.")

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Fraud probabilities must contain only finite values."
        )

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(
            "Fraud probabilities must be within the range [0.0, 1.0]."
        )

    return probabilities


def calculate_approve_expected_cost(
    fraud_probabilities: NDArray[np.floating],
    cost_matrix: CostMatrix,
) -> _FLOAT_ARRAY:
    """Calculate expected loss if each transaction is approved.

    Approving a transaction has no direct operational intervention cost.
    However, fraud is missed with probability p, producing the configured
    false-negative loss:

        expected_cost(approve) = p * false_negative_cost
    """
    probabilities = validate_probabilities(fraud_probabilities)

    return probabilities * cost_matrix.false_negative_cost


def calculate_review_expected_cost(
    fraud_probabilities: NDArray[np.floating],
    cost_matrix: CostMatrix,
) -> _FLOAT_ARRAY:
    """Calculate direct expected cost of sending transactions to review.

    For the initial Phase 5 policy, manual review is assumed to resolve the
    case correctly before final approval or blocking. Therefore, review cost is
    the fixed analyst-review cost per selected transaction:

        expected_cost(review) = manual_review_cost

    Fraud-prevention value is calculated separately so that costs and benefits
    remain transparent in reports.
    """
    probabilities = validate_probabilities(fraud_probabilities)

    return np.full(
        shape=probabilities.shape,
        fill_value=cost_matrix.manual_review_cost,
        dtype=np.float64,
    )


def calculate_block_expected_cost(
    fraud_probabilities: NDArray[np.floating],
    cost_matrix: CostMatrix,
) -> _FLOAT_ARRAY:
    """Calculate expected cost of blocking or escalating transactions.

    The initial Phase 5 policy assumes a blocked fraud is prevented, while a
    blocked legitimate transaction incurs the configured false-positive
    escalation cost:

        expected_cost(block) = (1 - p) * false_positive_escalation_cost
    """
    probabilities = validate_probabilities(fraud_probabilities)

    return (
        (1.0 - probabilities)
        * cost_matrix.false_positive_escalation_cost
    )


def calculate_intervention_prevention_value(
    fraud_probabilities: NDArray[np.floating],
    cost_matrix: CostMatrix,
) -> _FLOAT_ARRAY:
    """Calculate expected fraud-prevention value of review or block actions.

    The initial Phase 5 assumption is that intervention prevents the configured
    fraud value whenever the transaction is actually fraudulent:

        expected_prevention_value = p * fraud_prevention_value
    """
    probabilities = validate_probabilities(fraud_probabilities)

    return probabilities * cost_matrix.fraud_prevention_value


def calculate_expected_action_costs(
    fraud_probability: float,
    cost_matrix: CostMatrix,
) -> ExpectedActionCosts:
    """Return expected action costs for one calibrated fraud probability."""
    probability_array = validate_probabilities(
        np.asarray([fraud_probability], dtype=np.float64)
    )

    approve_expected_cost = float(
        calculate_approve_expected_cost(
            fraud_probabilities=probability_array,
            cost_matrix=cost_matrix,
        )[0]
    )

    review_expected_cost = float(
        calculate_review_expected_cost(
            fraud_probabilities=probability_array,
            cost_matrix=cost_matrix,
        )[0]
    )

    block_expected_cost = float(
        calculate_block_expected_cost(
            fraud_probabilities=probability_array,
            cost_matrix=cost_matrix,
        )[0]
    )

    intervention_prevention_value = float(
        calculate_intervention_prevention_value(
            fraud_probabilities=probability_array,
            cost_matrix=cost_matrix,
        )[0]
    )

    return ExpectedActionCosts(
        fraud_probability=float(probability_array[0]),
        approve_expected_cost=approve_expected_cost,
        review_expected_cost=review_expected_cost,
        block_expected_cost=block_expected_cost,
        review_expected_prevention_value=intervention_prevention_value,
        block_expected_prevention_value=intervention_prevention_value,
    )


def calculate_review_priority_value(
    fraud_probabilities: NDArray[np.floating],
    cost_matrix: CostMatrix,
) -> _FLOAT_ARRAY:
    """Calculate the expected net value of allocating scarce manual review.

    This is used later to rank transactions when analyst capacity is limited:

        review_priority_value =
            expected fraud-prevention value - manual-review cost

    A larger value means the transaction should be prioritised ahead of lower
    value transactions, subject to the configured review-capacity limit.
    """
    prevention_value = calculate_intervention_prevention_value(
        fraud_probabilities=fraud_probabilities,
        cost_matrix=cost_matrix,
    )

    review_cost = calculate_review_expected_cost(
        fraud_probabilities=fraud_probabilities,
        cost_matrix=cost_matrix,
    )

    return prevention_value - review_cost