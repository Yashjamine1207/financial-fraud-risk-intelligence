"""Unit tests for Phase 5 cost-sensitive fraud decision calculations."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_intelligence.decisioning.cost_model import (
    CostMatrix,
    calculate_approve_expected_cost,
    calculate_block_expected_cost,
    calculate_expected_action_costs,
    calculate_intervention_prevention_value,
    calculate_review_expected_cost,
    calculate_review_priority_value,
    validate_probabilities,
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


def test_cost_matrix_accepts_documented_phase5_values(
    cost_matrix: CostMatrix,
) -> None:
    """The documented portfolio cost assumptions should be valid."""
    assert cost_matrix.false_negative_cost == 500.0
    assert cost_matrix.manual_review_cost == 5.0
    assert cost_matrix.false_positive_escalation_cost == 10.0
    assert cost_matrix.fraud_prevention_value == 500.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "false_negative_cost": -1.0,
            "manual_review_cost": 5.0,
            "false_positive_escalation_cost": 10.0,
            "fraud_prevention_value": 500.0,
        },
        {
            "false_negative_cost": 500.0,
            "manual_review_cost": float("nan"),
            "false_positive_escalation_cost": 10.0,
            "fraud_prevention_value": 500.0,
        },
        {
            "false_negative_cost": 500.0,
            "manual_review_cost": 5.0,
            "false_positive_escalation_cost": float("inf"),
            "fraud_prevention_value": 500.0,
        },
    ],
)
def test_cost_matrix_rejects_invalid_costs(kwargs: dict[str, float]) -> None:
    """Cost assumptions must be finite and non-negative."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        CostMatrix(**kwargs)


def test_approve_expected_cost(
    cost_matrix: CostMatrix,
) -> None:
    """Approve cost equals fraud probability multiplied by missed-fraud loss."""
    probabilities = np.array([0.0, 0.10, 0.50, 1.0])

    result = calculate_approve_expected_cost(
        fraud_probabilities=probabilities,
        cost_matrix=cost_matrix,
    )

    np.testing.assert_allclose(
        result,
        np.array([0.0, 50.0, 250.0, 500.0]),
    )


def test_review_expected_cost(
    cost_matrix: CostMatrix,
) -> None:
    """Review cost is the fixed manual-review cost for every selected case."""
    probabilities = np.array([0.01, 0.50, 0.99])

    result = calculate_review_expected_cost(
        fraud_probabilities=probabilities,
        cost_matrix=cost_matrix,
    )

    np.testing.assert_allclose(
        result,
        np.array([5.0, 5.0, 5.0]),
    )


def test_block_expected_cost(
    cost_matrix: CostMatrix,
) -> None:
    """Block cost equals legitimate-transaction probability times escalation cost."""
    probabilities = np.array([0.0, 0.10, 0.50, 1.0])

    result = calculate_block_expected_cost(
        fraud_probabilities=probabilities,
        cost_matrix=cost_matrix,
    )

    np.testing.assert_allclose(
        result,
        np.array([10.0, 9.0, 5.0, 0.0]),
    )


def test_intervention_prevention_value(
    cost_matrix: CostMatrix,
) -> None:
    """Expected intervention value equals fraud probability times prevention value."""
    probabilities = np.array([0.0, 0.10, 0.50, 1.0])

    result = calculate_intervention_prevention_value(
        fraud_probabilities=probabilities,
        cost_matrix=cost_matrix,
    )

    np.testing.assert_allclose(
        result,
        np.array([0.0, 50.0, 250.0, 500.0]),
    )


def test_review_priority_value(
    cost_matrix: CostMatrix,
) -> None:
    """Review priority is expected prevention value minus analyst-review cost."""
    probabilities = np.array([0.01, 0.10, 0.50, 0.99])

    result = calculate_review_priority_value(
        fraud_probabilities=probabilities,
        cost_matrix=cost_matrix,
    )

    np.testing.assert_allclose(
        result,
        np.array([0.0, 45.0, 245.0, 490.0]),
    )


def test_expected_action_costs_for_one_transaction(
    cost_matrix: CostMatrix,
) -> None:
    """A single risk score should produce transparent expected action costs."""
    result = calculate_expected_action_costs(
        fraud_probability=0.80,
        cost_matrix=cost_matrix,
    )

    assert result.fraud_probability == pytest.approx(0.80)
    assert result.approve_expected_cost == pytest.approx(400.0)
    assert result.review_expected_cost == pytest.approx(5.0)
    assert result.block_expected_cost == pytest.approx(2.0)
    assert result.review_expected_prevention_value == pytest.approx(400.0)
    assert result.block_expected_prevention_value == pytest.approx(400.0)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([], dtype=np.float64),
        np.array([0.10, np.nan]),
        np.array([0.10, np.inf]),
        np.array([-0.01, 0.50]),
        np.array([0.50, 1.01]),
        np.array([[0.10, 0.90]]),
    ],
)
def test_validate_probabilities_rejects_invalid_values(
    probabilities: np.ndarray,
) -> None:
    """Invalid probability arrays must fail clearly before cost calculations."""
    with pytest.raises(ValueError):
        validate_probabilities(probabilities)


def test_higher_probability_produces_higher_review_priority(
    cost_matrix: CostMatrix,
) -> None:
    """Higher calibrated risk should receive a higher expected review value."""
    probabilities = np.array([0.05, 0.20, 0.60, 0.95])

    priorities = calculate_review_priority_value(
        fraud_probabilities=probabilities,
        cost_matrix=cost_matrix,
    )

    assert np.all(np.diff(priorities) > 0)