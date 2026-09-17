"""Unit tests for Phase 5 probability-calibration utilities."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_intelligence.models.calibration import (
    ProbabilityCalibrator,
    calculate_expected_calibration_error,
    evaluate_probability_calibration,
)


@pytest.fixture
def calibration_data() -> tuple[np.ndarray, np.ndarray]:
    """Return small labelled data suitable for deterministic calibration tests."""
    probabilities = np.array(
        [0.01, 0.05, 0.10, 0.20, 0.30, 0.55, 0.70, 0.80, 0.90, 0.99],
        dtype=np.float64,
    )
    labels = np.array(
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
        dtype=np.int64,
    )
    return probabilities, labels


def test_uncalibrated_method_returns_original_probabilities(
    calibration_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """The uncalibrated benchmark must return original model probabilities."""
    probabilities, labels = calibration_data
    calibrator = ProbabilityCalibrator(method="uncalibrated")

    calibrator.fit(probabilities, labels)
    result = calibrator.predict(probabilities)

    np.testing.assert_allclose(result, probabilities)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_fitted_calibrators_return_valid_probabilities(
    method: str,
    calibration_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Fitted calibration methods must output finite values within [0, 1]."""
    probabilities, labels = calibration_data
    calibrator = ProbabilityCalibrator(method=method)

    calibrator.fit(probabilities, labels)
    result = calibrator.predict(np.array([0.02, 0.25, 0.60, 0.95]))

    assert result.shape == (4,)
    assert np.isfinite(result).all()
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_rejects_prediction_before_fit(method: str) -> None:
    """Learned calibration mappings must not be used before fitting."""
    calibrator = ProbabilityCalibrator(method=method)

    with pytest.raises(RuntimeError, match="must be fitted"):
        calibrator.predict(np.array([0.10, 0.90]))


def test_sigmoid_calibration_uses_fit_period_labels_only(
    calibration_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Prediction must not require or consume labels from a later period."""
    fit_probabilities, fit_labels = calibration_data
    later_probabilities = np.array([0.15, 0.45, 0.85], dtype=np.float64)

    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(fit_probabilities, fit_labels)
    result = calibrator.predict(later_probabilities)

    assert result.shape == later_probabilities.shape
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()


@pytest.mark.parametrize(
    ("probabilities", "error_message"),
    [
        (np.array([], dtype=np.float64), "must not be empty"),
        (np.array([0.10, np.nan]), "finite values"),
        (np.array([-0.01, 0.50]), "range"),
        (np.array([0.50, 1.01]), "range"),
        (np.array([[0.10, 0.90]]), "one-dimensional"),
    ],
)
def test_calibrator_rejects_invalid_probabilities(
    probabilities: np.ndarray,
    error_message: str,
) -> None:
    """Invalid model-probability arrays must fail with a clear error."""
    calibrator = ProbabilityCalibrator(method="uncalibrated")

    with pytest.raises(ValueError, match=error_message):
        calibrator.predict(probabilities)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([], dtype=np.int64),
        np.array([0, 2], dtype=np.int64),
        np.array([0, -1], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64),
    ],
)
def test_calibrator_rejects_invalid_labels(labels: np.ndarray) -> None:
    """Calibration fitting requires a non-empty one-dimensional binary target."""
    calibrator = ProbabilityCalibrator(method="sigmoid")

    with pytest.raises(ValueError):
        calibrator.fit(np.array([0.20, 0.80]), labels)


def test_calibrator_requires_both_classes_to_fit() -> None:
    """A calibration mapping cannot be learned from a single target class."""
    calibrator = ProbabilityCalibrator(method="isotonic")

    with pytest.raises(ValueError, match="both fraud and non-fraud"):
        calibrator.fit(
            np.array([0.10, 0.20, 0.30], dtype=np.float64),
            np.array([0, 0, 0], dtype=np.int64),
        )


def test_expected_calibration_error_for_known_example() -> None:
    """ECE should match the manually calculated value for two equal-sized bins."""
    probabilities = np.array([0.10, 0.20, 0.80, 0.90], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    result = calculate_expected_calibration_error(
        labels=labels,
        probabilities=probabilities,
        n_bins=2,
    )

    # Lower bin error: |0.15 - 0.00| = 0.15
    # Upper bin error: |0.85 - 1.00| = 0.15
    # Both bins contain 50% of records, so ECE = 0.15.
    assert result == pytest.approx(0.15)


def test_calibration_evaluation_returns_required_phase5_metrics(
    calibration_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Phase 5 evaluation must return Brier, ECE, curve values, and bin counts."""
    probabilities, labels = calibration_data

    result = evaluate_probability_calibration(
        method="uncalibrated",
        labels=labels,
        probabilities=probabilities,
        n_bins=5,
        strategy="quantile",
    )

    assert result.method == "uncalibrated"
    assert result.brier_score >= 0.0
    assert result.expected_calibration_error >= 0.0
    assert len(result.mean_predicted_probability) > 0
    assert len(result.observed_fraud_rate) > 0
    assert len(result.bin_counts) > 0
    assert len(result.mean_predicted_probability) == len(
        result.observed_fraud_rate
    )