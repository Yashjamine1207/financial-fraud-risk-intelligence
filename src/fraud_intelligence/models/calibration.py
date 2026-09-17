"""Probability calibration utilities for fraud-risk model outputs.

This module calibrates probabilities from an already trained and frozen model.
It does not train the base fraud classifier and does not load raw project data.

Phase 5 calibration protocol:
- Fit the calibration mapping on the earlier chronological part of validation.
- Compare calibrated probabilities on the later chronological validation part.
- Keep the final holdout test period untouched until every Phase 5 choice is locked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

CalibrationMethod = Literal["uncalibrated", "sigmoid", "isotonic"]

_FLOAT_ARRAY = NDArray[np.float64]
_INT_ARRAY = NDArray[np.int64]


@dataclass(frozen=True)
class CalibrationMetrics:
    """Metrics and reliability-curve values for one calibration method."""

    method: str
    brier_score: float
    expected_calibration_error: float
    mean_predicted_probability: list[float]
    observed_fraud_rate: list[float]
    bin_counts: list[int]


class ProbabilityCalibrator:
    """Fit and apply a single probability-calibration method.

    Parameters
    ----------
    method:
        "uncalibrated" returns the original base-model probabilities.
        "sigmoid" applies Platt scaling.
        "isotonic" applies monotonic isotonic regression.
    probability_epsilon:
        Small clipping value used before taking log-odds for Platt scaling.
    """

    VALID_METHODS: tuple[str, ...] = (
        "uncalibrated",
        "sigmoid",
        "isotonic",
    )

    def __init__(
        self,
        method: CalibrationMethod,
        probability_epsilon: float = 1e-6,
    ) -> None:
        """Initialise an unfitted calibrator."""
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unsupported calibration method: {method}. "
                f"Expected one of {self.VALID_METHODS}."
            )

        if not 0.0 < probability_epsilon < 0.5:
            raise ValueError(
                "probability_epsilon must be greater than 0 and less than 0.5."
            )

        self.method = method
        self.probability_epsilon = probability_epsilon
        self._sigmoid_model: LogisticRegression | None = None
        self._isotonic_model: IsotonicRegression | None = None
        self._is_fitted = False

    def fit(
        self,
        base_probabilities: NDArray[np.floating],
        labels: NDArray[np.integer],
    ) -> ProbabilityCalibrator:
        """Fit the calibration mapping using labelled calibration-period rows.

        The calling workflow is responsible for supplying only the earlier
        chronological calibration portion of validation data. The final test
        split must never be passed to this method during model selection.
        """
        probabilities = self._validate_probabilities(base_probabilities)
        target = self._validate_binary_labels(labels)

        if probabilities.shape[0] != target.shape[0]:
            raise ValueError(
                "base_probabilities and labels must contain the same number of rows."
            )

        if np.unique(target).size != 2:
            raise ValueError(
                "Calibration requires both fraud and non-fraud labels."
            )

        if self.method == "sigmoid":
            # Platt scaling learns a logistic mapping over the base model's log-odds.
            log_odds = self._probabilities_to_log_odds(probabilities)

            self._sigmoid_model = LogisticRegression(
                C=1_000_000.0,
                class_weight=None,
                max_iter=1_000,
                random_state=42,
                solver="lbfgs",
            )
            self._sigmoid_model.fit(log_odds.reshape(-1, 1), target)

        elif self.method == "isotonic":
            # A monotonic non-parametric probability mapping.
            self._isotonic_model = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds="clip",
            )
            self._isotonic_model.fit(probabilities, target)

        self._is_fitted = True
        return self

    def predict(
        self,
        base_probabilities: NDArray[np.floating],
    ) -> _FLOAT_ARRAY:
        """Return calibrated probabilities for a fitted calibration mapping."""
        probabilities = self._validate_probabilities(base_probabilities)

        if self.method == "uncalibrated":
            return probabilities

        if not self._is_fitted:
            raise RuntimeError(
                "The calibrator must be fitted before calling predict()."
            )

        if self.method == "sigmoid":
            if self._sigmoid_model is None:
                raise RuntimeError("The fitted sigmoid calibration model is missing.")

            log_odds = self._probabilities_to_log_odds(probabilities)
            calibrated = self._sigmoid_model.predict_proba(
                log_odds.reshape(-1, 1)
            )[:, 1]

        else:
            if self._isotonic_model is None:
                raise RuntimeError("The fitted isotonic calibration model is missing.")

            calibrated = self._isotonic_model.predict(probabilities)

        return np.clip(
            np.asarray(calibrated, dtype=np.float64),
            0.0,
            1.0,
        )

    def fit_predict(
        self,
        calibration_probabilities: NDArray[np.floating],
        calibration_labels: NDArray[np.integer],
        prediction_probabilities: NDArray[np.floating],
    ) -> _FLOAT_ARRAY:
        """Fit on one period and predict probabilities for a later period."""
        self.fit(calibration_probabilities, calibration_labels)
        return self.predict(prediction_probabilities)

    def _probabilities_to_log_odds(
        self,
        probabilities: _FLOAT_ARRAY,
    ) -> _FLOAT_ARRAY:
        """Convert valid probabilities to finite log-odds values."""
        clipped = np.clip(
            probabilities,
            self.probability_epsilon,
            1.0 - self.probability_epsilon,
        )
        return np.log(clipped / (1.0 - clipped))

    @staticmethod
    def _validate_probabilities(
        values: NDArray[np.floating],
    ) -> _FLOAT_ARRAY:
        """Validate and return a one-dimensional finite probability array."""
        probabilities = np.asarray(values, dtype=np.float64)

        if probabilities.ndim != 1:
            raise ValueError("Probabilities must be a one-dimensional array.")

        if probabilities.size == 0:
            raise ValueError("Probabilities must not be empty.")

        if not np.isfinite(probabilities).all():
            raise ValueError("Probabilities must contain only finite values.")

        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError("Probabilities must be within the range [0.0, 1.0].")

        return probabilities

    @staticmethod
    def _validate_binary_labels(
        values: NDArray[np.integer],
    ) -> _INT_ARRAY:
        """Validate and return a one-dimensional binary label array."""
        labels = np.asarray(values, dtype=np.int64)

        if labels.ndim != 1:
            raise ValueError("Labels must be a one-dimensional array.")

        if labels.size == 0:
            raise ValueError("Labels must not be empty.")

        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Labels must contain only binary values: 0 and 1.")

        return labels


def calculate_expected_calibration_error(
    labels: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    n_bins: int = 10,
) -> float:
    """Calculate quantile-binned expected calibration error.

    ECE is the weighted average absolute difference between the mean predicted
    fraud probability and the observed fraud rate in each populated bin.
    """
    target = ProbabilityCalibrator._validate_binary_labels(labels)
    predicted = ProbabilityCalibrator._validate_probabilities(probabilities)

    if target.shape[0] != predicted.shape[0]:
        raise ValueError("labels and probabilities must contain the same number of rows.")

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    quantile_edges = np.quantile(
        predicted,
        np.linspace(0.0, 1.0, n_bins + 1),
    )
    quantile_edges[0] = 0.0
    quantile_edges[-1] = 1.0

    # Duplicate quantile edges occur when many transactions receive identical
    # model probabilities. Unique edges prevent invalid empty interval handling.
    bin_edges = np.unique(quantile_edges)

    if bin_edges.size < 2:
        return float(abs(float(target.mean()) - float(predicted.mean())))

    bin_indices = np.digitize(
        predicted,
        bin_edges[1:-1],
        right=True,
    )

    expected_error = 0.0
    total_rows = target.shape[0]

    for bin_index in range(bin_edges.size - 1):
        in_bin = bin_indices == bin_index
        bin_count = int(in_bin.sum())

        if bin_count == 0:
            continue

        observed_rate = float(target[in_bin].mean())
        mean_probability = float(predicted[in_bin].mean())

        expected_error += (bin_count / total_rows) * abs(
            observed_rate - mean_probability
        )

    return float(expected_error)


def evaluate_probability_calibration(
    method: str,
    labels: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "quantile",
) -> CalibrationMetrics:
    """Calculate Phase 5 calibration metrics for one set of probabilities."""
    target = ProbabilityCalibrator._validate_binary_labels(labels)
    predicted = ProbabilityCalibrator._validate_probabilities(probabilities)

    if target.shape[0] != predicted.shape[0]:
        raise ValueError("labels and probabilities must contain the same number of rows.")

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be either 'uniform' or 'quantile'.")

    observed_fraud_rate, mean_predicted_probability = calibration_curve(
        target,
        predicted,
        n_bins=n_bins,
        strategy=strategy,
    )

    bin_counts = _calculate_calibration_bin_counts(
        probabilities=predicted,
        n_bins=n_bins,
        strategy=strategy,
    )

    return CalibrationMetrics(
        method=method,
        brier_score=float(brier_score_loss(target, predicted)),
        expected_calibration_error=calculate_expected_calibration_error(
            labels=target,
            probabilities=predicted,
            n_bins=n_bins,
        ),
        mean_predicted_probability=mean_predicted_probability.astype(float).tolist(),
        observed_fraud_rate=observed_fraud_rate.astype(float).tolist(),
        bin_counts=bin_counts,
    )


def _calculate_calibration_bin_counts(
    probabilities: _FLOAT_ARRAY,
    n_bins: int,
    strategy: Literal["uniform", "quantile"],
) -> list[int]:
    """Return populated-bin counts matching the calibration-curve definition."""
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.quantile(
            probabilities,
            np.linspace(0.0, 1.0, n_bins + 1),
        )

    edges = np.unique(edges)

    if edges.size < 2:
        return [int(probabilities.shape[0])]

    bin_indices = np.digitize(
        probabilities,
        edges[1:-1],
        right=True,
    )

    return [
        int((bin_indices == index).sum())
        for index in range(edges.size - 1)
        if int((bin_indices == index).sum()) > 0
    ]