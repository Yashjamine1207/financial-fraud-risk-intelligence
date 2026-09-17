"""Leakage-safe Isolation Forest anomaly scoring for Phase 6.

The anomaly scorer is trained without fraud labels. However, the training
feature must still be generated safely:

- Training rows receive chronological out-of-fold anomaly scores.
- A row is scored only by an Isolation Forest fitted on earlier training folds.
- Validation and final-test rows are scored by an Isolation Forest fitted on
  the full training period only.
- Missing-value imputation statistics are learned during fitting and reused
  during transformation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ANOMALY_SCORE_COLUMN = "anomaly_score_isolation_forest"


@dataclass(frozen=True)
class AnomalyScorerConfig:
    """Configuration for the Phase 6 Isolation Forest anomaly scorer."""

    n_estimators: int = 200
    contamination: float = 0.035
    max_samples: int = 256
    random_state: int = 42
    n_jobs: int = -1


class TimeSafeAnomalyScorer:
    """Isolation Forest scorer fitted on one historical training period.

    The scorer must be fitted only with transactions available before the
    transactions being scored. It does not use fraud labels.
    """

    def __init__(self, config: AnomalyScorerConfig | None = None) -> None:
        """Initialise the anomaly scorer."""
        self.config = config or AnomalyScorerConfig()
        self._model: IsolationForest | None = None
        self._numeric_columns: list[str] | None = None
        self._training_medians: pd.Series | None = None

    def _prepare_fit_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select numeric features and learn training-only median values."""
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            raise ValueError("Anomaly scoring requires at least one numeric feature column.")

        numeric_data = X.loc[:, numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        training_medians = numeric_data.median()

        self._numeric_columns = numeric_columns
        self._training_medians = training_medians

        return numeric_data.fillna(training_medians).fillna(0.0)

    def _prepare_transform_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align scoring data to the fitted numeric schema and medians."""
        if self._numeric_columns is None or self._training_medians is None:
            raise RuntimeError("Anomaly scorer must be fitted before transform.")

        missing_columns = set(self._numeric_columns).difference(X.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Scoring data is missing numeric feature column(s) required "
                f"by the anomaly scorer: {missing}"
            )

        numeric_data = X.loc[:, self._numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return numeric_data.fillna(self._training_medians).fillna(0.0)

    def fit(self, X: pd.DataFrame) -> TimeSafeAnomalyScorer:
        """Fit Isolation Forest on a historical feature matrix only.

        Parameters
        ----------
        X:
            Historical feature matrix used to fit the anomaly model. Fraud
            labels are not used.

        Returns
        -------
        TimeSafeAnomalyScorer
            The fitted scorer.
        """
        if X is None or X.empty:
            raise ValueError("Training data for anomaly scoring cannot be empty.")

        X_prepared = self._prepare_fit_matrix(X)

        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        self._model.fit(X_prepared)

        return self

    def fit_transform(self, X: pd.DataFrame) -> pd.Series:
        """Fit the scorer and produce anomaly scores for the same rows.

        This method exists for controlled tests and exploration. The Phase 6
        feature-generation script must use chronological out-of-fold scoring
        for training data rather than use this method for train features.
        """
        return self.fit(X).transform(X)

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Return anomaly scores where larger values mean more anomalous.

        Parameters
        ----------
        X:
            Feature matrix to score using the already fitted anomaly model.

        Returns
        -------
        pandas.Series
            One anomaly score per row. Higher values indicate a more anomalous
            transaction according to Isolation Forest.
        """
        if self._model is None:
            raise RuntimeError("Anomaly scorer must be fitted before transform.")

        X_prepared = self._prepare_transform_matrix(X)

        # Isolation Forest returns lower decision values for more anomalous
        # observations. Negate the values so higher means more anomalous.
        anomaly_scores = -self._model.decision_function(X_prepared)

        return pd.Series(
            anomaly_scores,
            index=X.index,
            name=ANOMALY_SCORE_COLUMN,
            dtype="float64",
        )
