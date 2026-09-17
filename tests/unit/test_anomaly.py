"""Unit tests for the anomaly scoring module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.features.anomaly import AnomalyScorerConfig, TimeSafeAnomalyScorer


@pytest.fixture
def train_numeric_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        }
    )


def test_fit_transform_returns_series_with_correct_name(train_numeric_df: pd.DataFrame) -> None:
    scorer = TimeSafeAnomalyScorer(AnomalyScorerConfig(n_estimators=50))
    scores = scorer.fit_transform(train_numeric_df)
    assert isinstance(scores, pd.Series)
    assert scores.name == "anomaly_score_isolation_forest"
    assert len(scores) == len(train_numeric_df)


def test_transform_requires_fit(train_numeric_df: pd.DataFrame) -> None:
    scorer = TimeSafeAnomalyScorer()
    with pytest.raises(RuntimeError):
        scorer.transform(train_numeric_df)


def test_transform_on_new_data_has_same_shape(train_numeric_df: pd.DataFrame) -> None:
    scorer = TimeSafeAnomalyScorer(AnomalyScorerConfig(n_estimators=50))
    scorer.fit(train_numeric_df)

    # New data with same columns
    np.random.seed(99)
    n = 200
    new_df = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        }
    )
    scores = scorer.transform(new_df)
    assert len(scores) == n
    assert scores.name == "anomaly_score_isolation_forest"


def test_finite_scores(train_numeric_df: pd.DataFrame) -> None:
    scorer = TimeSafeAnomalyScorer(AnomalyScorerConfig(n_estimators=50))
    scores = scorer.fit_transform(train_numeric_df)
    assert np.all(np.isfinite(scores.values))
