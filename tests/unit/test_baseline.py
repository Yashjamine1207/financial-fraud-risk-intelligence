"""Tests for the Phase 4 Logistic Regression baseline."""

import pandas as pd
import pytest

from fraud_intelligence.models.baseline import (
    build_logistic_regression,
    build_logistic_regression_pipeline,
)


def _build_preprocessing_config() -> dict:
    """Return a minimal valid preprocessing configuration for testing."""
    return {
        "numeric": {
            "imputation_strategy": "median",
            "add_missing_indicator": True,
            "scaler": "standard",
        },
        "categorical": {
            "imputation_strategy": "most_frequent",
            "handle_unknown": "ignore",
            "encoding": "onehot",
        },
    }


def _build_model_config() -> dict:
    """Return a valid Logistic Regression configuration for testing."""
    return {
        "C": 0.1,
        "l1_ratio": 0.0,
        "solver": "saga",
        "max_iter": 500,
        "tolerance": 0.001,
        "class_weight": "balanced",
        "random_state": 42,
    }


def test_logistic_regression_uses_configured_regularisation() -> None:
    """The baseline classifier must use the documented configuration."""
    model = build_logistic_regression(_build_model_config())

    assert model.C == 0.1
    assert model.l1_ratio == 0.0
    assert model.solver == "saga"
    assert model.class_weight == "balanced"
    assert model.random_state == 42


def test_logistic_regression_rejects_missing_configuration() -> None:
    """Missing required hyperparameters must produce a clear error."""
    incomplete_config = _build_model_config()
    del incomplete_config["C"]

    with pytest.raises(ValueError, match="C"):
        build_logistic_regression(incomplete_config)


def test_logistic_regression_pipeline_fits_and_predicts_probabilities() -> None:
    """The full pipeline must fit training data and score validation data."""
    X_train = pd.DataFrame(
        {
            "TransactionAmt": [10.0, 15.0, 100.0, 120.0, 18.0, 150.0],
            "velocity_card1_60min": [0.0, 1.0, 8.0, 10.0, 0.0, 12.0],
            "ProductCD": ["W", "W", "C", "C", "W", "C"],
        }
    )
    y_train = pd.Series([0, 0, 1, 1, 0, 1])

    X_validation = pd.DataFrame(
        {
            "TransactionAmt": [20.0, 130.0],
            "velocity_card1_60min": [1.0, 11.0],
            "ProductCD": ["W", "R"],
        }
    )

    pipeline = build_logistic_regression_pipeline(
        numeric_columns=["TransactionAmt", "velocity_card1_60min"],
        categorical_columns=["ProductCD"],
        preprocessing_config=_build_preprocessing_config(),
        model_config=_build_model_config(),
    )

    pipeline.fit(X_train, y_train)
    fraud_probabilities = pipeline.predict_proba(X_validation)[:, 1]

    assert fraud_probabilities.shape == (2,)
    assert ((fraud_probabilities >= 0.0) & (fraud_probabilities <= 1.0)).all()
