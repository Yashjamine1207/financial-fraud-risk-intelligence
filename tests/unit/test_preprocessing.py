"""Tests for the shared Phase 4 preprocessing pipeline."""

import pandas as pd
import pytest

from fraud_intelligence.models.preprocessing import build_model_preprocessor


def _build_preprocessing_config() -> dict:
    """Return a minimal preprocessing configuration for unit tests."""
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


def test_preprocessor_transforms_missing_values_and_unseen_categories() -> None:
    """The train-fitted pipeline must transform validation data safely."""
    X_train = pd.DataFrame(
        {
            "TransactionAmt": [10.0, None, 30.0, 40.0],
            "velocity_card1_60min": [0.0, 2.0, None, 5.0],
            "ProductCD": ["W", "C", "W", None],
            "card4": ["visa", "mastercard", "visa", "visa"],
        }
    )
    X_validation = pd.DataFrame(
        {
            "TransactionAmt": [20.0, None],
            "velocity_card1_60min": [1.0, 4.0],
            "ProductCD": ["R", "W"],
            "card4": ["discover", "visa"],
        }
    )

    preprocessor = build_model_preprocessor(
        numeric_columns=["TransactionAmt", "velocity_card1_60min"],
        categorical_columns=["ProductCD", "card4"],
        preprocessing_config=_build_preprocessing_config(),
    )

    transformed_train = preprocessor.fit_transform(X_train)
    transformed_validation = preprocessor.transform(X_validation)

    assert transformed_train.shape[0] == len(X_train)
    assert transformed_validation.shape[0] == len(X_validation)
    assert transformed_train.shape[1] == transformed_validation.shape[1]
    assert transformed_train.shape[1] > len(X_train.columns)


def test_preprocessor_rejects_empty_feature_contract() -> None:
    """An empty feature contract must fail with a clear error."""
    with pytest.raises(ValueError, match="without numeric or categorical"):
        build_model_preprocessor(
            numeric_columns=[],
            categorical_columns=[],
            preprocessing_config=_build_preprocessing_config(),
        )
