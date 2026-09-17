"""Tests for the leakage-safe Phase 4 modelling data contract."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fraud_intelligence.models.data_contract import load_modelling_data


def _write_test_config(
    temporary_path: Path,
    train_path: Path,
    validation_path: Path,
) -> Path:
    """Create a minimal modelling configuration for an isolated unit test."""
    config = {
        "data": {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "test_path": "data/interim/temporal_splits/test.parquet",
            "test_locked": True,
            "target_column": "isFraud",
            "transaction_id_column": "TransactionID",
            "timestamp_column": "TransactionDT",
        },
        "feature_contract": {
            "mandatory_excluded_columns": [
                "TransactionID",
                "TransactionDT",
                "isFraud",
            ],
            "high_missing_columns": ["mostly_missing"],
            "manually_excluded_columns": [],
            "max_training_missing_rate": 0.95,
            "preserve_missingness_signal": True,
            "target_encoded_columns": [],
        },
        "preprocessing": {
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
        },
        "evaluation": {
            "review_capacity": 1000,
            "reference_probability_threshold": 0.50,
            "metrics": ["pr_auc"],
        },
    }

    config_path = temporary_path / "modeling.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    return config_path


def test_load_modelling_data_excludes_leakage_columns_and_high_null_columns(
    tmp_path: Path,
) -> None:
    """The contract must select only shared, permitted model features."""
    train_dataframe = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 200, 300],
            "isFraud": [0, 1, 0],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "ProductCD": ["W", "C", "W"],
            "mostly_missing": [None, None, 5.0],
        }
    )
    validation_dataframe = pd.DataFrame(
        {
            "TransactionID": [4, 5],
            "TransactionDT": [400, 500],
            "isFraud": [1, 0],
            "TransactionAmt": [40.0, 50.0],
            "ProductCD": ["C", "R"],
            "mostly_missing": [None, None],
        }
    )

    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"

    train_dataframe.to_parquet(train_path, index=False)
    validation_dataframe.to_parquet(validation_path, index=False)

    config_path = _write_test_config(
        temporary_path=tmp_path,
        train_path=train_path,
        validation_path=validation_path,
    )

    modelling_data = load_modelling_data(config_path)

    assert modelling_data.feature_columns == ["ProductCD", "TransactionAmt"]
    assert list(modelling_data.X_train.columns) == [
        "ProductCD",
        "TransactionAmt",
    ]
    assert list(modelling_data.X_validation.columns) == [
        "ProductCD",
        "TransactionAmt",
    ]
    assert modelling_data.numeric_columns == ["TransactionAmt"]
    assert modelling_data.categorical_columns == ["ProductCD"]
    assert modelling_data.y_train.tolist() == [0, 1, 0]
    assert modelling_data.y_validation.tolist() == [1, 0]


def test_load_modelling_data_rejects_invalid_temporal_split(tmp_path: Path) -> None:
    """Validation data must begin after the complete training period."""
    train_dataframe = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [100, 300],
            "isFraud": [0, 1],
            "TransactionAmt": [10.0, 20.0],
        }
    )
    validation_dataframe = pd.DataFrame(
        {
            "TransactionID": [3],
            "TransactionDT": [300],
            "isFraud": [0],
            "TransactionAmt": [30.0],
        }
    )

    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"

    train_dataframe.to_parquet(train_path, index=False)
    validation_dataframe.to_parquet(validation_path, index=False)

    config_path = _write_test_config(
        temporary_path=tmp_path,
        train_path=train_path,
        validation_path=validation_path,
    )

    with pytest.raises(ValueError, match="Temporal split violation"):
        load_modelling_data(config_path)
