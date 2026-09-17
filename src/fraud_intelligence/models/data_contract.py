"""Leakage-safe modelling data contract for Phase 4.

This module loads the Phase 3 train and validation feature tables, validates
their schema, and selects one shared feature set for every Phase 4 model.

The final test split is deliberately not loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class ModellingData:
    """Validated train/validation data and the shared selected feature columns."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    config: dict[str, Any]


def load_modeling_config(config_path: str | Path = "configs/modeling.yaml") -> dict[str, Any]:
    """Load and validate the shared Phase 4 modelling configuration."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Modelling configuration was not found: {path}. "
            "Create configs/modeling.yaml before running Phase 4."
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Configuration file is empty or invalid: {path}")

    required_sections = {"data", "feature_contract", "preprocessing", "evaluation"}
    missing_sections = required_sections.difference(config)

    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ValueError(f"Modelling configuration is missing required section(s): {missing}")

    return config


def _validate_input_table(
    dataframe: pd.DataFrame,
    dataset_name: str,
    target_column: str,
    transaction_id_column: str,
    timestamp_column: str,
) -> None:
    """Validate the minimum schema and integrity required for Phase 4."""
    required_columns = {
        target_column,
        transaction_id_column,
        timestamp_column,
    }
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{dataset_name} data is missing required column(s): {missing}")

    if dataframe.empty:
        raise ValueError(f"{dataset_name} data is empty.")

    if dataframe[transaction_id_column].isna().any():
        raise ValueError(f"{dataset_name} contains missing values in {transaction_id_column}.")

    if dataframe[transaction_id_column].duplicated().any():
        raise ValueError(f"{dataset_name} contains duplicate values in {transaction_id_column}.")

    if dataframe[timestamp_column].isna().any():
        raise ValueError(f"{dataset_name} contains missing values in {timestamp_column}.")

    valid_target_values = {0, 1}
    observed_target_values = set(dataframe[target_column].dropna().unique())

    if not observed_target_values.issubset(valid_target_values):
        raise ValueError(
            f"{dataset_name} target column {target_column} must contain only "
            f"{sorted(valid_target_values)}. Found: {sorted(observed_target_values)}"
        )


def _validate_temporal_order(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    timestamp_column: str,
) -> None:
    """Confirm validation starts strictly after the training period."""
    latest_train_timestamp = train_dataframe[timestamp_column].max()
    earliest_validation_timestamp = validation_dataframe[timestamp_column].min()

    if earliest_validation_timestamp <= latest_train_timestamp:
        raise ValueError(
            "Temporal split violation: validation data must begin strictly after "
            "the final training timestamp. "
            f"Latest train timestamp={latest_train_timestamp}, "
            f"earliest validation timestamp={earliest_validation_timestamp}."
        )


def _get_excluded_columns(
    train_dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> set[str]:
    """Build the exclusion set using configuration and training data only."""
    feature_contract = config["feature_contract"]

    mandatory_excluded = set(feature_contract.get("mandatory_excluded_columns", []))
    high_missing_columns = set(feature_contract.get("high_missing_columns", []))
    manually_excluded = set(feature_contract.get("manually_excluded_columns", []))

    missing_rate_threshold = feature_contract["max_training_missing_rate"]
    training_missing_rates = train_dataframe.isna().mean()

    dynamically_excluded = set(
        training_missing_rates[training_missing_rates > missing_rate_threshold].index.tolist()
    )

    return mandatory_excluded | high_missing_columns | manually_excluded | dynamically_excluded


def _get_feature_column_types(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Separate selected columns into numerical and categorical columns."""
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    return numeric_columns, categorical_columns


def load_modelling_data(
    config_path: str | Path = "configs/modeling.yaml",
) -> ModellingData:
    """Load train/validation data and return one leakage-safe feature contract.

    The function does not load the final test split. It is intentionally kept
    out of Phase 4 model development and model selection.
    """
    config = load_modeling_config(config_path)
    data_config = config["data"]

    train_path = Path(data_config["train_path"])
    validation_path = Path(data_config["validation_path"])

    if not train_path.exists():
        raise FileNotFoundError(f"Training feature table was not found: {train_path}")

    if not validation_path.exists():
        raise FileNotFoundError(f"Validation feature table was not found: {validation_path}")

    train_dataframe = pd.read_parquet(train_path)
    validation_dataframe = pd.read_parquet(validation_path)

    target_column = data_config["target_column"]
    transaction_id_column = data_config["transaction_id_column"]
    timestamp_column = data_config["timestamp_column"]

    _validate_input_table(
        dataframe=train_dataframe,
        dataset_name="Training",
        target_column=target_column,
        transaction_id_column=transaction_id_column,
        timestamp_column=timestamp_column,
    )
    _validate_input_table(
        dataframe=validation_dataframe,
        dataset_name="Validation",
        target_column=target_column,
        transaction_id_column=transaction_id_column,
        timestamp_column=timestamp_column,
    )
    _validate_temporal_order(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        timestamp_column=timestamp_column,
    )

    train_columns = set(train_dataframe.columns)
    validation_columns = set(validation_dataframe.columns)

    missing_from_validation = train_columns.difference(validation_columns)
    if missing_from_validation:
        missing = ", ".join(sorted(missing_from_validation))
        raise ValueError(
            "Validation data is missing column(s) present in training data: " f"{missing}"
        )

    excluded_columns = _get_excluded_columns(
        train_dataframe=train_dataframe,
        config=config,
    )

    feature_columns = sorted(train_columns.difference(excluded_columns))

    if not feature_columns:
        raise ValueError("No model features remain after applying the feature-exclusion rules.")

    numeric_columns, categorical_columns = _get_feature_column_types(
        dataframe=train_dataframe,
        feature_columns=feature_columns,
    )

    X_train = train_dataframe.loc[:, feature_columns].copy()
    y_train = train_dataframe[target_column].astype("int8").copy()

    X_validation = validation_dataframe.loc[:, feature_columns].copy()
    y_validation = validation_dataframe[target_column].astype("int8").copy()

    return ModellingData(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        config=config,
    )
