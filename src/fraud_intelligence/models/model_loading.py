"""Load frozen fraud-model components for Phase 5 and later phases.

This module loads the exact preprocessing pipeline and XGBoost classifier
recorded in the selected Phase 4 MLflow run.

It does not:
- Fit preprocessing.
- Train a model.
- Fit a calibration mapping.
- Load the final locked test split.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import yaml
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

DEFAULT_CALIBRATION_CONFIG_PATH = Path("configs/calibration.yaml")


@dataclass(frozen=True)
class FrozenXGBoostComponents:
    """Frozen Phase 4 XGBoost components loaded from MLflow."""

    preprocessor: ColumnTransformer
    classifier: XGBClassifier
    model_name: str
    model_version: str
    mlflow_run_id: str
    preprocessor_model_uri: str
    classifier_model_uri: str


def load_calibration_config(
    config_path: Path = DEFAULT_CALIBRATION_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the Phase 5 calibration configuration."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Calibration configuration was not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(
            "Calibration configuration must parse to a dictionary."
        )

    if "champion_model" not in config:
        raise ValueError(
            "Calibration configuration is missing the 'champion_model' section."
        )

    champion_model = config["champion_model"]

    if not isinstance(champion_model, dict):
        raise TypeError(
            "The 'champion_model' configuration must be a dictionary."
        )

    required_keys = (
        "model_name",
        "model_version",
        "mlflow_run_id",
        "tracking_uri_env_var",
        "preprocessor_artifact_path",
        "classifier_artifact_path",
    )

    missing_keys = [
        key
        for key in required_keys
        if key not in champion_model
        or champion_model[key] in {None, ""}
    ]

    if missing_keys:
        missing_key_text = ", ".join(missing_keys)
        raise ValueError(
            "Calibration configuration is missing required champion-model "
            f"settings: {missing_key_text}."
        )

    return config


def load_frozen_xgboost_components(
    config_path: Path = DEFAULT_CALIBRATION_CONFIG_PATH,
) -> FrozenXGBoostComponents:
    """Load the frozen Phase 4 XGBoost preprocessor and classifier from MLflow.

    The model run ID and artifact paths come only from `configs/calibration.yaml`.
    This function does not fit or alter either loaded object.
    """
    load_dotenv()

    config = load_calibration_config(config_path)
    champion_model = config["champion_model"]

    tracking_uri_environment_variable = champion_model["tracking_uri_env_var"]
    tracking_uri = os.getenv(tracking_uri_environment_variable)

    if not tracking_uri:
        raise RuntimeError(
            f"Environment variable '{tracking_uri_environment_variable}' is not set. "
            "Set it in the local .env file before loading MLflow models."
        )

    mlflow.set_tracking_uri(tracking_uri)

    run_id = str(champion_model["mlflow_run_id"])
    preprocessor_artifact_path = str(
        champion_model["preprocessor_artifact_path"]
    )
    classifier_artifact_path = str(
        champion_model["classifier_artifact_path"]
    )

    preprocessor_model_uri = (
        f"runs:/{run_id}/{preprocessor_artifact_path}"
    )
    classifier_model_uri = (
        f"runs:/{run_id}/{classifier_artifact_path}"
    )

    try:
        loaded_preprocessor = mlflow.sklearn.load_model(
            preprocessor_model_uri
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the frozen XGBoost preprocessor from MLflow. "
            f"Model URI: {preprocessor_model_uri}"
        ) from exc

    try:
        loaded_classifier = mlflow.xgboost.load_model(
            classifier_model_uri
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the frozen XGBoost classifier from MLflow. "
            f"Model URI: {classifier_model_uri}"
        ) from exc

    if not isinstance(loaded_preprocessor, ColumnTransformer):
        raise TypeError(
            "The loaded preprocessing artifact is not a sklearn "
            f"ColumnTransformer. Found: {type(loaded_preprocessor).__name__}."
        )

    if not isinstance(loaded_classifier, XGBClassifier):
        raise TypeError(
            "The loaded classifier artifact is not an XGBClassifier. "
            f"Found: {type(loaded_classifier).__name__}."
        )

    return FrozenXGBoostComponents(
        preprocessor=loaded_preprocessor,
        classifier=loaded_classifier,
        model_name=str(champion_model["model_name"]),
        model_version=str(champion_model["model_version"]),
        mlflow_run_id=run_id,
        preprocessor_model_uri=preprocessor_model_uri,
        classifier_model_uri=classifier_model_uri,
    )