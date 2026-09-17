"""Train and evaluate XGBoost with an Isolation Forest anomaly feature.

Phase 6 anomaly ablation experiment:

Baseline:
    Phase 4 XGBoost using the existing point-in-time feature contract.

Candidate:
    The same XGBoost configuration and preprocessing pipeline, plus one
    leakage-safe feature:
    anomaly_score_isolation_forest

Leakage controls:
- The modelling data contract loads train and validation only.
- The final test split is not loaded by this script.
- Training anomaly scores were generated through chronological out-of-fold
  scoring, using only earlier training folds for every scored fold.
- Validation anomaly scores were generated using an Isolation Forest fitted
  only on the full chronological training period.
- Fraud labels are never used by Isolation Forest.
- Preprocessing, class weighting, and XGBoost fitting use training data only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from fraud_intelligence.models.data_contract import (
    load_modeling_config,
    load_modelling_data,
)
from fraud_intelligence.models.evaluation import evaluate_binary_classifier
from fraud_intelligence.models.xgboost_model import (
    build_xgboost_classifier,
    build_xgboost_preprocessor,
    calculate_scale_pos_weight,
)

XGBOOST_CONFIG_PATH = Path("configs/xgboost.yaml")
ANOMALY_DIRECTORY = Path("data/features/anomaly")
METRICS_OUTPUT_PATH = Path("reports/tables/xgboost_anomaly_validation_metrics.json")
FEATURE_CONTRACT_OUTPUT_PATH = Path("reports/tables/xgboost_anomaly_feature_contract.json")

EXPERIMENT_NAME = "financial-fraud-risk-intelligence"
ANOMALY_FEATURE_NAME = "anomaly_score_isolation_forest"
ANOMALY_FEATURE_VERSION = "isolation-forest-v1.0.0"


def load_xgboost_config() -> dict[str, Any]:
    """Load and validate the existing XGBoost YAML configuration."""
    if not XGBOOST_CONFIG_PATH.exists():
        raise FileNotFoundError(f"XGBoost configuration was not found: {XGBOOST_CONFIG_PATH}")

    with XGBOOST_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("XGBoost configuration must parse to a dictionary.")

    if "model" not in config:
        raise ValueError("XGBoost configuration is missing the 'model' section.")

    return config


def configure_mlflow() -> None:
    """Configure MLflow with the project tracking URI and experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")


def load_anomaly_scores(
    filename: str,
    expected_rows: int,
) -> pd.Series:
    """Load and validate one saved anomaly score series."""
    path = ANOMALY_DIRECTORY / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Anomaly score file was not found: {path}. "
            "Run scripts/generate_anomaly_score_feature.py first."
        )

    anomaly_dataframe = pd.read_parquet(path)

    if ANOMALY_FEATURE_NAME not in anomaly_dataframe.columns:
        raise ValueError(
            f"Anomaly score file {path} does not contain the required column "
            f"'{ANOMALY_FEATURE_NAME}'."
        )

    anomaly_scores = anomaly_dataframe[ANOMALY_FEATURE_NAME].copy()

    if len(anomaly_scores) != expected_rows:
        raise ValueError(
            f"Anomaly score row count does not match the modelling data for {path}. "
            f"Expected {expected_rows:,}, found {len(anomaly_scores):,}."
        )

    if np.isinf(anomaly_scores.to_numpy(dtype=float)).any():
        raise ValueError(f"Anomaly score file contains infinite values: {path}")

    return anomaly_scores


def append_anomaly_feature(
    X: pd.DataFrame,
    anomaly_scores: pd.Series,
) -> pd.DataFrame:
    """Append one validated anomaly score feature without changing row order."""
    if len(X) != len(anomaly_scores):
        raise ValueError("Feature matrix and anomaly scores must contain the same number of rows.")

    if ANOMALY_FEATURE_NAME in X.columns:
        raise ValueError(
            f"Feature matrix already contains '{ANOMALY_FEATURE_NAME}'. "
            "Do not append the anomaly feature twice."
        )

    augmented_X = X.copy()
    augmented_X[ANOMALY_FEATURE_NAME] = anomaly_scores.to_numpy()

    return augmented_X


def save_json(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a small JSON report suitable for Git tracking."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Saved report to: {output_path}")


def print_validation_results(metrics: dict[str, Any]) -> None:
    """Print the main Phase 6 anomaly-ablation validation metrics."""
    print("\nXGBoost + anomaly validation results")
    print("-" * 56)
    print(f"PR-AUC: {metrics['pr_auc']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print("Precision at reference threshold: " f"{metrics['precision_at_reference_threshold']:.6f}")
    print("Recall at reference threshold: " f"{metrics['recall_at_reference_threshold']:.6f}")
    print("F1 at reference threshold: " f"{metrics['f1_at_reference_threshold']:.6f}")
    print(f"Precision@{metrics['review_capacity']:,}: " f"{metrics['precision_at_k']:.6f}")
    print(f"Recall@{metrics['review_capacity']:,}: " f"{metrics['recall_at_k']:.6f}")
    print(
        "Captured fraud at review capacity: "
        f"{metrics['captured_fraud_count_at_k']:,} / "
        f"{metrics['total_fraud_count']:,}"
    )
    print("Prediction latency per row: " f"{metrics['prediction_latency_ms_per_row']:.6f} ms")
    print(f"Training time: {metrics['training_time_seconds']:.2f} seconds")
    print(f"Best boosting iteration: {metrics['best_iteration']}")
    print(f"Training anomaly-score missing rows: {metrics['train_anomaly_missing_rows']:,}")
    print("Final test split loaded: False")


def main() -> None:
    """Run the Phase 6 XGBoost-plus-anomaly ablation on validation only."""
    load_dotenv()

    modeling_config = load_modeling_config()
    xgboost_config = load_xgboost_config()
    modelling_data = load_modelling_data()

    model_config = xgboost_config["model"]
    evaluation_config = modeling_config["evaluation"]
    reproducibility_config = modeling_config["reproducibility"]

    configure_mlflow()

    # The shared Phase 4 contract already excludes TransactionID,
    # TransactionDT, isFraud, and high-missingness columns.
    X_train_base = modelling_data.X_train.copy()
    X_validation_base = modelling_data.X_validation.copy()

    train_anomaly_scores = load_anomaly_scores(
        filename="train_anomaly_scores.parquet",
        expected_rows=len(X_train_base),
    )
    validation_anomaly_scores = load_anomaly_scores(
        filename="val_anomaly_scores.parquet",
        expected_rows=len(X_validation_base),
    )

    # The early chronological fold has no prior history and is intentionally
    # missing. Existing numeric preprocessing uses median imputation and a
    # missingness indicator fitted only on training data.
    train_anomaly_missing_rows = int(train_anomaly_scores.isna().sum())

    X_train = append_anomaly_feature(
        X=X_train_base,
        anomaly_scores=train_anomaly_scores,
    )
    X_validation = append_anomaly_feature(
        X=X_validation_base,
        anomaly_scores=validation_anomaly_scores,
    )

    augmented_feature_columns = [
        *modelling_data.feature_columns,
        ANOMALY_FEATURE_NAME,
    ]
    augmented_numeric_columns = [
        *modelling_data.numeric_columns,
        ANOMALY_FEATURE_NAME,
    ]
    augmented_categorical_columns = list(modelling_data.categorical_columns)

    scale_pos_weight = calculate_scale_pos_weight(modelling_data.y_train)

    # Reuse the exact Phase 4 preprocessing construction, with the one added
    # numeric anomaly feature.
    preprocessor = build_xgboost_preprocessor(
        numeric_columns=augmented_numeric_columns,
        categorical_columns=augmented_categorical_columns,
        preprocessing_config=modeling_config["preprocessing"],
    )

    print("\nFitting preprocessing on chronological training data...")
    preprocessing_start_time = time.perf_counter()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)

    preprocessing_time_seconds = time.perf_counter() - preprocessing_start_time

    # Reuse the exact current XGBoost hyperparameters from configs/xgboost.yaml.
    classifier = build_xgboost_classifier(
        model_config=model_config,
        scale_pos_weight=scale_pos_weight,
    )

    run_name = "xgboost-plus-isolation-forest-anomaly-v1.0.0"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project_name": modeling_config["project"]["name"],
                "phase": "6",
                "experiment_type": "controlled_anomaly_ablation",
                "model_name": model_config["name"],
                "model_version": "xgboost-plus-anomaly-v1.0.0",
                "base_model_version": model_config["model_version"],
                "anomaly_feature": ANOMALY_FEATURE_NAME,
                "anomaly_feature_version": ANOMALY_FEATURE_VERSION,
                "dataset_version": reproducibility_config["dataset_version"],
                "feature_version": reproducibility_config["feature_version"],
                "target_encoding_version": reproducibility_config["target_encoding_version"],
                "validation_strategy": "chronological_train_validation",
                "training_anomaly_protocol": "chronological_out_of_fold",
                "validation_anomaly_protocol": "fit_train_transform_validation",
                "fraud_labels_used_by_anomaly_model": "false",
                "test_split_loaded": "false",
            }
        )

        mlflow.log_params(
            {
                **{
                    key: value
                    for key, value in model_config.items()
                    if key not in {"name", "model_version"}
                },
                "scale_pos_weight_calculated_from_train": scale_pos_weight,
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "base_selected_feature_count": len(modelling_data.feature_columns),
                "augmented_selected_feature_count": len(augmented_feature_columns),
                "base_numeric_feature_count": len(modelling_data.numeric_columns),
                "augmented_numeric_feature_count": len(augmented_numeric_columns),
                "categorical_feature_count": len(augmented_categorical_columns),
                "review_capacity": evaluation_config["review_capacity"],
                "reference_probability_threshold": evaluation_config[
                    "reference_probability_threshold"
                ],
                "train_anomaly_missing_rows": train_anomaly_missing_rows,
            }
        )

        print("Fitting XGBoost + anomaly on chronological training data...")
        training_start_time = time.perf_counter()

        classifier.fit(
            X_train_transformed,
            modelling_data.y_train,
            eval_set=[
                (
                    X_validation_transformed,
                    modelling_data.y_validation,
                )
            ],
            verbose=False,
        )

        training_time_seconds = time.perf_counter() - training_start_time

        print("Scoring chronological validation data...")
        prediction_start_time = time.perf_counter()

        validation_probabilities = classifier.predict_proba(X_validation_transformed)[:, 1]

        total_prediction_time_seconds = time.perf_counter() - prediction_start_time

        # X_validation_transformed is sparse, so shape[0] must be used instead
        # of len(...).
        prediction_latency_ms_per_row = (
            total_prediction_time_seconds / X_validation_transformed.shape[0]
        ) * 1_000

        metrics = evaluate_binary_classifier(
            y_true=modelling_data.y_validation,
            fraud_probabilities=validation_probabilities,
            review_capacity=evaluation_config["review_capacity"],
            reference_threshold=evaluation_config["reference_probability_threshold"],
            prediction_latency_ms_per_row=prediction_latency_ms_per_row,
        )

        metrics["training_time_seconds"] = float(training_time_seconds)
        metrics["preprocessing_time_seconds"] = float(preprocessing_time_seconds)
        metrics["best_iteration"] = int(classifier.best_iteration)
        metrics["best_validation_aucpr"] = float(classifier.best_score)
        metrics["scale_pos_weight"] = float(scale_pos_weight)
        metrics["run_id"] = run.info.run_id
        metrics["model_name"] = model_config["name"]
        metrics["model_version"] = "xgboost-plus-anomaly-v1.0.0"
        metrics["base_model_version"] = model_config["model_version"]
        metrics["anomaly_feature_name"] = ANOMALY_FEATURE_NAME
        metrics["anomaly_feature_version"] = ANOMALY_FEATURE_VERSION
        metrics["train_anomaly_missing_rows"] = train_anomaly_missing_rows
        metrics["dataset_version"] = reproducibility_config["dataset_version"]
        metrics["feature_version"] = reproducibility_config["feature_version"]
        metrics["target_encoding_version"] = reproducibility_config["target_encoding_version"]
        metrics["test_split_loaded"] = False

        numeric_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        mlflow.log_metrics(numeric_metrics)

        feature_contract = {
            "base_feature_columns": modelling_data.feature_columns,
            "augmented_feature_columns": augmented_feature_columns,
            "numeric_columns": augmented_numeric_columns,
            "categorical_columns": augmented_categorical_columns,
            "added_feature": ANOMALY_FEATURE_NAME,
            "added_feature_version": ANOMALY_FEATURE_VERSION,
            "training_anomaly_protocol": "chronological_out_of_fold",
            "validation_anomaly_protocol": "fit_train_transform_validation",
            "test_split_loaded": False,
        }

        save_json(metrics, METRICS_OUTPUT_PATH)
        save_json(feature_contract, FEATURE_CONTRACT_OUTPUT_PATH)

        mlflow.log_dict(
            feature_contract,
            "anomaly_feature_contract.json",
        )

        # Different artifact names ensure the frozen Phase 4 model artifacts
        # cannot be overwritten or confused with this Phase 6 candidate model.
        mlflow.sklearn.log_model(
            sk_model=preprocessor,
            name="xgboost_anomaly_preprocessor",
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype"],
        )

        mlflow.xgboost.log_model(
            xgb_model=classifier,
            name="xgboost_anomaly_classifier",
            model_format="json",
        )

        print_validation_results(metrics)
        print(f"\nMLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
