"""Train and evaluate the Phase 4 XGBoost fraud model.

This script:
- Uses only chronological Phase 3 train and validation feature tables.
- Fits preprocessing only on training data.
- Uses validation data only for early stopping and evaluation.
- Does not load the locked final test split.
- Logs configuration, metrics, versions, and artifacts to MLflow.
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
import yaml
from dotenv import load_dotenv

from fraud_intelligence.models.data_contract import load_modeling_config, load_modelling_data
from fraud_intelligence.models.evaluation import evaluate_binary_classifier
from fraud_intelligence.models.xgboost_model import (
    build_xgboost_classifier,
    build_xgboost_preprocessor,
    calculate_scale_pos_weight,
)

XGBOOST_CONFIG_PATH = Path("configs/xgboost.yaml")
METRICS_OUTPUT_PATH = Path("reports/tables/xgboost_validation_metrics.json")
EXPERIMENT_NAME = "financial-fraud-risk-intelligence"


def load_xgboost_config() -> dict[str, Any]:
    """Load the XGBoost configuration from YAML."""
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
    """Configure MLflow using the project environment configuration."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")


def save_validation_metrics(metrics: dict[str, Any]) -> None:
    """Save the small, Git-trackable XGBoost validation metrics report."""
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved validation metrics to: {METRICS_OUTPUT_PATH}")


def print_validation_results(metrics: dict[str, Any]) -> None:
    """Print the main XGBoost validation results in the terminal."""
    print("\nXGBoost validation results")
    print("-" * 48)
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
    print("Final test split loaded: False")


def main() -> None:
    """Train, validate, save, and track the Phase 4 XGBoost experiment."""
    load_dotenv()

    modeling_config = load_modeling_config()
    xgboost_config = load_xgboost_config()
    modelling_data = load_modelling_data()

    model_config = xgboost_config["model"]
    evaluation_config = modeling_config["evaluation"]
    reproducibility_config = modeling_config["reproducibility"]

    configure_mlflow()

    scale_pos_weight = calculate_scale_pos_weight(modelling_data.y_train)

    preprocessor = build_xgboost_preprocessor(
        numeric_columns=modelling_data.numeric_columns,
        categorical_columns=modelling_data.categorical_columns,
        preprocessing_config=modeling_config["preprocessing"],
    )

    print("\nFitting preprocessing on chronological training data...")
    preprocessing_start_time = time.perf_counter()

    X_train_transformed = preprocessor.fit_transform(modelling_data.X_train)
    X_validation_transformed = preprocessor.transform(modelling_data.X_validation)

    preprocessing_time_seconds = time.perf_counter() - preprocessing_start_time

    classifier = build_xgboost_classifier(
        model_config=model_config,
        scale_pos_weight=scale_pos_weight,
    )

    with mlflow.start_run(run_name=model_config["model_version"]) as run:
        mlflow.set_tags(
            {
                "project_name": modeling_config["project"]["name"],
                "phase": str(modeling_config["project"]["phase"]),
                "model_name": model_config["name"],
                "model_version": model_config["model_version"],
                "dataset_version": reproducibility_config["dataset_version"],
                "feature_version": reproducibility_config["feature_version"],
                "target_encoding_version": reproducibility_config["target_encoding_version"],
                "validation_strategy": "chronological_train_validation",
                "early_stopping_metric": model_config["eval_metric"],
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
                "train_rows": len(modelling_data.X_train),
                "validation_rows": len(modelling_data.X_validation),
                "selected_feature_count": len(modelling_data.feature_columns),
                "numeric_feature_count": len(modelling_data.numeric_columns),
                "categorical_feature_count": len(modelling_data.categorical_columns),
                "review_capacity": evaluation_config["review_capacity"],
            }
        )

        print("Fitting XGBoost on chronological training data...")
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
        prediction_latency_ms_per_row = (
            total_prediction_time_seconds / len(modelling_data.X_validation)
        ) * 1000

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
        metrics["model_version"] = model_config["model_version"]
        metrics["dataset_version"] = reproducibility_config["dataset_version"]
        metrics["feature_version"] = reproducibility_config["feature_version"]
        metrics["target_encoding_version"] = reproducibility_config["target_encoding_version"]
        metrics["test_split_loaded"] = False

        mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items() if isinstance(value, int | float)}
        )

        save_validation_metrics(metrics)

        mlflow.log_dict(
            {
                "selected_feature_columns": modelling_data.feature_columns,
                "numeric_columns": modelling_data.numeric_columns,
                "categorical_columns": modelling_data.categorical_columns,
            },
            "feature_contract.json",
        )

        mlflow.sklearn.log_model(
            sk_model=preprocessor,
            name="xgboost_preprocessor",
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype"],
        )

        mlflow.xgboost.log_model(
            xgb_model=classifier,
            name="xgboost_classifier",
            model_format="json",
        )

        print_validation_results(metrics)
        print(f"\nMLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
