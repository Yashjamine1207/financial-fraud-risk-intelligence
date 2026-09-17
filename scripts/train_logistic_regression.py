"""Train and evaluate the Phase 4 Logistic Regression fraud baseline.

This script:
- Loads only Phase 3 train and validation feature tables.
- Fits preprocessing and Logistic Regression on training data only.
- Evaluates only on validation data.
- Does not load the locked final test split.
- Logs parameters, versions, metrics, and the model to MLflow.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import yaml
from dotenv import load_dotenv

from fraud_intelligence.models.baseline import (
    build_logistic_regression_pipeline,
)
from fraud_intelligence.models.data_contract import load_modeling_config, load_modelling_data
from fraud_intelligence.models.evaluation import evaluate_binary_classifier

LOGISTIC_CONFIG_PATH = Path("configs/logistic_regression.yaml")
METRICS_OUTPUT_PATH = Path("reports/tables/logistic_regression_validation_metrics.json")
EXPERIMENT_NAME = "financial-fraud-risk-intelligence"


def load_logistic_regression_config() -> dict[str, Any]:
    """Load the Logistic Regression configuration from YAML."""
    if not LOGISTIC_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Logistic Regression configuration was not found: " f"{LOGISTIC_CONFIG_PATH}"
        )

    with LOGISTIC_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Logistic Regression configuration must parse to a dictionary.")

    if "model" not in config:
        raise ValueError("Logistic Regression configuration is missing the 'model' section.")

    return config


def configure_mlflow() -> None:
    """Configure local or environment-defined MLflow tracking."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")


def save_validation_metrics(metrics: dict[str, Any]) -> None:
    """Save a small, versioned validation-metrics report for Phase 4."""
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved validation metrics to: {METRICS_OUTPUT_PATH}")


def print_validation_results(metrics: dict[str, Any]) -> None:
    """Print the key baseline results in a readable terminal format."""
    print("\nLogistic Regression validation results")
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
    print("Final test split loaded: False")


def main() -> None:
    """Train, validate, save, and track the baseline experiment."""
    load_dotenv()

    modeling_config = load_modeling_config()
    logistic_config = load_logistic_regression_config()
    modelling_data = load_modelling_data()

    evaluation_config = modeling_config["evaluation"]
    reproducibility_config = modeling_config["reproducibility"]
    model_config = logistic_config["model"]

    configure_mlflow()

    pipeline = build_logistic_regression_pipeline(
        numeric_columns=modelling_data.numeric_columns,
        categorical_columns=modelling_data.categorical_columns,
        preprocessing_config=modeling_config["preprocessing"],
        model_config=model_config,
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
                "test_split_loaded": "false",
            }
        )

        mlflow.log_params(
            {
                "C": model_config["C"],
                "l1_ratio": model_config["l1_ratio"],
                "solver": model_config["solver"],
                "max_iter": model_config["max_iter"],
                "tolerance": model_config["tolerance"],
                "class_weight": model_config["class_weight"],
                "random_state": model_config["random_state"],
                "train_rows": len(modelling_data.X_train),
                "validation_rows": len(modelling_data.X_validation),
                "selected_feature_count": len(modelling_data.feature_columns),
                "numeric_feature_count": len(modelling_data.numeric_columns),
                "categorical_feature_count": len(modelling_data.categorical_columns),
                "review_capacity": evaluation_config["review_capacity"],
            }
        )

        print("\nFitting Logistic Regression on chronological training data...")
        training_start_time = time.perf_counter()

        pipeline.fit(modelling_data.X_train, modelling_data.y_train)

        training_time_seconds = time.perf_counter() - training_start_time

        print("Scoring chronological validation data...")
        prediction_start_time = time.perf_counter()

        validation_probabilities = pipeline.predict_proba(modelling_data.X_validation)[:, 1]

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
        mlflow.log_dict(
            {
                "selected_feature_columns": modelling_data.feature_columns,
                "numeric_columns": modelling_data.numeric_columns,
                "categorical_columns": modelling_data.categorical_columns,
            },
            "feature_contract.json",
        )

        save_validation_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="logistic_regression_pipeline",
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype"],
        )

        print_validation_results(metrics)

        print(f"\nMLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
