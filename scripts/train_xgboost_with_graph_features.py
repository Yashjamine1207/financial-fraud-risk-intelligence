"""Train and evaluate XGBoost with leakage-safe graph features.

Phase 6 graph-feature ablation experiment:

Baseline:
    Phase 4 XGBoost using the existing point-in-time feature contract.

Candidate:
    The same XGBoost configuration and preprocessing pipeline, plus ten
    leakage-safe graph features built from historical card-device and
    card-recipient-email relationships.

Leakage controls:
- The modelling data contract loads train and validation only.
- The final test split is not loaded by this script.
- Training graph features use only earlier training timestamps.
- Validation graph features begin with training graph history and use only
  earlier validation timestamps.
- All same-timestamp transactions are scored before graph state is updated.
- Missing entity values do not create shared graph nodes.
- Fraud labels are never used in graph construction or graph features.
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

from fraud_intelligence.features.graph_features import GRAPH_FEATURE_COLUMNS
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
GRAPH_FEATURE_CONFIG_PATH = Path("configs/graph_features.yaml")

GRAPH_DIRECTORY = Path("data/features/graph")
METRICS_OUTPUT_PATH = Path("reports/tables/xgboost_graph_validation_metrics.json")
FEATURE_CONTRACT_OUTPUT_PATH = Path("reports/tables/xgboost_graph_feature_contract.json")

EXPERIMENT_NAME = "financial-fraud-risk-intelligence"

CANDIDATE_MODEL_VERSION = "xgboost-plus-graph-v1.0.0"
GRAPH_FEATURE_VERSION = "graph-features-v1.0.0"


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and validate one YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Configuration file must parse to a dictionary: {path}")

    return config


def load_xgboost_config() -> dict[str, Any]:
    """Load and validate the existing XGBoost YAML configuration."""
    config = load_yaml_config(XGBOOST_CONFIG_PATH)

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


def load_graph_features(
    filename: str,
    expected_rows: int,
) -> pd.DataFrame:
    """Load, validate, and return only the ten graph-feature columns."""
    path = GRAPH_DIRECTORY / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Graph feature file was not found: {path}. "
            "Run scripts/generate_graph_features.py first."
        )

    graph_dataframe = pd.read_parquet(path)

    missing_columns = [
        column for column in GRAPH_FEATURE_COLUMNS if column not in graph_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Graph feature file is missing required columns: {missing_columns}. "
            f"File: {path}"
        )

    if len(graph_dataframe) != expected_rows:
        raise ValueError(
            f"Graph feature row count does not match the modelling data for {path}. "
            f"Expected {expected_rows:,}, found {len(graph_dataframe):,}."
        )

    graph_features = graph_dataframe[GRAPH_FEATURE_COLUMNS].copy()

    duplicate_columns = graph_features.columns[graph_features.columns.duplicated()].tolist()

    if duplicate_columns:
        raise ValueError(
            f"Graph feature file contains duplicate feature columns: {duplicate_columns}"
        )

    numeric_values = graph_features.to_numpy(dtype=float)

    if np.isinf(numeric_values).any():
        raise ValueError(f"Graph feature file contains infinite values: {path}")

    return graph_features


def append_graph_features(
    X: pd.DataFrame,
    graph_features: pd.DataFrame,
) -> pd.DataFrame:
    """Append validated graph features without changing row order."""
    if len(X) != len(graph_features):
        raise ValueError(
            "Feature matrix and graph-feature matrix must contain the same number of rows."
        )

    overlapping_columns = sorted(set(X.columns).intersection(graph_features.columns))

    if overlapping_columns:
        raise ValueError(
            "Graph features already exist in the modelling matrix. "
            f"Do not append them twice: {overlapping_columns}"
        )

    return pd.concat(
        [
            X.reset_index(drop=True),
            graph_features.reset_index(drop=True),
        ],
        axis=1,
    )


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
    """Print the main Phase 6 graph-ablation validation metrics."""
    print("\nXGBoost + graph features validation results")
    print("-" * 56)
    print(f"PR-AUC: {metrics['pr_auc']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(
        "Precision at reference threshold: "
        f"{metrics['precision_at_reference_threshold']:.6f}"
    )
    print(
        "Recall at reference threshold: "
        f"{metrics['recall_at_reference_threshold']:.6f}"
    )
    print(f"F1 at reference threshold: {metrics['f1_at_reference_threshold']:.6f}")
    print(f"Precision@{metrics['review_capacity']:,}: {metrics['precision_at_k']:.6f}")
    print(f"Recall@{metrics['review_capacity']:,}: {metrics['recall_at_k']:.6f}")
    print(
        "Captured fraud at review capacity: "
        f"{metrics['captured_fraud_count_at_k']:,} / "
        f"{metrics['total_fraud_count']:,}"
    )
    print(
        "Prediction latency per row: "
        f"{metrics['prediction_latency_ms_per_row']:.6f} ms"
    )
    print(f"Training time: {metrics['training_time_seconds']:.2f} seconds")
    print(f"Best boosting iteration: {metrics['best_iteration']}")
    print(f"Graph feature count: {metrics['graph_feature_count']}")
    print(f"Training graph feature rows with any missing value: "
          f"{metrics['train_graph_rows_with_missing_values']:,}")
    print("Final test split loaded: False")


def main() -> None:
    """Run the Phase 6 XGBoost-plus-graph-feature ablation on validation only."""
    load_dotenv()

    modeling_config = load_modeling_config()
    xgboost_config = load_xgboost_config()
    graph_config = load_yaml_config(GRAPH_FEATURE_CONFIG_PATH)
    modelling_data = load_modelling_data()

    if graph_config["data"].get("final_test_locked") is not True:
        raise RuntimeError(
            "Final-test protection is disabled in configs/graph_features.yaml. "
            "Set data.final_test_locked to true before training."
        )

    model_config = xgboost_config["model"]
    evaluation_config = modeling_config["evaluation"]
    reproducibility_config = modeling_config["reproducibility"]

    configure_mlflow()

    # The shared Phase 4 contract already excludes TransactionID,
    # TransactionDT, isFraud, and high-missingness columns.
    X_train_base = modelling_data.X_train.copy()
    X_validation_base = modelling_data.X_validation.copy()

    train_graph_features = load_graph_features(
        filename="train_graph_features.parquet",
        expected_rows=len(X_train_base),
    )
    validation_graph_features = load_graph_features(
        filename="validation_graph_features.parquet",
        expected_rows=len(X_validation_base),
    )

    train_graph_rows_with_missing_values = int(
        train_graph_features.isna().any(axis=1).sum()
    )
    validation_graph_rows_with_missing_values = int(
        validation_graph_features.isna().any(axis=1).sum()
    )

    X_train = append_graph_features(
        X=X_train_base,
        graph_features=train_graph_features,
    )
    X_validation = append_graph_features(
        X=X_validation_base,
        graph_features=validation_graph_features,
    )

    augmented_feature_columns = [
        *modelling_data.feature_columns,
        *GRAPH_FEATURE_COLUMNS,
    ]
    augmented_numeric_columns = [
        *modelling_data.numeric_columns,
        *GRAPH_FEATURE_COLUMNS,
    ]
    augmented_categorical_columns = list(modelling_data.categorical_columns)

    scale_pos_weight = calculate_scale_pos_weight(modelling_data.y_train)

    # Reuse the exact Phase 4 preprocessing construction, with the graph
    # features treated as numeric values. Median imputation and missingness
    # indicators are fitted solely on the chronological training period.
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

    classifier = build_xgboost_classifier(
        model_config=model_config,
        scale_pos_weight=scale_pos_weight,
    )

    run_name = "xgboost-plus-time-safe-graph-features-v1.0.0"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project_name": modeling_config["project"]["name"],
                "phase": "6",
                "experiment_type": "controlled_graph_feature_ablation",
                "model_name": model_config["name"],
                "model_version": CANDIDATE_MODEL_VERSION,
                "base_model_version": model_config["model_version"],
                "graph_feature_version": GRAPH_FEATURE_VERSION,
                "graph_feature_count": str(len(GRAPH_FEATURE_COLUMNS)),
                "graph_type": graph_config["graph"]["graph_type"],
                "graph_primary_entity": graph_config["graph"]["primary_entity"],
                "graph_relationship_entities": ",".join(
                    graph_config["graph"]["relationship_entities"]
                ),
                "same_timestamp_policy": graph_config["graph"][
                    "same_timestamp_policy"
                ],
                "fraud_labels_used_by_graph_features": "false",
                "dataset_version": reproducibility_config["dataset_version"],
                "feature_version": reproducibility_config["feature_version"],
                "target_encoding_version": reproducibility_config[
                    "target_encoding_version"
                ],
                "validation_strategy": "chronological_train_validation",
                "training_graph_protocol": "prior_timestamp_history_only",
                "validation_graph_protocol": (
                    "training_history_plus_prior_validation_timestamp_history_only"
                ),
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
                "graph_feature_count": len(GRAPH_FEATURE_COLUMNS),
                "review_capacity": evaluation_config["review_capacity"],
                "reference_probability_threshold": evaluation_config[
                    "reference_probability_threshold"
                ],
                "train_graph_rows_with_missing_values": (
                    train_graph_rows_with_missing_values
                ),
                "validation_graph_rows_with_missing_values": (
                    validation_graph_rows_with_missing_values
                ),
            }
        )

        print("Fitting XGBoost + graph features on chronological training data...")
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

        validation_probabilities = classifier.predict_proba(
            X_validation_transformed
        )[:, 1]

        total_prediction_time_seconds = time.perf_counter() - prediction_start_time

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
        metrics["model_version"] = CANDIDATE_MODEL_VERSION
        metrics["base_model_version"] = model_config["model_version"]
        metrics["graph_feature_version"] = GRAPH_FEATURE_VERSION
        metrics["graph_feature_count"] = len(GRAPH_FEATURE_COLUMNS)
        metrics["graph_feature_columns"] = GRAPH_FEATURE_COLUMNS
        metrics["train_graph_rows_with_missing_values"] = (
            train_graph_rows_with_missing_values
        )
        metrics["validation_graph_rows_with_missing_values"] = (
            validation_graph_rows_with_missing_values
        )
        metrics["dataset_version"] = reproducibility_config["dataset_version"]
        metrics["feature_version"] = reproducibility_config["feature_version"]
        metrics["target_encoding_version"] = reproducibility_config[
            "target_encoding_version"
        ]
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
            "added_features": GRAPH_FEATURE_COLUMNS,
            "added_feature_count": len(GRAPH_FEATURE_COLUMNS),
            "added_feature_version": GRAPH_FEATURE_VERSION,
            "training_graph_protocol": "prior_timestamp_history_only",
            "validation_graph_protocol": (
                "training_history_plus_prior_validation_timestamp_history_only"
            ),
            "same_timestamp_policy": graph_config["graph"][
                "same_timestamp_policy"
            ],
            "fraud_labels_used_by_graph_features": False,
            "test_split_loaded": False,
        }

        save_json(metrics, METRICS_OUTPUT_PATH)
        save_json(feature_contract, FEATURE_CONTRACT_OUTPUT_PATH)

        mlflow.log_dict(feature_contract, "graph_feature_contract.json")

        # Candidate-specific artifact names prevent overwriting or confusing
        # baseline and anomaly-ablation artifacts.
        mlflow.sklearn.log_model(
            sk_model=preprocessor,
            name="xgboost_graph_preprocessor",
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype"],
        )

        mlflow.xgboost.log_model(
            xgb_model=classifier,
            name="xgboost_graph_classifier",
            model_format="json",
        )

        print_validation_results(metrics)
        print(f"\nMLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()