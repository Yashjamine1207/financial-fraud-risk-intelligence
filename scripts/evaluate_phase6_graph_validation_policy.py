"""Evaluate the calibrated Phase 6 graph candidate on validation only."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from fraud_intelligence.decisioning.cost_model import CostMatrix
from fraud_intelligence.decisioning.threshold_policy import (
    evaluate_policy_at_capacity,
)
from fraud_intelligence.features.graph_features import GRAPH_FEATURE_COLUMNS
from fraud_intelligence.models.calibration import ProbabilityCalibrator
from fraud_intelligence.models.data_contract import load_modelling_data

CALIBRATION_CONFIG_PATH = Path("configs/calibration.yaml")
GRAPH_FEATURE_CONFIG_PATH = Path("configs/graph_features.yaml")
GRAPH_DIRECTORY = Path("data/features/graph")
GRAPH_METRICS_PATH = Path("reports/tables/xgboost_graph_validation_metrics.json")

POLICY_OUTPUT_PATH = Path(
    "models/metrics/phase6_graph_validation_policy_evaluation.json"
)
POLICY_FIGURE_PATH = Path(
    "reports/figures/phase6_graph_validation_policy_evaluation.png"
)

CANDIDATE_MODEL_VERSION = "xgboost-plus-graph-v1.0.0"
GRAPH_FEATURE_VERSION = "graph-features-v1.0.0"
CALIBRATION_METHOD = "sigmoid"


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and validate one YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Configuration file must parse to a dictionary: {path}")

    return config


def load_json_report(path: Path) -> dict[str, Any]:
    """Load and validate one JSON report."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required report was not found: {path}. "
            "Run scripts/train_xgboost_with_graph_features.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise TypeError(f"JSON report must contain an object: {path}")

    return report


def load_cost_matrix(config: dict[str, Any]) -> CostMatrix:
    """Create the existing Phase 5 cost matrix from calibration.yaml."""
    decision_policy = config["decision_policy"]

    return CostMatrix(
        false_negative_cost=float(decision_policy["false_negative_cost"]),
        manual_review_cost=float(decision_policy["manual_review_cost"]),
        false_positive_escalation_cost=float(
            decision_policy["false_positive_escalation_cost"]
        ),
        fraud_prevention_value=float(decision_policy["fraud_prevention_value"]),
    )


def load_graph_features(
    filename: str,
    expected_rows: int,
) -> pd.DataFrame:
    """Load one graph-feature dataset and validate its contract."""
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

    graph_features = graph_dataframe[GRAPH_FEATURE_COLUMNS].copy()

    if len(graph_features) != expected_rows:
        raise ValueError(
            f"Unexpected row count in {path}. Expected {expected_rows:,}, "
            f"found {len(graph_features):,}."
        )

    if np.isinf(graph_features.to_numpy(dtype=float)).any():
        raise ValueError(f"Graph feature file contains infinite values: {path}")

    return graph_features


def append_graph_features(
    X: pd.DataFrame,
    graph_features: pd.DataFrame,
) -> pd.DataFrame:
    """Append graph features without changing row order."""
    if len(X) != len(graph_features):
        raise ValueError(
            "Feature matrix and graph feature matrix must have the same number of rows."
        )

    overlapping_columns = sorted(set(X.columns).intersection(graph_features.columns))

    if overlapping_columns:
        raise ValueError(
            "Feature matrix already contains graph columns. "
            f"Do not append graph features twice: {overlapping_columns}"
        )

    return pd.concat(
        [
            X.reset_index(drop=True),
            graph_features.reset_index(drop=True),
        ],
        axis=1,
    )


def create_chronological_split_masks(
    timestamps: np.ndarray,
    calibration_fit_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split validation at a timestamp boundary without splitting ties."""
    timestamp_values = np.asarray(timestamps, dtype=np.float64)

    if timestamp_values.ndim != 1:
        raise ValueError("Validation timestamps must be one-dimensional.")

    if timestamp_values.size < 2:
        raise ValueError("At least two validation rows are required.")

    if not np.isfinite(timestamp_values).all():
        raise ValueError("Validation timestamps must contain only finite values.")

    if np.any(np.diff(timestamp_values) < 0):
        raise ValueError(
            "Validation timestamps must be sorted in ascending chronological order."
        )

    if not 0.0 < calibration_fit_fraction < 1.0:
        raise ValueError(
            "calibration_fit_fraction must be strictly between 0 and 1."
        )

    requested_split_index = int(
        np.floor(timestamp_values.size * calibration_fit_fraction)
    )
    requested_split_index = min(
        max(requested_split_index, 1),
        timestamp_values.size - 1,
    )

    requested_split_timestamp = timestamp_values[requested_split_index]

    split_index = int(
        np.searchsorted(
            timestamp_values,
            requested_split_timestamp,
            side="left",
        )
    )

    if split_index == 0 or split_index >= timestamp_values.size:
        raise RuntimeError(
            "Unable to create non-empty calibration-fit and policy-selection "
            "periods without splitting timestamp ties."
        )

    calibration_fit_mask = np.arange(timestamp_values.size) < split_index
    policy_selection_mask = ~calibration_fit_mask

    return (
        calibration_fit_mask,
        policy_selection_mask,
        float(requested_split_timestamp),
    )


def validate_binary_period(
    labels: np.ndarray,
    period_name: str,
) -> None:
    """Confirm that one evaluation period contains both outcome classes."""
    unique_labels = np.unique(labels)

    if unique_labels.size != 2 or not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(f"{period_name} must contain both fraud and non-fraud labels.")


def create_policy_figure(
    evaluation_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Create capacity-sensitivity charts for the graph candidate."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capacities = [result["review_capacity"] for result in evaluation_results]
    capture_rates = [result["fraud_capture_rate"] for result in evaluation_results]
    net_values = [result["net_expected_value"] for result in evaluation_results]

    figure, (capture_axis, value_axis) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    capture_axis.plot(
        capacities,
        capture_rates,
        marker="o",
        color="#0077B6",
        linewidth=2.0,
    )
    capture_axis.set_title("Fraud capture rate vs review capacity")
    capture_axis.set_xlabel("Review capacity")
    capture_axis.set_ylabel("Fraud capture rate")
    capture_axis.grid(alpha=0.25)

    value_axis.plot(
        capacities,
        net_values,
        marker="s",
        color="#D62828",
        linewidth=2.0,
    )
    value_axis.set_title("Net expected value vs review capacity")
    value_axis.set_xlabel("Review capacity")
    value_axis.set_ylabel("Net expected value (GBP)")
    value_axis.grid(alpha=0.25)

    figure.suptitle(
        "Phase 6 Validation Policy - Sigmoid-Calibrated XGBoost + Graph Features"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a lightweight, Git-trackable JSON policy report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Saved Phase 6 graph policy report to: {output_path}")


def main() -> None:
    """Evaluate the calibrated Phase 6 graph candidate on validation only."""
    load_dotenv()

    calibration_config = load_yaml_config(CALIBRATION_CONFIG_PATH)
    graph_config = load_yaml_config(GRAPH_FEATURE_CONFIG_PATH)

    if calibration_config["data"].get("final_test_locked") is not True:
        raise RuntimeError(
            "Final-test protection is disabled in calibration.yaml. "
            "Set data.final_test_locked to true before policy evaluation."
        )

    if graph_config["data"].get("final_test_locked") is not True:
        raise RuntimeError(
            "Final-test protection is disabled in graph_features.yaml. "
            "Set data.final_test_locked to true before policy evaluation."
        )

    graph_metrics = load_json_report(GRAPH_METRICS_PATH)

    if graph_metrics.get("test_split_loaded") is not False:
        raise RuntimeError(
            "The candidate training report indicates final-test access. "
            "Phase 6 policy evaluation requires validation-only candidate metrics."
        )

    candidate_run_id = graph_metrics.get("run_id")

    if not isinstance(candidate_run_id, str) or not candidate_run_id:
        raise ValueError(
            "The graph candidate metrics report does not contain a valid MLflow run_id."
        )

    tracking_uri_env_var = calibration_config["champion_model"]["tracking_uri_env_var"]
    tracking_uri = os.getenv(tracking_uri_env_var)

    if not tracking_uri:
        raise RuntimeError(f"Environment variable '{tracking_uri_env_var}' is not set.")

    mlflow.set_tracking_uri(tracking_uri)

    candidate_preprocessor_uri = (
        f"runs:/{candidate_run_id}/xgboost_graph_preprocessor"
    )
    candidate_classifier_uri = (
        f"runs:/{candidate_run_id}/xgboost_graph_classifier"
    )

    print("Loading Phase 6 XGBoost + graph components from MLflow...")
    print(f"Candidate model version: {CANDIDATE_MODEL_VERSION}")
    print(f"Candidate MLflow run ID: {candidate_run_id}")
    print("Final test split loaded: False")

    candidate_preprocessor = mlflow.sklearn.load_model(candidate_preprocessor_uri)
    candidate_classifier = mlflow.xgboost.load_model(candidate_classifier_uri)

    modelling_data = load_modelling_data()

    train_graph_features = load_graph_features(
        filename="train_graph_features.parquet",
        expected_rows=len(modelling_data.X_train),
    )
    validation_graph_features = load_graph_features(
        filename="validation_graph_features.parquet",
        expected_rows=len(modelling_data.X_validation),
    )

    train_graph_rows_with_missing_values = int(
        train_graph_features.isna().any(axis=1).sum()
    )

    X_validation = append_graph_features(
        X=modelling_data.X_validation,
        graph_features=validation_graph_features,
    )

    validation_feature_path = Path(
        calibration_config["data"]["validation_feature_path"]
    )
    timestamp_column = calibration_config["data"]["timestamp_column"]

    validation_timestamp_dataframe = pd.read_parquet(
        validation_feature_path,
        columns=[timestamp_column],
    )

    if len(validation_timestamp_dataframe) != len(X_validation):
        raise RuntimeError(
            "Validation timestamp rows and validation graph feature rows do not match."
        )

    validation_timestamps = validation_timestamp_dataframe[
        timestamp_column
    ].to_numpy()

    validation_labels = np.asarray(
        modelling_data.y_validation,
        dtype=np.int64,
    )

    print("\nTransforming chronological validation features...")
    transformed_validation_features = candidate_preprocessor.transform(X_validation)

    print("Scoring chronological validation data with XGBoost + graph features...")
    validation_probabilities = candidate_classifier.predict_proba(
        transformed_validation_features
    )[:, 1]

    calibration_fit_fraction = float(
        calibration_config["validation_protocol"]["calibration_fit_fraction"]
    )

    (
        calibration_fit_mask,
        policy_selection_mask,
        split_timestamp,
    ) = create_chronological_split_masks(
        timestamps=validation_timestamps,
        calibration_fit_fraction=calibration_fit_fraction,
    )

    calibration_fit_probabilities = validation_probabilities[calibration_fit_mask]
    calibration_fit_labels = validation_labels[calibration_fit_mask]

    policy_base_probabilities = validation_probabilities[policy_selection_mask]
    policy_selection_labels = validation_labels[policy_selection_mask]

    validate_binary_period(
        labels=calibration_fit_labels,
        period_name="Calibration-fit period",
    )
    validate_binary_period(
        labels=policy_selection_labels,
        period_name="Policy-selection period",
    )

    print("\nChronological validation split")
    print(f"Calibration-fit rows: {calibration_fit_mask.sum():,}")
    print(f"Policy-selection rows: {policy_selection_mask.sum():,}")
    print(f"Split timestamp: {split_timestamp:.0f}")

    print("\nFitting sigmoid calibration on earlier validation data...")
    sigmoid_calibrator = ProbabilityCalibrator(method=CALIBRATION_METHOD)
    sigmoid_calibrator.fit(
        base_probabilities=calibration_fit_probabilities,
        labels=calibration_fit_labels,
    )

    print("Applying sigmoid calibration to later policy-selection data...")
    policy_probabilities = sigmoid_calibrator.predict(policy_base_probabilities)

    cost_matrix = load_cost_matrix(calibration_config)

    review_capacities = [500, 1000, 1500, 2000]
    evaluation_results: list[dict[str, Any]] = []

    print("\nEvaluating capacity-constrained review policy...")

    for review_capacity in review_capacities:
        evaluation = evaluate_policy_at_capacity(
            fraud_probabilities=policy_probabilities,
            labels=policy_selection_labels,
            review_capacity=review_capacity,
            min_probability=0.0,
            max_probability=1.0,
            cost_matrix=cost_matrix,
            action_labels=("approve", "review", "block"),
            policy_version=(
                "phase6-graph-sigmoid-policy-v1.0.0"
                f"-capacity-{review_capacity}"
            ),
        )

        evaluation_result = asdict(evaluation)
        evaluation_results.append(evaluation_result)

        print(
            f"Capacity {review_capacity:>4}: "
            f"captured {evaluation.captured_fraud_count:>4} / "
            f"{evaluation.total_fraud_count} fraud, "
            f"capture rate {evaluation.fraud_capture_rate:.2%}, "
            f"net value GBP {evaluation.net_expected_value:,.0f}"
        )

    configured_review_capacity = int(
        calibration_config["decision_policy"]["daily_review_capacity"]
    )

    selected_capacity_result = next(
        result
        for result in evaluation_results
        if result["review_capacity"] == configured_review_capacity
    )

    report = {
        "phase": 6,
        "purpose": "graph_ablation_validation_policy_evaluation",
        "candidate_model": {
            "model_name": "xgboost_plus_time_safe_graph_features",
            "model_version": CANDIDATE_MODEL_VERSION,
            "mlflow_run_id": candidate_run_id,
            "preprocessor_model_uri": candidate_preprocessor_uri,
            "classifier_model_uri": candidate_classifier_uri,
        },
        "graph_features": {
            "feature_version": GRAPH_FEATURE_VERSION,
            "feature_count": len(GRAPH_FEATURE_COLUMNS),
            "feature_columns": GRAPH_FEATURE_COLUMNS,
            "graph_type": graph_config["graph"]["graph_type"],
            "primary_entity": graph_config["graph"]["primary_entity"],
            "relationship_entities": graph_config["graph"][
                "relationship_entities"
            ],
            "training_protocol": "prior_timestamp_history_only",
            "validation_protocol": (
                "training_history_plus_prior_validation_timestamp_history_only"
            ),
            "same_timestamp_policy": graph_config["graph"][
                "same_timestamp_policy"
            ],
            "fraud_labels_used_by_graph_features": False,
            "training_rows_with_any_missing_graph_feature": (
                train_graph_rows_with_missing_values
            ),
        },
        "calibration": {
            "method": CALIBRATION_METHOD,
            "calibration_fit_period": calibration_config["validation_protocol"][
                "calibration_fit_period"
            ],
            "policy_selection_period": calibration_config["validation_protocol"][
                "calibration_selection_period"
            ],
        },
        "data_protection": {
            "final_test_locked": True,
            "final_test_loaded": False,
        },
        "chronological_split": {
            "split_timestamp": split_timestamp,
            "calibration_fit_rows": int(calibration_fit_mask.sum()),
            "policy_selection_rows": int(policy_selection_mask.sum()),
            "calibration_fit_fraud_count": int(calibration_fit_labels.sum()),
            "policy_selection_fraud_count": int(policy_selection_labels.sum()),
        },
        "cost_assumptions": asdict(cost_matrix),
        "configured_operational_capacity": configured_review_capacity,
        "selected_capacity_result": selected_capacity_result,
        "capacity_sensitivity_results": evaluation_results,
    }

    create_policy_figure(
        evaluation_results=evaluation_results,
        output_path=POLICY_FIGURE_PATH,
    )
    print(f"Saved Phase 6 graph policy figure to: {POLICY_FIGURE_PATH}")

    save_json_report(
        report=report,
        output_path=POLICY_OUTPUT_PATH,
    )

    mlflow.set_experiment("fraud-intelligence-phase-6-graph-ablation")

    with mlflow.start_run(
        run_name="phase6-graph-validation-policy-evaluation"
    ) as run:
        mlflow.set_tags(
            {
                "project_phase": "6",
                "experiment_type": "graph_ablation_policy_evaluation",
                "candidate_model_version": CANDIDATE_MODEL_VERSION,
                "candidate_training_run_id": candidate_run_id,
                "graph_feature_version": GRAPH_FEATURE_VERSION,
                "graph_feature_count": str(len(GRAPH_FEATURE_COLUMNS)),
                "calibration_method": CALIBRATION_METHOD,
                "validation_strategy": (
                    "chronological_calibration_fit_then_policy_selection"
                ),
                "final_test_loaded": "false",
            }
        )

        mlflow.log_metrics(
            {
                "calibration_fit_rows": float(calibration_fit_mask.sum()),
                "policy_selection_rows": float(policy_selection_mask.sum()),
                "configured_operational_capacity": float(
                    configured_review_capacity
                ),
                "train_graph_rows_with_missing_values": float(
                    train_graph_rows_with_missing_values
                ),
            }
        )

        for result in evaluation_results:
            capacity_label = str(result["review_capacity"])

            mlflow.log_metrics(
                {
                    f"capacity_{capacity_label}_captured_fraud": float(
                        result["captured_fraud_count"]
                    ),
                    f"capacity_{capacity_label}_capture_rate": float(
                        result["fraud_capture_rate"]
                    ),
                    f"capacity_{capacity_label}_net_expected_value": float(
                        result["net_expected_value"]
                    ),
                }
            )

        mlflow.log_artifact(POLICY_OUTPUT_PATH)
        mlflow.log_artifact(POLICY_FIGURE_PATH)

        print(f"\nPhase 6 graph policy MLflow run ID: {run.info.run_id}")

    print("\nPhase 6 graph validation policy evaluation complete.")
    print("Final test split loaded: False")


if __name__ == "__main__":
    main()