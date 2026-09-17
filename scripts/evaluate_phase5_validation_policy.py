"""Evaluate the Phase 5 calibrated fraud-decision policy on validation data.

This workflow:
- Loads the frozen Phase 4 XGBoost model and its fitted preprocessor.
- Fits the selected sigmoid calibration mapping on the earlier validation period.
- Tunes and evaluates the review policy only on the later validation period.
- Uses a capacity-constrained top-k review policy.
- Does not load, fit on, inspect, or score the locked final test split.

Important:
The earlier calibration-fit validation period is never used to evaluate the
selected review policy. This prevents policy metrics from being calculated on
the same labels used to fit the calibration mapping.
"""

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
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from fraud_intelligence.decisioning.cost_model import CostMatrix
from fraud_intelligence.decisioning.threshold_policy import (
    evaluate_policy_at_capacity,
)
from fraud_intelligence.models.calibration import ProbabilityCalibrator
from fraud_intelligence.models.data_contract import load_modelling_data
from fraud_intelligence.models.model_loading import (
    load_calibration_config,
    load_frozen_xgboost_components,
)

PHASE5_POLICY_OUTPUT_PATH = Path(
    "models/metrics/phase5_validation_policy_evaluation.json"
)
PHASE5_POLICY_FIGURE_PATH = Path(
    "reports/figures/phase5_validation_policy_evaluation.png"
)


def load_cost_matrix(config: dict[str, Any]) -> CostMatrix:
    """Create the Phase 5 cost matrix from versioned YAML assumptions."""
    decision_policy = config["decision_policy"]

    return CostMatrix(
        false_negative_cost=float(decision_policy["false_negative_cost"]),
        manual_review_cost=float(decision_policy["manual_review_cost"]),
        false_positive_escalation_cost=float(
            decision_policy["false_positive_escalation_cost"]
        ),
        fraud_prevention_value=float(
            decision_policy["fraud_prevention_value"]
        ),
    )


def create_chronological_split_masks(
    timestamps: np.ndarray,
    calibration_fit_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split validation rows at a timestamp boundary without splitting ties."""
    timestamp_values = np.asarray(timestamps, dtype=np.float64)

    if timestamp_values.ndim != 1:
        raise ValueError("Validation timestamps must be one-dimensional.")

    if timestamp_values.size < 2:
        raise ValueError(
            "At least two validation rows are required for chronological splitting."
        )

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

    # Move to the first row for this timestamp so all same-second rows remain
    # entirely in the later policy-selection period.
    split_index = int(
        np.searchsorted(
            timestamp_values,
            requested_split_timestamp,
            side="left",
        )
    )

    if split_index == 0 or split_index >= timestamp_values.size:
        raise RuntimeError(
            "Unable to create non-empty calibration-fit and policy-selection periods."
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
    """Confirm a labelled period contains both fraud and non-fraud outcomes."""
    unique_labels = np.unique(labels)

    if unique_labels.size != 2 or not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(
            f"{period_name} must contain both fraud and non-fraud labels."
        )


def create_policy_figure(
    evaluation_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Create capacity-vs-capture and capacity-vs-value policy plots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capacities = [result["review_capacity"] for result in evaluation_results]
    capture_rates = [
        result["fraud_capture_rate"] for result in evaluation_results
    ]
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
        color="#0077b6",
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
        color="#d62828",
        linewidth=2.0,
    )
    value_axis.set_title("Net expected value vs review capacity")
    value_axis.set_xlabel("Review capacity")
    value_axis.set_ylabel("Net expected value (GBP)")
    value_axis.grid(alpha=0.25)

    figure.suptitle(
        "Phase 5 Validation Policy Selection — Sigmoid-Calibrated XGBoost"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the lightweight, Git-trackable policy evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Saved Phase 5 policy report to: {output_path}")


def main() -> None:
    """Evaluate capacity-constrained policy on the later validation period."""
    load_dotenv()

    config = load_calibration_config()

    if config["data"].get("final_test_locked") is not True:
        raise RuntimeError(
            "Final test protection is disabled. "
            "Set data.final_test_locked to true before running Phase 5."
        )

    modelling_data = load_modelling_data()
    frozen_components = load_frozen_xgboost_components()
    cost_matrix = load_cost_matrix(config)

    validation_feature_path = Path(
        config["data"]["validation_feature_path"]
    )
    timestamp_column = config["data"]["timestamp_column"]

    validation_timestamp_dataframe = pd.read_parquet(
        validation_feature_path,
        columns=[timestamp_column],
    )

    if len(validation_timestamp_dataframe) != len(
        modelling_data.X_validation
    ):
        raise RuntimeError(
            "Validation timestamps and validation model features have "
            "different row counts."
        )

    validation_timestamps = validation_timestamp_dataframe[
        timestamp_column
    ].to_numpy()

    validation_labels = np.asarray(
        modelling_data.y_validation,
        dtype=np.int64,
    )

    print("Loading frozen Phase 4 XGBoost components from MLflow...")
    print(f"Model version: {frozen_components.model_version}")
    print(f"MLflow run ID: {frozen_components.mlflow_run_id}")
    print("Final test split loaded: False")

    print("\nTransforming chronological validation features...")
    transformed_validation_features = (
        frozen_components.preprocessor.transform(
            modelling_data.X_validation
        )
    )

    print("Scoring chronological validation data with the frozen model...")
    base_validation_probabilities = (
        frozen_components.classifier.predict_proba(
            transformed_validation_features
        )[:, 1]
    )

    calibration_fit_fraction = float(
        config["validation_protocol"]["calibration_fit_fraction"]
    )

    (
        calibration_fit_mask,
        policy_selection_mask,
        split_timestamp,
    ) = create_chronological_split_masks(
        timestamps=validation_timestamps,
        calibration_fit_fraction=calibration_fit_fraction,
    )

    calibration_fit_probabilities = base_validation_probabilities[
        calibration_fit_mask
    ]
    calibration_fit_labels = validation_labels[calibration_fit_mask]

    policy_base_probabilities = base_validation_probabilities[
        policy_selection_mask
    ]
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
    sigmoid_calibrator = ProbabilityCalibrator(method="sigmoid")
    sigmoid_calibrator.fit(
        base_probabilities=calibration_fit_probabilities,
        labels=calibration_fit_labels,
    )

    print(
        "Applying frozen sigmoid mapping to later policy-selection data..."
    )
    policy_probabilities = sigmoid_calibrator.predict(
        policy_base_probabilities
    )

    decision_policy_config = config["decision_policy"]

    # The policy evaluates several capacities for sensitivity analysis.
    # The documented operational capacity remains 1,000 reviews per day.
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
                f"{config['project']['threshold_policy_version']}"
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
            f"net value £{evaluation.net_expected_value:,.0f}"
        )

    configured_review_capacity = int(
        decision_policy_config["daily_review_capacity"]
    )

    selected_capacity_result = next(
        result
        for result in evaluation_results
        if result["review_capacity"] == configured_review_capacity
    )

    report = {
        "phase": 5,
        "purpose": "validation_policy_selection",
        "champion_model": {
            "model_name": frozen_components.model_name,
            "model_version": frozen_components.model_version,
            "mlflow_run_id": frozen_components.mlflow_run_id,
            "preprocessor_model_uri": (
                frozen_components.preprocessor_model_uri
            ),
            "classifier_model_uri": frozen_components.classifier_model_uri,
        },
        "calibration": {
            "method": "sigmoid",
            "calibration_fit_period": (
                config["validation_protocol"]["calibration_fit_period"]
            ),
            "policy_selection_period": (
                config["validation_protocol"][
                    "calibration_selection_period"
                ]
            ),
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
            "policy_selection_fraud_count": int(
                policy_selection_labels.sum()
            ),
        },
        "cost_assumptions": asdict(cost_matrix),
        "configured_operational_capacity": configured_review_capacity,
        "selected_capacity_result": selected_capacity_result,
        "capacity_sensitivity_results": evaluation_results,
    }

    create_policy_figure(
        evaluation_results=evaluation_results,
        output_path=PHASE5_POLICY_FIGURE_PATH,
    )
    print(f"Saved Phase 5 policy figure to: {PHASE5_POLICY_FIGURE_PATH}")

    save_json_report(
        report=report,
        output_path=PHASE5_POLICY_OUTPUT_PATH,
    )

    tracking_uri_environment_variable = config["champion_model"][
        "tracking_uri_env_var"
    ]
    tracking_uri = os.getenv(tracking_uri_environment_variable)

    if not tracking_uri:
        raise RuntimeError(
            f"Environment variable '{tracking_uri_environment_variable}' "
            "is not set."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(
        run_name="phase5-validation-policy-selection"
    ) as run:
        mlflow.set_tags(config["mlflow"]["tags"])
        mlflow.set_tag(
            "frozen_phase4_xgboost_run_id",
            frozen_components.mlflow_run_id,
        )
        mlflow.set_tag("calibration_method", "sigmoid")
        mlflow.set_tag("final_test_loaded", "false")
        mlflow.set_tag(
            "policy_selection_period",
            config["validation_protocol"][
                "calibration_selection_period"
            ],
        )

        mlflow.log_metrics(
            {
                "calibration_fit_rows": float(
                    calibration_fit_mask.sum()
                ),
                "policy_selection_rows": float(
                    policy_selection_mask.sum()
                ),
                "configured_operational_capacity": float(
                    configured_review_capacity
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

        mlflow.log_artifact(PHASE5_POLICY_OUTPUT_PATH)
        mlflow.log_artifact(PHASE5_POLICY_FIGURE_PATH)

        print(f"\nPhase 5 MLflow run ID: {run.info.run_id}")

    print("\nPhase 5 validation policy selection complete.")
    print("Final test split loaded: False")


if __name__ == "__main__":
    main()