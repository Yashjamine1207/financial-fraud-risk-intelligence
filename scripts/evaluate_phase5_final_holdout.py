"""One-time evaluation of the Phase 5 policy on the locked final test split.

This script:
- Loads the frozen Phase 4 XGBoost champion model.
- Applies the sigmoid calibration mapping fitted on the earlier validation period.
- Evaluates the locked top-k review policy at the locked operational capacity.
- Uses ONLY the final test split; it does not refit or retune anything.
- Creates one final holdout report and figure for the portfolio.

WARNING:
This workflow must be executed exactly once. After it runs, the final test
split must remain untouched and must not be used for further model selection,
threshold tuning, or policy tuning.
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
from fraud_intelligence.models.data_contract import load_modeling_config
from fraud_intelligence.models.model_loading import (
    load_calibration_config,
    load_frozen_xgboost_components,
)

PHASE5_HOLDOUT_OUTPUT_PATH = Path(
    "models/metrics/phase5_final_holdout_evaluation.json"
)
PHASE5_HOLDOUT_FIGURE_PATH = Path(
    "reports/figures/phase5_final_holdout_evaluation.png"
)


def load_cost_matrix(config: dict[str, Any]) -> CostMatrix:
    """Load the Phase 5 illustrative cost assumptions from configuration."""
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


def load_final_test_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load ONLY the final test feature table and its labels and timestamps.

    This function deliberately does not load train or validation data.
    """
    # Use the generated Phase 3 feature table, not the raw temporal split.
    test_path = Path(
        "data/features/point_in_time/test_behavioural_and_encoded_features.parquet"
    )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Final test feature table was not found: {test_path}. "
            "Run scripts/generate_final_test_features.py first."
        )

    if config["data"].get("final_test_locked") is not True:
        raise RuntimeError(
            "Final test protection is disabled. "
            "Set data.final_test_locked to true before running holdout evaluation."
        )

    target_column = config["data"]["target_column"]
    timestamp_column = config["data"]["timestamp_column"]

    test_dataframe = pd.read_parquet(test_path)

    X_test = test_dataframe.loc[:, ~test_dataframe.columns.isin([target_column, timestamp_column])].copy()
    y_test = test_dataframe[target_column].astype("int8").copy()
    test_timestamps = test_dataframe[timestamp_column].to_numpy()

    return X_test, y_test, test_timestamps


def create_holdout_figure(
    evaluation_result: dict[str, Any],
    output_path: Path,
) -> None:
    """Create a simple holdout policy summary figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))

    capacity = evaluation_result["review_capacity"]
    capture_rate = evaluation_result["fraud_capture_rate"]
    net_value = evaluation_result["net_expected_value"]

    axis.bar(
        ["Fraud capture rate", "Net value (scaled)"],
        [capture_rate, net_value / 1e6],
        color=["#0077b6", "#d62828"],
    )

    axis.set_title(
        f"Phase 5 Final Holdout Policy Result\n"
        f"Capacity = {capacity:,} reviews, "
        f"Capture rate = {capture_rate:.2%}, "
        f"Net value = £{net_value:,.0f}"
    )
    axis.set_ylabel("Value")
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the final holdout evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Saved Phase 5 holdout report to: {output_path}")


def main() -> None:
    """Run the one-time Phase 5 final holdout policy evaluation."""
    load_dotenv()

    config = load_calibration_config()

    if config["data"].get("final_test_locked") is not True:
        raise RuntimeError(
            "Final test protection is disabled. "
            "Set data.final_test_locked to true before running holdout evaluation."
        )

    frozen_components = load_frozen_xgboost_components()
    cost_matrix = load_cost_matrix(config)

    print("Loading final test data...")
    X_test, y_test, _test_timestamps = load_final_test_data(config)

    print("Loading frozen Phase 4 XGBoost components from MLflow...")
    print(f"Model version: {frozen_components.model_version}")
    print(f"MLflow run ID: {frozen_components.mlflow_run_id}")

    print("\nTransforming final test features...")
    X_test_transformed = frozen_components.preprocessor.transform(X_test)

    print("Scoring final test data with the frozen model...")
    base_test_probabilities = (
        frozen_components.classifier.predict_proba(X_test_transformed)[:, 1]
    )

    # Re-fit sigmoid calibration on the earlier validation period only.
    # This reproduces exactly the calibration used in validation policy selection.
    modeling_config = load_modeling_config()
    validation_path = Path(modeling_config["data"]["validation_path"])
    validation_dataframe = pd.read_parquet(validation_path)

    validation_timestamps = validation_dataframe[
        modeling_config["data"]["timestamp_column"]
    ].to_numpy()

    calibration_fit_fraction = float(
        config["validation_protocol"]["calibration_fit_fraction"]
    )
    split_index = int(np.floor(len(validation_timestamps) * calibration_fit_fraction))

    calibration_fit_labels = validation_dataframe[
        modeling_config["data"]["target_column"]
    ].iloc[:split_index].to_numpy()

    # Re-score validation fit period to obtain calibration probabilities.
    X_validation_fit = validation_dataframe.iloc[:split_index].loc[:, X_test.columns]
    X_validation_fit_transformed = frozen_components.preprocessor.transform(X_validation_fit)

    calibration_fit_probabilities = (
        frozen_components.classifier.predict_proba(X_validation_fit_transformed)[:, 1]
    )

    print("\nFitting sigmoid calibration on earlier validation data...")
    sigmoid_calibrator = ProbabilityCalibrator(method="sigmoid")
    sigmoid_calibrator.fit(calibration_fit_probabilities, calibration_fit_labels)

    print("Applying frozen sigmoid mapping to final test data...")
    calibrated_test_probabilities = sigmoid_calibrator.predict(base_test_probabilities)

    configured_capacity = int(config["project"]["configured_operational_capacity"])

    print(f"\nEvaluating locked policy at capacity = {configured_capacity:,}...")
    evaluation = evaluate_policy_at_capacity(
        fraud_probabilities=calibrated_test_probabilities,
        labels=y_test.to_numpy(),
        review_capacity=configured_capacity,
        min_probability=0.0,
        max_probability=1.0,
        cost_matrix=cost_matrix,
        action_labels=("approve", "review", "block"),
        policy_version=f"{config['project']['threshold_policy_version']}-holdout",
    )

    evaluation_result = asdict(evaluation)

    print(
        f"Final holdout result:\n"
        f"  Captured fraud: {evaluation.captured_fraud_count} / {evaluation.total_fraud_count}\n"
        f"  Fraud capture rate: {evaluation.fraud_capture_rate:.2%}\n"
        f"  Net expected value: £{evaluation.net_expected_value:,.0f}"
    )

    report = {
        "phase": 5,
        "purpose": "final_holdout_evaluation",
        "champion_model": {
            "model_name": frozen_components.model_name,
            "model_version": frozen_components.model_version,
            "mlflow_run_id": frozen_components.mlflow_run_id,
        },
        "calibration": {
            "method": "sigmoid",
            "calibration_fit_period": (
                config["validation_protocol"]["calibration_fit_period"]
            ),
        },
        "data_protection": {
            "final_test_locked": True,
            "final_test_loaded": True,
            "holdout_evaluation_complete": True,
        },
        "policy_configuration": {
            "threshold_policy_version": config["project"]["threshold_policy_version"],
            "configured_operational_capacity": configured_capacity,
            "currency": config["decision_policy"]["currency"],
        },
        "holdout_result": evaluation_result,
    }

    create_holdout_figure(
        evaluation_result=evaluation_result,
        output_path=PHASE5_HOLDOUT_FIGURE_PATH,
    )
    print(f"Saved Phase 5 holdout figure to: {PHASE5_HOLDOUT_FIGURE_PATH}")

    save_json_report(
        report=report,
        output_path=PHASE5_HOLDOUT_OUTPUT_PATH,
    )

    tracking_uri = os.getenv(config["champion_model"]["tracking_uri_env_var"])

    if not tracking_uri:
        raise RuntimeError("MLflow tracking URI is not configured.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="phase5-final-holdout-evaluation") as run:
        mlflow.set_tags(config["mlflow"]["tags"])
        mlflow.set_tag("purpose", "final_holdout_evaluation")
        mlflow.set_tag("final_test_loaded", "true")
        mlflow.set_tag("holdout_evaluation_complete", "true")

        mlflow.log_metrics(
            {
                "holdout_rows": float(len(y_test)),
                "holdout_fraud_count": float(evaluation.total_fraud_count),
                "holdout_captured_fraud": float(evaluation.captured_fraud_count),
                "holdout_fraud_capture_rate": float(evaluation.fraud_capture_rate),
                "holdout_net_expected_value": float(evaluation.net_expected_value),
            }
        )

        mlflow.log_artifact(PHASE5_HOLDOUT_OUTPUT_PATH)
        mlflow.log_artifact(PHASE5_HOLDOUT_FIGURE_PATH)

        print(f"\nPhase 5 MLflow holdout run ID: {run.info.run_id}")

    print("\nPhase 5 final holdout evaluation complete.")
    print("WARNING: Do not use the final test split for further tuning.")


if __name__ == "__main__":
    main()