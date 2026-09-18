"""Evaluate the best compact sequence candidate: the Phase 6 LSTM."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from fraud_intelligence.decisioning.cost_model import CostMatrix
from fraud_intelligence.decisioning.threshold_policy import (
    evaluate_policy_at_capacity,
)
from fraud_intelligence.models.calibration import ProbabilityCalibrator

CALIBRATION_CONFIG_PATH = Path("configs/calibration.yaml")
SEQUENCE_CONFIG_PATH = Path("configs/sequence_model.yaml")

LSTM_MODEL_PATH = Path("models/artifacts/phase6_lstm_sequence_model.keras")
LSTM_METRICS_PATH = Path(
    "reports/tables/phase6_lstm_sequence_validation_metrics.json"
)

POLICY_OUTPUT_PATH = Path(
    "models/metrics/phase6_lstm_sequence_validation_policy_evaluation.json"
)
POLICY_FIGURE_PATH = Path(
    "reports/figures/phase6_lstm_sequence_validation_policy_evaluation.png"
)

CANDIDATE_MODEL_VERSION = "compact-sequence-lstm-v1.0.0"
CALIBRATION_METHOD = "sigmoid"


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and validate one YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Configuration must parse to a dictionary: {path}")

    return config


def load_json_report(path: Path) -> dict[str, Any]:
    """Load one JSON metrics report."""
    if not path.exists():
        raise FileNotFoundError(f"Required report was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise TypeError(f"JSON report must contain an object: {path}")

    return report


def load_cost_matrix(config: dict[str, Any]) -> CostMatrix:
    """Create the existing Phase 5 cost matrix."""
    decision_policy = config["decision_policy"]

    return CostMatrix(
        false_negative_cost=float(decision_policy["false_negative_cost"]),
        manual_review_cost=float(decision_policy["manual_review_cost"]),
        false_positive_escalation_cost=float(
            decision_policy["false_positive_escalation_cost"]
        ),
        fraud_prevention_value=float(decision_policy["fraud_prevention_value"]),
    )


def load_validation_sequences(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load validation sequences, labels, and timestamps."""
    if not path.exists():
        raise FileNotFoundError(
            f"Validation sequence dataset was not found: {path}. "
            "Run scripts/generate_sequence_datasets.py first."
        )

    with np.load(path) as dataset:
        required_arrays = {"sequences", "labels", "timestamps"}
        missing_arrays = required_arrays.difference(dataset.files)

        if missing_arrays:
            raise ValueError(
                f"Validation sequence dataset is missing arrays: "
                f"{sorted(missing_arrays)}"
            )

        sequences = dataset["sequences"].astype(np.float32)
        labels = dataset["labels"].astype(np.int64)
        timestamps = dataset["timestamps"].astype(np.float64)

    if len(sequences) != len(labels) or len(sequences) != len(timestamps):
        raise ValueError("Validation sequence arrays do not have matching row counts.")

    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Validation sequence timestamps are not chronological.")

    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Validation sequence labels must contain fraud and non-fraud.")

    return sequences, labels, timestamps


def create_chronological_split_masks(
    timestamps: np.ndarray,
    calibration_fit_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split validation without separating rows sharing a timestamp."""
    if not 0.0 < calibration_fit_fraction < 1.0:
        raise ValueError("calibration_fit_fraction must be between zero and one.")

    requested_index = int(np.floor(len(timestamps) * calibration_fit_fraction))
    requested_index = min(max(requested_index, 1), len(timestamps) - 1)

    split_timestamp = timestamps[requested_index]
    split_index = int(np.searchsorted(timestamps, split_timestamp, side="left"))

    if split_index == 0 or split_index >= len(timestamps):
        raise RuntimeError(
            "Could not create non-empty chronological calibration and policy periods."
        )

    calibration_mask = np.arange(len(timestamps)) < split_index
    policy_mask = ~calibration_mask

    return calibration_mask, policy_mask, float(split_timestamp)


def validate_binary_period(labels: np.ndarray, period_name: str) -> None:
    """Ensure a calibration or policy period contains both classes."""
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError(f"{period_name} must contain fraud and non-fraud outcomes.")


def create_policy_figure(
    evaluation_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Create LSTM capacity-sensitivity charts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capacities = [result["review_capacity"] for result in evaluation_results]
    capture_rates = [result["fraud_capture_rate"] for result in evaluation_results]
    net_values = [result["net_expected_value"] for result in evaluation_results]

    figure, (capture_axis, value_axis) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    capture_axis.plot(capacities, capture_rates, marker="o", color="#0077B6")
    capture_axis.set_title("Fraud capture rate vs review capacity")
    capture_axis.set_xlabel("Review capacity")
    capture_axis.set_ylabel("Fraud capture rate")
    capture_axis.grid(alpha=0.25)

    value_axis.plot(capacities, net_values, marker="s", color="#D62828")
    value_axis.set_title("Net expected value vs review capacity")
    value_axis.set_xlabel("Review capacity")
    value_axis.set_ylabel("Net expected value (GBP)")
    value_axis.grid(alpha=0.25)

    figure.suptitle(
        "Phase 6 Validation Policy - Sigmoid-Calibrated Compact LSTM"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_json_report(report: dict[str, Any], output_path: Path) -> None:
    """Save the Git-trackable policy report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Saved Phase 6 LSTM policy report to: {output_path}")


def main() -> None:
    """Evaluate the LSTM candidate under the Phase 5 validation policy."""
    calibration_config = load_yaml_config(CALIBRATION_CONFIG_PATH)
    sequence_config = load_yaml_config(SEQUENCE_CONFIG_PATH)

    if calibration_config["data"].get("final_test_locked") is not True:
        raise RuntimeError("Final test protection is disabled in calibration.yaml.")

    if sequence_config["data"].get("final_test_locked") is not True:
        raise RuntimeError("Final test protection is disabled in sequence_model.yaml.")

    lstm_metrics = load_json_report(LSTM_METRICS_PATH)

    if lstm_metrics.get("test_split_loaded") is not False:
        raise RuntimeError(
            "LSTM training metrics indicate final-test access. "
            "Policy evaluation must remain validation-only."
        )

    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM model artifact was not found: {LSTM_MODEL_PATH}. "
            "Run scripts/train_sequence_model.py --architecture lstm first."
        )

    validation_sequence_path = Path(
        sequence_config["outputs"]["validation_sequence_path"]
    )
    sequences, labels, timestamps = load_validation_sequences(
        validation_sequence_path
    )

    print("Loading compact Phase 6 LSTM model...")
    print(f"Candidate model version: {CANDIDATE_MODEL_VERSION}")
    print(f"Candidate MLflow training run ID: {lstm_metrics['run_id']}")
    print("Final test split loaded: False")

    model = tf.keras.models.load_model(LSTM_MODEL_PATH)

    print("\nScoring chronological validation sequences...")
    validation_probabilities = model.predict(
        sequences,
        batch_size=int(sequence_config["model"]["batch_size"]),
        verbose=0,
    ).reshape(-1)

    calibration_fraction = float(
        calibration_config["validation_protocol"]["calibration_fit_fraction"]
    )

    calibration_mask, policy_mask, split_timestamp = (
        create_chronological_split_masks(
            timestamps=timestamps,
            calibration_fit_fraction=calibration_fraction,
        )
    )

    calibration_labels = labels[calibration_mask]
    policy_labels = labels[policy_mask]

    validate_binary_period(calibration_labels, "Calibration-fit period")
    validate_binary_period(policy_labels, "Policy-selection period")

    print("\nChronological validation split")
    print(f"Calibration-fit rows: {calibration_mask.sum():,}")
    print(f"Policy-selection rows: {policy_mask.sum():,}")
    print(f"Split timestamp: {split_timestamp:.0f}")

    print("\nFitting sigmoid calibration on earlier validation data...")
    calibrator = ProbabilityCalibrator(method=CALIBRATION_METHOD)
    calibrator.fit(
        base_probabilities=validation_probabilities[calibration_mask],
        labels=calibration_labels,
    )

    print("Applying sigmoid calibration to later policy-selection data...")
    policy_probabilities = calibrator.predict(
        validation_probabilities[policy_mask]
    )

    cost_matrix = load_cost_matrix(calibration_config)
    review_capacities = [500, 1000, 1500, 2000]
    evaluation_results: list[dict[str, Any]] = []

    print("\nEvaluating capacity-constrained review policy...")

    for review_capacity in review_capacities:
        evaluation = evaluate_policy_at_capacity(
            fraud_probabilities=policy_probabilities,
            labels=policy_labels,
            review_capacity=review_capacity,
            min_probability=0.0,
            max_probability=1.0,
            cost_matrix=cost_matrix,
            action_labels=("approve", "review", "block"),
            policy_version=(
                "phase6-lstm-sigmoid-policy-v1.0.0"
                f"-capacity-{review_capacity}"
            ),
        )

        result = asdict(evaluation)
        evaluation_results.append(result)

        print(
            f"Capacity {review_capacity:>4}: "
            f"captured {evaluation.captured_fraud_count:>4} / "
            f"{evaluation.total_fraud_count} fraud, "
            f"capture rate {evaluation.fraud_capture_rate:.2%}, "
            f"net value GBP {evaluation.net_expected_value:,.0f}"
        )

    operational_capacity = int(
        calibration_config["decision_policy"]["daily_review_capacity"]
    )

    selected_result = next(
        result
        for result in evaluation_results
        if result["review_capacity"] == operational_capacity
    )

    report = {
        "phase": 6,
        "purpose": "lstm_sequence_ablation_validation_policy_evaluation",
        "candidate_model": {
            "model_name": "compact_lstm_sequence_classifier",
            "model_version": CANDIDATE_MODEL_VERSION,
            "mlflow_run_id": lstm_metrics["run_id"],
            "model_path": str(LSTM_MODEL_PATH),
        },
        "sequence_features": {
            "feature_version": sequence_config["project"][
                "sequence_feature_version"
            ],
            "entity_column": sequence_config["data"]["entity_column"],
            "sequence_length": sequence_config["sequence"][
                "max_sequence_length"
            ],
            "input_features": sequence_config["sequence"]["input_features"],
            "same_timestamp_policy": sequence_config["sequence"][
                "same_timestamp_policy"
            ],
            "training_protocol": "strictly_prior_entity_history_only",
            "validation_protocol": (
                "training_history_plus_strictly_prior_validation_history_only"
            ),
            "fraud_labels_used_as_sequence_inputs": False,
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
            "calibration_fit_rows": int(calibration_mask.sum()),
            "policy_selection_rows": int(policy_mask.sum()),
            "calibration_fit_fraud_count": int(calibration_labels.sum()),
            "policy_selection_fraud_count": int(policy_labels.sum()),
        },
        "cost_assumptions": asdict(cost_matrix),
        "configured_operational_capacity": operational_capacity,
        "selected_capacity_result": selected_result,
        "capacity_sensitivity_results": evaluation_results,
    }

    create_policy_figure(
        evaluation_results=evaluation_results,
        output_path=POLICY_FIGURE_PATH,
    )
    print(f"Saved Phase 6 LSTM policy figure to: {POLICY_FIGURE_PATH}")

    save_json_report(report, POLICY_OUTPUT_PATH)

    print("\nPhase 6 LSTM validation policy evaluation complete.")
    print("Final test split loaded: False")


if __name__ == "__main__":
    main()