"""Compare Phase 5 fraud-probability calibration methods safely.

This script:
- Loads the frozen Phase 4 XGBoost champion and its fitted preprocessor.
- Uses chronological validation data only.
- Does not fit, score, load, or inspect the locked final test split.
- Fits calibration mappings on the earlier validation period.
- Selects the provisional calibration method on the later validation period.

Calibration methods compared:
- Uncalibrated XGBoost probabilities.
- Sigmoid calibration (Platt scaling).
- Isotonic regression calibration.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_intelligence.models.calibration import (
    ProbabilityCalibrator,
    evaluate_probability_calibration,
)
from fraud_intelligence.models.data_contract import load_modelling_data
from fraud_intelligence.models.model_loading import (
    load_calibration_config,
    load_frozen_xgboost_components,
)

CALIBRATION_COMPARISON_OUTPUT_PATH = Path(
    "models/metrics/calibration_comparison_validation.json"
)
CALIBRATION_FIGURE_OUTPUT_PATH = Path(
    "reports/figures/phase5_calibration_comparison.png"
)


def validate_phase5_config(config: dict[str, Any]) -> None:
    """Validate the Phase 5 configuration before loading any model data."""
    if "data" not in config:
        raise ValueError("Calibration configuration is missing the 'data' section.")

    if "validation_protocol" not in config:
        raise ValueError(
            "Calibration configuration is missing the 'validation_protocol' section."
        )

    if "calibration" not in config:
        raise ValueError(
            "Calibration configuration is missing the 'calibration' section."
        )

    if config["data"].get("final_test_locked") is not True:
        raise RuntimeError(
            "Final test protection is disabled. "
            "Set data.final_test_locked to true before Phase 5 calibration."
        )

    calibration_fit_fraction = config["validation_protocol"].get(
        "calibration_fit_fraction"
    )

    if not isinstance(calibration_fit_fraction, float | int):
        raise TypeError(
            "validation_protocol.calibration_fit_fraction must be numeric."
        )

    if not 0.0 < calibration_fit_fraction < 1.0:
        raise ValueError(
            "validation_protocol.calibration_fit_fraction must be between 0 and 1."
        )

    methods = config["calibration"].get("methods")

    if not isinstance(methods, list) or not methods:
        raise ValueError(
            "calibration.methods must be a non-empty list."
        )

    unsupported_methods = [
        method
        for method in methods
        if method not in ProbabilityCalibrator.VALID_METHODS
    ]

    if unsupported_methods:
        raise ValueError(
            "Unsupported calibration methods in configuration: "
            f"{unsupported_methods}."
        )


def split_validation_chronologically(
    timestamps: np.ndarray,
    calibration_fit_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split validation rows into earlier calibration and later selection periods.

    The split is made at a timestamp boundary. All rows sharing the same
    timestamp remain in the same period, preventing same-timestamp leakage.
    """
    timestamp_values = np.asarray(timestamps)

    if timestamp_values.ndim != 1:
        raise ValueError("Validation timestamps must be one-dimensional.")

    if timestamp_values.size < 2:
        raise ValueError(
            "At least two validation transactions are required for calibration."
        )

    if not np.isfinite(timestamp_values).all():
        raise ValueError("Validation timestamps must contain only finite values.")

    if np.any(np.diff(timestamp_values) < 0):
        raise ValueError(
            "Validation timestamps must be in chronological ascending order."
        )

    unique_timestamps = np.unique(timestamp_values)

    if unique_timestamps.size < 2:
        raise ValueError(
            "Validation data must contain at least two distinct timestamps."
        )

    requested_row_index = int(
        np.floor(timestamp_values.size * calibration_fit_fraction)
    )
    requested_row_index = min(
        max(requested_row_index, 1),
        timestamp_values.size - 1,
    )

    split_timestamp = timestamp_values[requested_row_index]

    # Move the boundary to the first row at the selected timestamp.
    # This ensures rows at the boundary timestamp are not split between periods.
    split_index = int(
        np.searchsorted(timestamp_values, split_timestamp, side="left")
    )

    if split_index == 0:
        split_timestamp = unique_timestamps[1]
        split_index = int(
            np.searchsorted(timestamp_values, split_timestamp, side="left")
        )

    if split_index >= timestamp_values.size:
        split_timestamp = unique_timestamps[-1]
        split_index = int(
            np.searchsorted(timestamp_values, split_timestamp, side="left")
        )

    if split_index == 0 or split_index >= timestamp_values.size:
        raise RuntimeError(
            "Unable to create non-empty chronological calibration and "
            "selection periods."
        )

    calibration_fit_mask = np.arange(timestamp_values.size) < split_index
    calibration_selection_mask = ~calibration_fit_mask

    return calibration_fit_mask, calibration_selection_mask, float(split_timestamp)


def validate_binary_period_labels(
    labels: np.ndarray,
    period_name: str,
) -> None:
    """Ensure each calibration period contains both classes."""
    unique_labels = np.unique(labels)

    if unique_labels.size != 2 or not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(
            f"The {period_name} period must contain both fraud and "
            "non-fraud transactions."
        )


def choose_provisional_calibration_method(
    results: dict[str, dict[str, Any]],
) -> str:
    """Choose the best method using Brier score, then ECE, then method name."""
    ranked_methods = sorted(
        results,
        key=lambda method: (
            results[method]["brier_score"],
            results[method]["expected_calibration_error"],
            method,
        ),
    )
    return ranked_methods[0]


def create_calibration_figure(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Create a reliability diagram for calibration-selection results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 7))

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Perfect calibration",
    )

    method_colours = {
        "uncalibrated": "#6c757d",
        "sigmoid": "#0077b6",
        "isotonic": "#d62828",
    }

    for method, metrics in results.items():
        brier_score = metrics["brier_score"]
        expected_calibration_error = metrics["expected_calibration_error"]

        axis.plot(
            metrics["mean_predicted_probability"],
            metrics["observed_fraud_rate"],
            marker="o",
            linewidth=2.0,
            color=method_colours.get(method),
            label=(
                f"{method.capitalize()} "
                f"(Brier={brier_score:.5f}, ECE={expected_calibration_error:.5f})"
            ),
        )

    axis.set_title(
        "Phase 5 Calibration Comparison — Later Chronological Validation Period"
    )
    axis.set_xlabel("Mean predicted fraud probability")
    axis.set_ylabel("Observed fraud rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the small Phase 5 comparison report for Git tracking."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Saved calibration comparison report to: {output_path}")


def main() -> None:
    """Run the Phase 5 chronological calibration-method comparison."""
    load_dotenv()

    config = load_calibration_config()
    validate_phase5_config(config)

    # This shared Phase 4 loader verifies train/validation contract integrity
    # and deliberately does not load the locked final test split.
    modelling_data = load_modelling_data()

    frozen_components = load_frozen_xgboost_components()

    # TransactionDT is deliberately excluded from X_validation because it must
    # never enter the model feature matrix. Read only that timestamp column from
    # the same validation Parquet table to create the Phase 5 time-safe split.
    validation_feature_path = Path(
        config["data"]["validation_feature_path"]
    )
    timestamp_column = config["data"]["timestamp_column"]

    if not validation_feature_path.exists():
        raise FileNotFoundError(
            "Validation feature table was not found: "
            f"{validation_feature_path}"
        )

    validation_timestamp_dataframe = pd.read_parquet(
        validation_feature_path,
        columns=[timestamp_column],
    )

    if len(validation_timestamp_dataframe) != len(modelling_data.X_validation):
        raise RuntimeError(
            "Validation timestamp data and modelling validation data have "
            "different row counts. Calibration cannot continue safely."
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
    validation_features_transformed = frozen_components.preprocessor.transform(
        modelling_data.X_validation
    )

    print("Scoring chronological validation data with the frozen model...")
    base_validation_probabilities = (
        frozen_components.classifier.predict_proba(
            validation_features_transformed
        )[:, 1]
    )

    calibration_fit_fraction = float(
        config["validation_protocol"]["calibration_fit_fraction"]
    )

    calibration_fit_mask, calibration_selection_mask, split_timestamp = (
        split_validation_chronologically(
            timestamps=validation_timestamps,
            calibration_fit_fraction=calibration_fit_fraction,
        )
    )

    calibration_fit_labels = validation_labels[calibration_fit_mask]
    calibration_selection_labels = validation_labels[calibration_selection_mask]

    validate_binary_period_labels(
        labels=calibration_fit_labels,
        period_name="calibration-fit",
    )
    validate_binary_period_labels(
        labels=calibration_selection_labels,
        period_name="calibration-selection",
    )

    calibration_fit_probabilities = base_validation_probabilities[
        calibration_fit_mask
    ]
    calibration_selection_probabilities = base_validation_probabilities[
        calibration_selection_mask
    ]

    print("\nChronological validation split created")
    print(f"Calibration-fit rows: {calibration_fit_mask.sum():,}")
    print(f"Calibration-selection rows: {calibration_selection_mask.sum():,}")
    print(f"Split timestamp: {split_timestamp:.0f}")

    calibration_config = config["calibration"]
    curve_config = calibration_config["calibration_curve"]
    ece_config = calibration_config["expected_calibration_error"]

    if curve_config["n_bins"] != ece_config["n_bins"]:
        raise ValueError(
            "Calibration-curve and ECE bin counts must match in this workflow."
        )

    methods = calibration_config["methods"]
    comparison_results: dict[str, dict[str, Any]] = {}

    print("\nComparing calibration methods on the later validation period...")

    for method in methods:
        calibrator = ProbabilityCalibrator(method=method)

        calibrated_selection_probabilities = calibrator.fit_predict(
            calibration_probabilities=calibration_fit_probabilities,
            calibration_labels=calibration_fit_labels,
            prediction_probabilities=calibration_selection_probabilities,
        )

        calibration_metrics = evaluate_probability_calibration(
            method=method,
            labels=calibration_selection_labels,
            probabilities=calibrated_selection_probabilities,
            n_bins=int(curve_config["n_bins"]),
            strategy=str(curve_config["strategy"]),
        )

        comparison_results[method] = {
            **asdict(calibration_metrics),
            "pr_auc": float(
                average_precision_score(
                    calibration_selection_labels,
                    calibrated_selection_probabilities,
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    calibration_selection_labels,
                    calibrated_selection_probabilities,
                )
            ),
        }

        print(
            f"{method.capitalize():<14} "
            f"Brier={comparison_results[method]['brier_score']:.6f} | "
            f"ECE={comparison_results[method]['expected_calibration_error']:.6f} | "
            f"PR-AUC={comparison_results[method]['pr_auc']:.6f}"
        )

    selected_method = choose_provisional_calibration_method(
        comparison_results
    )

    report = {
        "phase": 5,
        "purpose": "calibration_method_comparison",
        "champion_model": {
            "model_name": frozen_components.model_name,
            "model_version": frozen_components.model_version,
            "mlflow_run_id": frozen_components.mlflow_run_id,
            "preprocessor_model_uri": frozen_components.preprocessor_model_uri,
            "classifier_model_uri": frozen_components.classifier_model_uri,
        },
        "data_protection": {
            "final_test_locked": True,
            "final_test_loaded": False,
            "validation_strategy": (
                "chronological calibration-fit period followed by "
                "chronological calibration-selection period"
            ),
        },
        "validation_split": {
            "validation_rows": int(validation_labels.size),
            "calibration_fit_fraction_requested": calibration_fit_fraction,
            "calibration_fit_rows": int(calibration_fit_mask.sum()),
            "calibration_selection_rows": int(
                calibration_selection_mask.sum()
            ),
            "split_timestamp": split_timestamp,
            "calibration_fit_fraud_count": int(
                calibration_fit_labels.sum()
            ),
            "calibration_selection_fraud_count": int(
                calibration_selection_labels.sum()
            ),
        },
        "selection_rule": {
            "primary_metric": calibration_config["primary_selection_metric"],
            "secondary_metric": calibration_config[
                "secondary_selection_metric"
            ],
            "provisional_selected_method": selected_method,
        },
        "results": comparison_results,
    }

    create_calibration_figure(
        results=comparison_results,
        output_path=CALIBRATION_FIGURE_OUTPUT_PATH,
    )
    print(
        "Saved calibration comparison figure to: "
        f"{CALIBRATION_FIGURE_OUTPUT_PATH}"
    )

    save_json_report(
        report=report,
        output_path=CALIBRATION_COMPARISON_OUTPUT_PATH,
    )

    tracking_uri = os.getenv(
        config["champion_model"]["tracking_uri_env_var"]
    )

    if not tracking_uri:
        raise RuntimeError(
            "MLflow tracking URI is not configured in the local .env file."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="phase5-calibration-comparison") as run:
        mlflow.set_tags(config["mlflow"]["tags"])
        mlflow.set_tag(
            "frozen_phase4_xgboost_run_id",
            frozen_components.mlflow_run_id,
        )
        mlflow.set_tag("final_test_loaded", "false")
        mlflow.set_tag("selected_calibration_method", selected_method)

        mlflow.log_metrics(
            {
                "calibration_fit_rows": float(calibration_fit_mask.sum()),
                "calibration_selection_rows": float(
                    calibration_selection_mask.sum()
                ),
            }
        )

        for method, metrics in comparison_results.items():
            mlflow.log_metrics(
                {
                    f"{method}_brier_score": float(metrics["brier_score"]),
                    f"{method}_expected_calibration_error": float(
                        metrics["expected_calibration_error"]
                    ),
                    f"{method}_pr_auc": float(metrics["pr_auc"]),
                    f"{method}_roc_auc": float(metrics["roc_auc"]),
                }
            )

        mlflow.log_artifact(CALIBRATION_COMPARISON_OUTPUT_PATH)
        mlflow.log_artifact(CALIBRATION_FIGURE_OUTPUT_PATH)

        print(f"\nPhase 5 MLflow run ID: {run.info.run_id}")

    print("\nProvisional calibration-method selection")
    print("-" * 48)
    print(f"Selected method: {selected_method}")
    print("Selection period: later chronological validation period")
    print("Final test split loaded: False")


if __name__ == "__main__":
    main()