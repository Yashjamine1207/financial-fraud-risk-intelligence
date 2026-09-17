"""Create the Phase 4 Logistic Regression versus XGBoost comparison report.

This script reads saved validation metrics only. It does not retrain models and
does not load the locked final test split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

LOGISTIC_METRICS_PATH = Path("reports/tables/logistic_regression_validation_metrics.json")
XGBOOST_METRICS_PATH = Path("reports/tables/xgboost_validation_metrics.json")

COMPARISON_CSV_PATH = Path("models/metrics/model_comparison.csv")
COMPARISON_REPORT_PATH = Path("reports/evaluation/phase4_model_comparison.md")

MODEL_ORDER = ["logistic_regression", "xgboost"]

COMPARISON_METRICS = [
    "pr_auc",
    "roc_auc",
    "precision_at_reference_threshold",
    "recall_at_reference_threshold",
    "f1_at_reference_threshold",
    "precision_at_k",
    "recall_at_k",
    "captured_fraud_count_at_k",
    "total_fraud_count",
    "prediction_latency_ms_per_row",
    "training_time_seconds",
]


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    """Load one saved model-validation metrics JSON file."""
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file was not found: {metrics_path}. "
            "Train the model before creating the comparison report."
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    required_keys = {
        "model_name",
        "model_version",
        "dataset_version",
        "feature_version",
        "review_capacity",
        "test_split_loaded",
        *COMPARISON_METRICS,
    }
    missing_keys = required_keys.difference(metrics)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Metrics file {metrics_path} is missing required key(s): {missing}")

    if metrics["test_split_loaded"]:
        raise ValueError(
            f"Metrics file {metrics_path} indicates that the locked test split "
            "was loaded. Phase 4 comparison must use validation metrics only."
        )

    return metrics


def validate_comparable_experiments(
    logistic_metrics: dict[str, Any],
    xgboost_metrics: dict[str, Any],
) -> None:
    """Confirm both models used the same validation decision context."""
    comparable_fields = [
        "dataset_version",
        "feature_version",
        "target_encoding_version",
        "review_capacity",
        "total_fraud_count",
    ]

    mismatches = [
        field for field in comparable_fields if logistic_metrics[field] != xgboost_metrics[field]
    ]

    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise ValueError(
            "Model comparison is invalid because experiments differ in: " f"{mismatch_text}"
        )


def build_comparison_table(
    logistic_metrics: dict[str, Any],
    xgboost_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Build a model-by-metric comparison table."""
    model_metrics = {
        "logistic_regression": logistic_metrics,
        "xgboost": xgboost_metrics,
    }

    comparison_rows: list[dict[str, Any]] = []

    for metric_name in COMPARISON_METRICS:
        logistic_value = logistic_metrics[metric_name]
        xgboost_value = xgboost_metrics[metric_name]

        comparison_rows.append(
            {
                "metric": metric_name,
                "logistic_regression": logistic_value,
                "xgboost": xgboost_value,
                "absolute_difference_xgboost_minus_logistic": (xgboost_value - logistic_value),
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)

    model_metrics["logistic_regression"]
    model_metrics["xgboost"]

    return comparison_table


def select_validation_champion(
    logistic_metrics: dict[str, Any],
    xgboost_metrics: dict[str, Any],
) -> tuple[str, str]:
    """Select a provisional champion using validation ranking performance."""
    if xgboost_metrics["pr_auc"] > logistic_metrics["pr_auc"]:
        return (
            "xgboost",
            "XGBoost has the higher validation PR-AUC.",
        )

    if logistic_metrics["pr_auc"] > xgboost_metrics["pr_auc"]:
        return (
            "logistic_regression",
            "Logistic Regression has the higher validation PR-AUC.",
        )

    if xgboost_metrics["recall_at_k"] > logistic_metrics["recall_at_k"]:
        return (
            "xgboost",
            "PR-AUC tied, and XGBoost captured more fraud at review capacity.",
        )

    return (
        "logistic_regression",
        (
            "PR-AUC tied, and Logistic Regression matched or exceeded fraud "
            "capture at review capacity."
        ),
    )


def write_markdown_report(
    logistic_metrics: dict[str, Any],
    xgboost_metrics: dict[str, Any],
    champion_name: str,
    champion_reason: str,
) -> None:
    """Write the Phase 4 validation-only model-selection report."""
    capacity = logistic_metrics["review_capacity"]

    captured_fraud_difference = (
        xgboost_metrics["captured_fraud_count_at_k"] - logistic_metrics["captured_fraud_count_at_k"]
    )
    precision_at_k_difference = (
        xgboost_metrics["precision_at_k"] - logistic_metrics["precision_at_k"]
    )
    recall_at_k_difference = xgboost_metrics["recall_at_k"] - logistic_metrics["recall_at_k"]

    report = f"""# Phase 4 Model Comparison

## Scope

This report compares the Phase 4 Logistic Regression baseline and XGBoost
tabular model using only the chronological validation split.

The final test split was not loaded by either experiment and remains locked for
later final evaluation.

## Shared evaluation context

| Item | Value |
|---|---:|
| Dataset version | {logistic_metrics["dataset_version"]} |
| Feature version | {logistic_metrics["feature_version"]} |
| Target-encoding version | {logistic_metrics["target_encoding_version"]} |
| Validation review capacity | {capacity:,} transactions |
| Validation fraud cases | {logistic_metrics["total_fraud_count"]:,} |
| Test split loaded | False |

## Validation metrics

| Metric | Logistic Regression | XGBoost | Difference: XGBoost − Logistic Regression |
|---|---:|---:|---:|
| PR-AUC | {logistic_metrics["pr_auc"]:.6f} | {xgboost_metrics["pr_auc"]:.6f} | {xgboost_metrics["pr_auc"] - logistic_metrics["pr_auc"]:.6f} |
| ROC-AUC | {logistic_metrics["roc_auc"]:.6f} | {xgboost_metrics["roc_auc"]:.6f} | {xgboost_metrics["roc_auc"] - logistic_metrics["roc_auc"]:.6f} |
| Precision@{capacity:,} | {logistic_metrics["precision_at_k"]:.6f} | {xgboost_metrics["precision_at_k"]:.6f} | {precision_at_k_difference:.6f} |
| Recall@{capacity:,} | {logistic_metrics["recall_at_k"]:.6f} | {xgboost_metrics["recall_at_k"]:.6f} | {recall_at_k_difference:.6f} |
| Fraud captured at {capacity:,} reviews | {logistic_metrics["captured_fraud_count_at_k"]:,} | {xgboost_metrics["captured_fraud_count_at_k"]:,} | {captured_fraud_difference:+,} |
| Precision at 0.50 reference threshold | {logistic_metrics["precision_at_reference_threshold"]:.6f} | {xgboost_metrics["precision_at_reference_threshold"]:.6f} | {xgboost_metrics["precision_at_reference_threshold"] - logistic_metrics["precision_at_reference_threshold"]:.6f} |
| Recall at 0.50 reference threshold | {logistic_metrics["recall_at_reference_threshold"]:.6f} | {xgboost_metrics["recall_at_reference_threshold"]:.6f} | {xgboost_metrics["recall_at_reference_threshold"] - logistic_metrics["recall_at_reference_threshold"]:.6f} |
| F1 at 0.50 reference threshold | {logistic_metrics["f1_at_reference_threshold"]:.6f} | {xgboost_metrics["f1_at_reference_threshold"]:.6f} | {xgboost_metrics["f1_at_reference_threshold"] - logistic_metrics["f1_at_reference_threshold"]:.6f} |
| Prediction latency per row (ms) | {logistic_metrics["prediction_latency_ms_per_row"]:.6f} | {xgboost_metrics["prediction_latency_ms_per_row"]:.6f} | {xgboost_metrics["prediction_latency_ms_per_row"] - logistic_metrics["prediction_latency_ms_per_row"]:.6f} |

## Provisional validation champion

**{champion_name}**

Reason: {champion_reason}

At the fixed capacity of {capacity:,} manual reviews, XGBoost captured
{captured_fraud_difference:,} more known fraud cases than Logistic Regression.

## Decision boundary

This is a validation-based model-selection decision only.

It does not represent:
- final test-set performance;
- calibrated fraud probabilities;
- an approved production threshold;
- an estimated financial-cost outcome.

Phase 5 will calibrate the selected model and optimise the approve, review, and
block decision policy using documented cost assumptions.
"""

    COMPARISON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with COMPARISON_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)


def main() -> None:
    """Create comparison CSV and Markdown report from saved model metrics."""
    logistic_metrics = load_metrics(LOGISTIC_METRICS_PATH)
    xgboost_metrics = load_metrics(XGBOOST_METRICS_PATH)

    validate_comparable_experiments(
        logistic_metrics=logistic_metrics,
        xgboost_metrics=xgboost_metrics,
    )

    comparison_table = build_comparison_table(
        logistic_metrics=logistic_metrics,
        xgboost_metrics=xgboost_metrics,
    )

    champion_name, champion_reason = select_validation_champion(
        logistic_metrics=logistic_metrics,
        xgboost_metrics=xgboost_metrics,
    )

    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_table.to_csv(COMPARISON_CSV_PATH, index=False)

    write_markdown_report(
        logistic_metrics=logistic_metrics,
        xgboost_metrics=xgboost_metrics,
        champion_name=champion_name,
        champion_reason=champion_reason,
    )

    print("Phase 4 comparison completed.")
    print(f"Validation champion: {champion_name}")
    print(f"Reason: {champion_reason}")
    print(f"Comparison CSV: {COMPARISON_CSV_PATH}")
    print(f"Comparison report: {COMPARISON_REPORT_PATH}")
    print("Final test split loaded: False")


if __name__ == "__main__":
    main()
