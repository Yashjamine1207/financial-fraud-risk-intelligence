"""Compare Phase 4 XGBoost baseline with the Phase 6 anomaly ablation.

This script compares validation-only results for:

1. Baseline:
   Phase 4 XGBoost using the leakage-safe point-in-time feature contract.

2. Candidate:
   The same XGBoost pipeline plus the Isolation Forest anomaly-score feature.

Important:
- This script reads only small saved metric JSON files.
- It does not load train, validation, or final-test feature tables.
- It does not tune a model, calibration method, threshold, or review capacity.
- It does not use the locked final test split.
- It creates a preliminary ranking comparison only.
- The formal keep/reject decision will be made later using the identical
  Phase 5 calibration and capacity-constrained policy-selection protocol.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

BASELINE_METRICS_PATH = Path("reports/tables/xgboost_validation_metrics.json")
ANOMALY_METRICS_PATH = Path("reports/tables/xgboost_anomaly_validation_metrics.json")

COMPARISON_CSV_PATH = Path("reports/tables/phase6_anomaly_ablation_comparison.csv")
COMPARISON_REPORT_PATH = Path("reports/evaluation/phase6_anomaly_ablation_comparison.md")

BASELINE_MODEL_VERSION = "xgboost-v1.0.0"
CANDIDATE_MODEL_VERSION = "xgboost-plus-anomaly-v1.0.0"
ANOMALY_FEATURE_NAME = "anomaly_score_isolation_forest"


METRICS_TO_COMPARE: list[tuple[str, str, str]] = [
    ("pr_auc", "PR-AUC", ".6f"),
    ("roc_auc", "ROC-AUC", ".6f"),
    (
        "precision_at_reference_threshold",
        "Precision at reference threshold",
        ".6f",
    ),
    (
        "recall_at_reference_threshold",
        "Recall at reference threshold",
        ".6f",
    ),
    (
        "f1_at_reference_threshold",
        "F1 at reference threshold",
        ".6f",
    ),
    ("precision_at_k", "Precision@1,000", ".6f"),
    ("recall_at_k", "Recall@1,000", ".6f"),
    ("captured_fraud_count_at_k", "Fraud captured at 1,000 reviews", ".0f"),
    ("total_fraud_count", "Total known fraud in validation", ".0f"),
    (
        "prediction_latency_ms_per_row",
        "Prediction latency per row (ms)",
        ".6f",
    ),
    ("training_time_seconds", "Training time (seconds)", ".2f"),
    ("best_iteration", "Best boosting iteration", ".0f"),
]


def load_metrics(path: Path) -> dict[str, Any]:
    """Load one saved metrics JSON report and validate its basic structure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file was not found: {path}. "
            "Run the corresponding training script before comparison."
        )

    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    if not isinstance(metrics, dict):
        raise TypeError(f"Metrics file must contain a JSON object: {path}")

    return metrics


def validate_comparison_inputs(
    baseline_metrics: dict[str, Any],
    anomaly_metrics: dict[str, Any],
) -> None:
    """Confirm both experiments are comparable before reporting deltas."""
    required_common_fields = [
        "dataset_version",
        "feature_version",
        "target_encoding_version",
        "review_capacity",
        "total_fraud_count",
        "test_split_loaded",
    ]

    for field in required_common_fields:
        if field not in baseline_metrics:
            raise ValueError(f"Baseline metrics are missing required field: '{field}'.")

        if field not in anomaly_metrics:
            raise ValueError(f"Anomaly metrics are missing required field: '{field}'.")

    for field in [
        "dataset_version",
        "feature_version",
        "target_encoding_version",
        "review_capacity",
        "total_fraud_count",
    ]:
        if baseline_metrics[field] != anomaly_metrics[field]:
            raise ValueError(
                f"Experiments are not comparable because '{field}' differs. "
                f"Baseline={baseline_metrics[field]!r}, "
                f"candidate={anomaly_metrics[field]!r}."
            )

    if baseline_metrics["test_split_loaded"] is not False:
        raise ValueError(
            "Baseline metrics indicate that the final test split was loaded. "
            "This comparison must use validation-only metrics."
        )

    if anomaly_metrics["test_split_loaded"] is not False:
        raise ValueError(
            "Anomaly metrics indicate that the final test split was loaded. "
            "This comparison must use validation-only metrics."
        )

    if baseline_metrics["review_capacity"] != 1000:
        raise ValueError(
            "This Phase 6 preliminary comparison expects the documented "
            "review capacity of 1,000 transactions."
        )

    for metric_key, _, _ in METRICS_TO_COMPARE:
        if metric_key not in baseline_metrics:
            raise ValueError(f"Baseline metrics are missing comparison metric: '{metric_key}'.")

        if metric_key not in anomaly_metrics:
            raise ValueError(f"Anomaly metrics are missing comparison metric: '{metric_key}'.")


def build_comparison_rows(
    baseline_metrics: dict[str, Any],
    anomaly_metrics: dict[str, Any],
) -> list[dict[str, float | str]]:
    """Build rows containing baseline, candidate, and candidate-minus-baseline."""
    rows: list[dict[str, float | str]] = []

    for metric_key, metric_name, _ in METRICS_TO_COMPARE:
        baseline_value = float(baseline_metrics[metric_key])
        anomaly_value = float(anomaly_metrics[metric_key])
        difference = anomaly_value - baseline_value

        rows.append(
            {
                "metric_key": metric_key,
                "metric": metric_name,
                "baseline_xgboost": baseline_value,
                "xgboost_plus_anomaly": anomaly_value,
                "candidate_minus_baseline": difference,
            }
        )

    return rows


def save_comparison_csv(rows: list[dict[str, float | str]]) -> None:
    """Save the tabular ablation comparison for the portfolio repository."""
    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "metric_key",
        "metric",
        "baseline_xgboost",
        "xgboost_plus_anomaly",
        "candidate_minus_baseline",
    ]

    with COMPARISON_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison CSV to: {COMPARISON_CSV_PATH}")


def format_value(value: float, format_specifier: str) -> str:
    """Format a metric consistently for the Markdown report."""
    return format(value, format_specifier)


def build_markdown_report(
    baseline_metrics: dict[str, Any],
    anomaly_metrics: dict[str, Any],
    rows: list[dict[str, float | str]],
) -> str:
    """Build the validation-only Phase 6 anomaly comparison report."""
    metric_format_by_key = {
        metric_key: format_specifier for metric_key, _, format_specifier in METRICS_TO_COMPARE
    }

    lines = [
        "# Phase 6 — Isolation Forest Anomaly Ablation",
        "",
        "## Objective",
        "",
        "Compare the Phase 4 XGBoost baseline with the same XGBoost model plus",
        "`anomaly_score_isolation_forest`.",
        "",
        "The purpose is to test whether an unsupervised anomaly signal adds",
        "measurable fraud-prioritisation value beyond the existing leakage-safe",
        "point-in-time behavioural features.",
        "",
        "## Experimental protocol",
        "",
        f"- Baseline model: `{BASELINE_MODEL_VERSION}`",
        f"- Candidate model: `{CANDIDATE_MODEL_VERSION}`",
        f"- Added candidate feature: `{ANOMALY_FEATURE_NAME}`",
        f"- Dataset version: `{baseline_metrics['dataset_version']}`",
        f"- Base feature version: `{baseline_metrics['feature_version']}`",
        f"- Target-encoding version: `{baseline_metrics['target_encoding_version']}`",
        f"- Validation review capacity: {baseline_metrics['review_capacity']:,} transactions",
        f"- Validation known fraud count: {baseline_metrics['total_fraud_count']:,}",
        "- Evaluation split: chronological validation data only",
        "- Final test split loaded: False",
        "",
        "## Leakage controls",
        "",
        "- Isolation Forest was trained without `isFraud` labels.",
        "- Training anomaly scores were generated using chronological out-of-fold scoring.",
        "- Each training scoring fold used an Isolation Forest fitted only on earlier training folds.",
        "- Validation anomaly scores were produced with an Isolation Forest fitted on the full chronological training period only.",
        "- The earliest training fold has no earlier history and therefore has intentionally missing anomaly scores.",
        "- The existing training-fitted numeric preprocessing pipeline handles those missing values using median imputation and missingness indicators.",
        "",
        "## Validation comparison",
        "",
        "| Metric | Baseline XGBoost | XGBoost + anomaly | Candidate minus baseline |",
        "|---|---:|---:|---:|",
    ]

    for row in rows:
        metric_key = str(row["metric_key"])
        format_specifier = metric_format_by_key[metric_key]

        baseline_value = format_value(
            float(row["baseline_xgboost"]),
            format_specifier,
        )
        anomaly_value = format_value(
            float(row["xgboost_plus_anomaly"]),
            format_specifier,
        )
        difference_value = format_value(
            float(row["candidate_minus_baseline"]),
            f"+{format_specifier}",
        )

        lines.append(
            f"| {row['metric']} | {baseline_value} | " f"{anomaly_value} | {difference_value} |"
        )

    baseline_captured = int(baseline_metrics["captured_fraud_count_at_k"])
    anomaly_captured = int(anomaly_metrics["captured_fraud_count_at_k"])
    captured_difference = anomaly_captured - baseline_captured

    lines.extend(
        [
            "",
            "## Preliminary interpretation",
            "",
            (
                f"At the fixed review capacity of "
                f"{baseline_metrics['review_capacity']:,} transactions, the "
                f"candidate captured {anomaly_captured:,} known fraud cases "
                f"compared with {baseline_captured:,} for the baseline "
                f"({captured_difference:+,} cases)."
            ),
            "",
            (
                "This is a preliminary ranking comparison only. It is not yet "
                "the final keep/reject decision because the candidate still "
                "needs the same calibration and capacity-constrained policy "
                "evaluation used by the Phase 5 model."
            ),
            "",
            "## Next evaluation requirement",
            "",
            (
                "Fit the candidate calibration mapping using only the earlier "
                "Phase 5 validation calibration-fit period. Then compare the "
                "baseline and candidate on the later chronological "
                "policy-selection period at the same review capacities and "
                "cost assumptions. Do not use the final test split during "
                "this decision."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def save_markdown_report(report: str) -> None:
    """Save the human-readable Phase 6 anomaly ablation report."""
    COMPARISON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with COMPARISON_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    print(f"Saved comparison report to: {COMPARISON_REPORT_PATH}")


def main() -> None:
    """Create the Phase 6 validation-only anomaly ablation comparison."""
    baseline_metrics = load_metrics(BASELINE_METRICS_PATH)
    anomaly_metrics = load_metrics(ANOMALY_METRICS_PATH)

    validate_comparison_inputs(
        baseline_metrics=baseline_metrics,
        anomaly_metrics=anomaly_metrics,
    )

    rows = build_comparison_rows(
        baseline_metrics=baseline_metrics,
        anomaly_metrics=anomaly_metrics,
    )

    save_comparison_csv(rows)

    report = build_markdown_report(
        baseline_metrics=baseline_metrics,
        anomaly_metrics=anomaly_metrics,
        rows=rows,
    )
    save_markdown_report(report)

    print("\nPhase 6 anomaly ablation comparison completed.")
    print(
        "Fraud captured at 1,000 reviews: "
        f"{int(baseline_metrics['captured_fraud_count_at_k']):,} "
        "(baseline) vs "
        f"{int(anomaly_metrics['captured_fraud_count_at_k']):,} "
        "(XGBoost + anomaly)"
    )
    print(
        "PR-AUC: "
        f"{float(baseline_metrics['pr_auc']):.6f} "
        "(baseline) vs "
        f"{float(anomaly_metrics['pr_auc']):.6f} "
        "(XGBoost + anomaly)"
    )
    print("Final test split loaded: False")


if __name__ == "__main__":
    main()
