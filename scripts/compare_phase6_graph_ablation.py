"""Compare Phase 4 XGBoost baseline with the Phase 6 graph-feature ablation.

This script compares validation-only results for:

1. Baseline:
   Phase 4 XGBoost using the leakage-safe point-in-time feature contract.

2. Candidate:
   The same XGBoost pipeline plus ten leakage-safe graph features.

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
GRAPH_METRICS_PATH = Path("reports/tables/xgboost_graph_validation_metrics.json")

COMPARISON_CSV_PATH = Path("reports/tables/phase6_graph_ablation_comparison.csv")
COMPARISON_REPORT_PATH = Path("reports/evaluation/phase6_graph_ablation_comparison.md")

BASELINE_MODEL_VERSION = "xgboost-v1.0.0"
CANDIDATE_MODEL_VERSION = "xgboost-plus-graph-v1.0.0"
GRAPH_FEATURE_VERSION = "graph-features-v1.0.0"
GRAPH_FEATURE_COUNT = 10


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
    """Load one saved metrics JSON report."""
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
    graph_metrics: dict[str, Any],
) -> None:
    """Confirm that baseline and graph candidate are comparable."""
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

        if field not in graph_metrics:
            raise ValueError(
                f"Graph candidate metrics are missing required field: '{field}'."
            )

    for field in [
        "dataset_version",
        "feature_version",
        "target_encoding_version",
        "review_capacity",
        "total_fraud_count",
    ]:
        if baseline_metrics[field] != graph_metrics[field]:
            raise ValueError(
                f"Experiments are not comparable because '{field}' differs. "
                f"Baseline={baseline_metrics[field]!r}, "
                f"graph_candidate={graph_metrics[field]!r}."
            )

    if baseline_metrics["test_split_loaded"] is not False:
        raise ValueError(
            "Baseline metrics indicate final-test access. "
            "This comparison must use validation-only metrics."
        )

    if graph_metrics["test_split_loaded"] is not False:
        raise ValueError(
            "Graph candidate metrics indicate final-test access. "
            "This comparison must use validation-only metrics."
        )

    if baseline_metrics["review_capacity"] != 1000:
        raise ValueError(
            "This Phase 6 comparison expects the documented review capacity "
            "of 1,000 transactions."
        )

    if graph_metrics.get("graph_feature_count") != GRAPH_FEATURE_COUNT:
        raise ValueError(
            f"Graph candidate must contain exactly {GRAPH_FEATURE_COUNT} "
            "graph features for this controlled ablation."
        )

    for metric_key, _, _ in METRICS_TO_COMPARE:
        if metric_key not in baseline_metrics:
            raise ValueError(
                f"Baseline metrics are missing comparison metric: '{metric_key}'."
            )

        if metric_key not in graph_metrics:
            raise ValueError(
                f"Graph candidate metrics are missing comparison metric: '{metric_key}'."
            )


def build_comparison_rows(
    baseline_metrics: dict[str, Any],
    graph_metrics: dict[str, Any],
) -> list[dict[str, float | str]]:
    """Build baseline, candidate, and candidate-minus-baseline comparison rows."""
    rows: list[dict[str, float | str]] = []

    for metric_key, metric_name, _ in METRICS_TO_COMPARE:
        baseline_value = float(baseline_metrics[metric_key])
        graph_value = float(graph_metrics[metric_key])

        rows.append(
            {
                "metric_key": metric_key,
                "metric": metric_name,
                "baseline_xgboost": baseline_value,
                "xgboost_plus_graph_features": graph_value,
                "candidate_minus_baseline": graph_value - baseline_value,
            }
        )

    return rows


def save_comparison_csv(rows: list[dict[str, float | str]]) -> None:
    """Save the ablation comparison table."""
    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "metric_key",
        "metric",
        "baseline_xgboost",
        "xgboost_plus_graph_features",
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

    print(f"Saved graph comparison CSV to: {COMPARISON_CSV_PATH}")


def build_markdown_report(
    baseline_metrics: dict[str, Any],
    graph_metrics: dict[str, Any],
    rows: list[dict[str, float | str]],
) -> str:
    """Build the validation-only graph-feature ablation report."""
    metric_format_by_key = {
        metric_key: format_specifier
        for metric_key, _, format_specifier in METRICS_TO_COMPARE
    }

    lines = [
        "# Phase 6 - Time-Safe Graph-Feature Ablation",
        "",
        "## Objective",
        "",
        "Compare the Phase 4 XGBoost baseline with the same XGBoost model plus",
        "ten time-safe graph features derived from historical card-device and",
        "card-recipient-email relationships.",
        "",
        "The purpose is to test whether historical relationship structure adds",
        "measurable fraud-prioritisation value beyond the existing leakage-safe",
        "point-in-time behavioural features.",
        "",
        "## Experimental protocol",
        "",
        f"- Baseline model: `{BASELINE_MODEL_VERSION}`",
        f"- Candidate model: `{CANDIDATE_MODEL_VERSION}`",
        f"- Graph feature version: `{GRAPH_FEATURE_VERSION}`",
        f"- Added graph features: {GRAPH_FEATURE_COUNT}",
        f"- Dataset version: `{baseline_metrics['dataset_version']}`",
        f"- Base feature version: `{baseline_metrics['feature_version']}`",
        f"- Target-encoding version: `{baseline_metrics['target_encoding_version']}`",
        (
            "- Validation review capacity: "
            f"{baseline_metrics['review_capacity']:,} transactions"
        ),
        (
            "- Validation known fraud count: "
            f"{baseline_metrics['total_fraud_count']:,}"
        ),
        "- Evaluation split: chronological validation data only",
        "- Final test split loaded: False",
        "",
        "## Leakage controls",
        "",
        "- Graph features use only earlier transaction timestamps.",
        "- All same-timestamp transactions are scored before graph state updates.",
        "- Validation graph state begins with training history only.",
        "- Validation graph state then updates only from earlier validation timestamps.",
        "- Missing device and recipient-email values do not form shared graph nodes.",
        "- Fraud labels are never used in graph construction or graph features.",
        "- Numeric preprocessing and missingness handling are fitted on training data only.",
        "",
        "## Added graph features",
        "",
        "| Feature family | Features |",
        "|---|---|",
        (
            "| Historical graph degree | "
            "`graph_card1_prior_degree`, "
            "`graph_deviceinfo_prior_card_degree`, "
            "`graph_remail_prior_card_degree` |"
        ),
        (
            "| Historical edge frequency | "
            "`graph_card1_device_prior_edge_count`, "
            "`graph_card1_remail_prior_edge_count` |"
        ),
        (
            "| Historical component size | "
            "`graph_card1_prior_component_size`, "
            "`graph_deviceinfo_prior_component_size`, "
            "`graph_remail_prior_component_size` |"
        ),
        (
            "| New relationship indicators | "
            "`is_new_card1_device_relationship`, "
            "`is_new_card1_remail_relationship` |"
        ),
        "",
        "## Validation comparison",
        "",
        (
            "| Metric | Baseline XGBoost | XGBoost + graph features | "
            "Candidate minus baseline |"
        ),
        "|---|---:|---:|---:|",
    ]

    for row in rows:
        metric_key = str(row["metric_key"])
        format_specifier = metric_format_by_key[metric_key]

        baseline_value = format(
            float(row["baseline_xgboost"]),
            format_specifier,
        )
        graph_value = format(
            float(row["xgboost_plus_graph_features"]),
            format_specifier,
        )
        difference_value = format(
            float(row["candidate_minus_baseline"]),
            f"+{format_specifier}",
        )

        lines.append(
            f"| {row['metric']} | {baseline_value} | "
            f"{graph_value} | {difference_value} |"
        )

    baseline_captured = int(baseline_metrics["captured_fraud_count_at_k"])
    graph_captured = int(graph_metrics["captured_fraud_count_at_k"])
    captured_difference = graph_captured - baseline_captured

    lines.extend(
        [
            "",
            "## Preliminary interpretation",
            "",
            (
                f"At the fixed review capacity of "
                f"{baseline_metrics['review_capacity']:,} transactions, the "
                f"candidate captured {graph_captured:,} known fraud cases "
                f"compared with {baseline_captured:,} for the baseline "
                f"({captured_difference:+,} cases)."
            ),
            "",
            (
                "This is a preliminary ranking comparison only. It is not the "
                "final keep/reject decision because the graph candidate must "
                "still undergo the same calibration and capacity-constrained "
                "policy evaluation used by the Phase 5 baseline."
            ),
            "",
            "## Next evaluation requirement",
            "",
            (
                "Fit sigmoid calibration on only the earlier chronological "
                "validation calibration-fit period. Compare the baseline and "
                "graph candidate on the later policy-selection period using "
                "the same capacities and cost assumptions. Do not use the "
                "locked final test split during this decision."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def save_markdown_report(report: str) -> None:
    """Save the human-readable graph-ablation comparison report."""
    COMPARISON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with COMPARISON_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    print(f"Saved graph comparison report to: {COMPARISON_REPORT_PATH}")


def main() -> None:
    """Create the Phase 6 validation-only graph-ablation comparison."""
    baseline_metrics = load_metrics(BASELINE_METRICS_PATH)
    graph_metrics = load_metrics(GRAPH_METRICS_PATH)

    validate_comparison_inputs(
        baseline_metrics=baseline_metrics,
        graph_metrics=graph_metrics,
    )

    rows = build_comparison_rows(
        baseline_metrics=baseline_metrics,
        graph_metrics=graph_metrics,
    )

    save_comparison_csv(rows)

    report = build_markdown_report(
        baseline_metrics=baseline_metrics,
        graph_metrics=graph_metrics,
        rows=rows,
    )
    save_markdown_report(report)

    print("\nPhase 6 graph-feature ablation comparison completed.")
    print(
        "Fraud captured at 1,000 reviews: "
        f"{int(baseline_metrics['captured_fraud_count_at_k']):,} "
        "(baseline) vs "
        f"{int(graph_metrics['captured_fraud_count_at_k']):,} "
        "(XGBoost + graph features)"
    )
    print(
        "PR-AUC: "
        f"{float(baseline_metrics['pr_auc']):.6f} "
        "(baseline) vs "
        f"{float(graph_metrics['pr_auc']):.6f} "
        "(XGBoost + graph features)"
    )
    print("Final test split loaded: False")


if __name__ == "__main__":
    main()