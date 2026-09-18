"""Document the Phase 6 compact GRU/LSTM sequence-ablation decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

BASELINE_POLICY_PATH = Path("models/metrics/phase5_validation_policy_evaluation.json")
LSTM_POLICY_PATH = Path(
    "models/metrics/phase6_lstm_sequence_validation_policy_evaluation.json"
)

GRU_METRICS_PATH = Path(
    "reports/tables/phase6_gru_sequence_validation_metrics.json"
)
LSTM_METRICS_PATH = Path(
    "reports/tables/phase6_lstm_sequence_validation_metrics.json"
)

POLICY_COMPARISON_CSV_PATH = Path(
    "reports/tables/phase6_sequence_policy_comparison.csv"
)
MODEL_COMPARISON_CSV_PATH = Path(
    "reports/tables/phase6_sequence_model_comparison.csv"
)
DECISION_REPORT_PATH = Path(
    "docs/experiments/phase_6_sequence_ablation_decision.md"
)

OPERATIONAL_CAPACITY = 1000


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate one JSON object."""
    if not path.exists():
        raise FileNotFoundError(f"Required report was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise TypeError(f"Expected a JSON object in: {path}")

    return report


def get_capacity_result(report: dict[str, Any], capacity: int) -> dict[str, Any]:
    """Return exactly one policy result at the requested capacity."""
    results = report.get("capacity_sensitivity_results")

    if not isinstance(results, list):
        raise TypeError(
            "Policy report must include a list named 'capacity_sensitivity_results'."
        )

    matched = [
        result
        for result in results
        if isinstance(result, dict) and result.get("review_capacity") == capacity
    ]

    if len(matched) != 1:
        raise ValueError(
            f"Expected one policy result at capacity {capacity:,}; found {len(matched)}."
        )

    return matched[0]


def validate_policy_reports(
    baseline_report: dict[str, Any],
    lstm_report: dict[str, Any],
) -> None:
    """Confirm the baseline and LSTM policy results are comparable."""
    for report_name, report in [
        ("Baseline", baseline_report),
        ("LSTM candidate", lstm_report),
    ]:
        protection = report.get("data_protection")

        if not isinstance(protection, dict):
            raise TypeError(
                f"{report_name} report must contain 'data_protection' metadata."
            )

        if protection.get("final_test_loaded") is not False:
            raise ValueError(
                f"{report_name} report indicates final-test access. "
                "The Phase 6 decision must use validation only."
            )

    baseline_split = baseline_report.get("chronological_split")
    lstm_split = lstm_report.get("chronological_split")

    if not isinstance(baseline_split, dict) or not isinstance(lstm_split, dict):
        raise TypeError("Both reports require chronological split metadata.")

    required_split_fields = [
        "split_timestamp",
        "calibration_fit_rows",
        "policy_selection_rows",
        "calibration_fit_fraud_count",
        "policy_selection_fraud_count",
    ]

    for field in required_split_fields:
        if baseline_split.get(field) != lstm_split.get(field):
            raise ValueError(
                f"Policy reports are not comparable because '{field}' differs. "
                f"Baseline={baseline_split.get(field)!r}, "
                f"LSTM={lstm_split.get(field)!r}."
            )

    if baseline_report.get("cost_assumptions") != lstm_report.get("cost_assumptions"):
        raise ValueError("Policy reports use different cost assumptions.")

    if baseline_report.get("configured_operational_capacity") != OPERATIONAL_CAPACITY:
        raise ValueError("Baseline does not use the configured 1,000-review capacity.")

    if lstm_report.get("configured_operational_capacity") != OPERATIONAL_CAPACITY:
        raise ValueError("LSTM does not use the configured 1,000-review capacity.")


def validate_sequence_metrics(
    gru_metrics: dict[str, Any],
    lstm_metrics: dict[str, Any],
) -> None:
    """Confirm GRU and LSTM preliminary ranking metrics are comparable."""
    required_fields = [
        "dataset_version",
        "feature_version",
        "target_encoding_version",
        "review_capacity",
        "total_fraud_count",
        "pr_auc",
        "roc_auc",
        "precision_at_k",
        "recall_at_k",
        "captured_fraud_count_at_k",
        "test_split_loaded",
    ]

    for report_name, report in [
        ("GRU", gru_metrics),
        ("LSTM", lstm_metrics),
    ]:
        for field in required_fields:
            if field not in report:
                raise ValueError(f"{report_name} metrics are missing '{field}'.")

        if report["test_split_loaded"] is not False:
            raise ValueError(
                f"{report_name} metrics indicate final-test access."
            )

    comparable_fields = [
        "dataset_version",
        "feature_version",
        "target_encoding_version",
        "review_capacity",
        "total_fraud_count",
    ]

    for field in comparable_fields:
        if gru_metrics[field] != lstm_metrics[field]:
            raise ValueError(
                f"GRU and LSTM are not comparable because '{field}' differs."
            )

    if gru_metrics["review_capacity"] != OPERATIONAL_CAPACITY:
        raise ValueError("Sequence metrics must use the 1,000-review capacity.")


def build_model_comparison_rows(
    gru_metrics: dict[str, Any],
    lstm_metrics: dict[str, Any],
) -> list[dict[str, float | int | str]]:
    """Compare the two compact sequence architectures on validation ranking."""
    metric_definitions = [
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("precision_at_k", "Precision@1,000"),
        ("recall_at_k", "Recall@1,000"),
        ("captured_fraud_count_at_k", "Fraud captured at 1,000 reviews"),
        ("training_time_seconds", "Training time (seconds)"),
        ("epochs_completed", "Epochs completed"),
    ]

    rows: list[dict[str, float | int | str]] = []

    for metric_key, metric_name in metric_definitions:
        gru_value = float(gru_metrics[metric_key])
        lstm_value = float(lstm_metrics[metric_key])

        rows.append(
            {
                "metric_key": metric_key,
                "metric": metric_name,
                "gru": gru_value,
                "lstm": lstm_value,
                "lstm_minus_gru": lstm_value - gru_value,
            }
        )

    return rows


def build_policy_comparison_rows(
    baseline_report: dict[str, Any],
    lstm_report: dict[str, Any],
) -> list[dict[str, float | int]]:
    """Compare baseline and best recurrent candidate across capacities."""
    lstm_by_capacity = {
        int(result["review_capacity"]): result
        for result in lstm_report["capacity_sensitivity_results"]
    }

    rows: list[dict[str, float | int]] = []

    for baseline_result in baseline_report["capacity_sensitivity_results"]:
        capacity = int(baseline_result["review_capacity"])

        if capacity not in lstm_by_capacity:
            raise ValueError(f"LSTM has no policy result at capacity {capacity}.")

        lstm_result = lstm_by_capacity[capacity]

        rows.append(
            {
                "review_capacity": capacity,
                "baseline_captured_fraud": int(
                    baseline_result["captured_fraud_count"]
                ),
                "lstm_captured_fraud": int(lstm_result["captured_fraud_count"]),
                "captured_fraud_difference": int(
                    lstm_result["captured_fraud_count"]
                )
                - int(baseline_result["captured_fraud_count"]),
                "baseline_capture_rate": float(
                    baseline_result["fraud_capture_rate"]
                ),
                "lstm_capture_rate": float(lstm_result["fraud_capture_rate"]),
                "capture_rate_difference": float(
                    lstm_result["fraud_capture_rate"]
                )
                - float(baseline_result["fraud_capture_rate"]),
                "baseline_net_expected_value": float(
                    baseline_result["net_expected_value"]
                ),
                "lstm_net_expected_value": float(
                    lstm_result["net_expected_value"]
                ),
                "net_expected_value_difference": float(
                    lstm_result["net_expected_value"]
                )
                - float(baseline_result["net_expected_value"]),
            }
        )

    return rows


def save_csv(
    rows: list[dict[str, float | int | str]],
    path: Path,
) -> None:
    """Save one comparison table."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved comparison CSV to: {path}")


def build_markdown_report(
    baseline_report: dict[str, Any],
    lstm_report: dict[str, Any],
    gru_metrics: dict[str, Any],
    lstm_metrics: dict[str, Any],
    model_rows: list[dict[str, float | int | str]],
    policy_rows: list[dict[str, float | int]],
) -> str:
    """Build the final compact-sequence ablation report."""
    baseline_operational = get_capacity_result(
        baseline_report,
        OPERATIONAL_CAPACITY,
    )
    lstm_operational = get_capacity_result(
        lstm_report,
        OPERATIONAL_CAPACITY,
    )

    captured_difference = int(lstm_operational["captured_fraud_count"]) - int(
        baseline_operational["captured_fraud_count"]
    )
    net_value_difference = float(lstm_operational["net_expected_value"]) - float(
        baseline_operational["net_expected_value"]
    )

    baseline_model = baseline_report["champion_model"]
    lstm_model = lstm_report["candidate_model"]
    split = baseline_report["chronological_split"]

    lines = [
        "# Phase 6 - Compact LSTM and GRU Sequence Ablation Decision",
        "",
        "## Decision",
        "",
        "**REJECT** both compact GRU and compact LSTM sequence models.",
        "",
        (
            "The better recurrent candidate, the compact LSTM, failed to "
            "improve fraud capture or illustrative net expected value at the "
            "configured operational capacity of 1,000 reviews."
        ),
        "",
        (
            "The compact GRU was also rejected. It was inferior to the LSTM in "
            "validation PR-AUC and fraud captured at 1,000 reviews, so it was "
            "not advanced to a redundant calibrated-policy evaluation."
        ),
        "",
        (
            "Neither sequence candidate will be evaluated on the locked final "
            "test split. The Phase 5 calibrated XGBoost baseline remains the "
            "selected champion."
        ),
        "",
        "## Sequence protocol",
        "",
        "- Framework: TensorFlow / Keras",
        "- Entity sequence key: `card1`",
        "- Maximum sequence length: 10 transactions",
        "- Per-event inputs: log transaction amount and log prior card time gap",
        "- Same-timestamp rule: score all tied transactions before history updates",
        "- Training history: strictly earlier chronological card1 transactions",
        "- Validation history: training history plus earlier validation transactions",
        "- Fraud labels used as sequence inputs: False",
        "- Final test split loaded: False",
        "",
        "## Preliminary architecture comparison",
        "",
        "| Metric | Compact GRU | Compact LSTM | LSTM minus GRU |",
        "|---|---:|---:|---:|",
    ]

    formatting = {
        "pr_auc": ".6f",
        "roc_auc": ".6f",
        "precision_at_k": ".6f",
        "recall_at_k": ".6f",
        "captured_fraud_count_at_k": ".0f",
        "training_time_seconds": ".2f",
        "epochs_completed": ".0f",
    }

    for row in model_rows:
        format_specifier = formatting[str(row["metric_key"])]
        lines.append(
            f"| {row['metric']} | "
            f"{float(row['gru']):{format_specifier}} | "
            f"{float(row['lstm']):{format_specifier}} | "
            f"{float(row['lstm_minus_gru']):+{format_specifier}} |"
        )

    lines.extend(
        [
            "",
            "## Comparable calibrated policy result",
            "",
            f"- Baseline model: `{baseline_model['model_version']}`",
            f"- LSTM candidate: `{lstm_model['model_version']}`",
            "- Calibration: sigmoid / Platt scaling",
            "- Calibration period: earliest chronological 50% of validation",
            "- Policy period: latest chronological 50% of validation",
            f"- Timestamp boundary: `{split['split_timestamp']:.0f}`",
            (
                "- Policy-selection transactions: "
                f"{split['policy_selection_rows']:,}"
            ),
            (
                "- Policy-selection known fraud: "
                f"{split['policy_selection_fraud_count']:,}"
            ),
            "",
            f"The operating capacity is **{OPERATIONAL_CAPACITY:,} reviews**.",
            "",
            "| Metric | Baseline XGBoost | Compact LSTM | LSTM minus baseline |",
            "|---|---:|---:|---:|",
            (
                "| Fraud captured | "
                f"{int(baseline_operational['captured_fraud_count']):,} | "
                f"{int(lstm_operational['captured_fraud_count']):,} | "
                f"{captured_difference:+,} |"
            ),
            (
                "| Fraud capture rate | "
                f"{float(baseline_operational['fraud_capture_rate']):.2%} | "
                f"{float(lstm_operational['fraud_capture_rate']):.2%} | "
                f"{float(lstm_operational['fraud_capture_rate']) - float(baseline_operational['fraud_capture_rate']):+.2%} |"
            ),
            (
                "| Net expected value | "
                f"GBP {float(baseline_operational['net_expected_value']):,.0f} | "
                f"GBP {float(lstm_operational['net_expected_value']):,.0f} | "
                f"GBP {net_value_difference:+,.0f} |"
            ),
            "",
            "## Capacity sensitivity",
            "",
            (
                "| Capacity | Baseline captured | LSTM captured | Difference | "
                "Baseline net value | LSTM net value | Difference |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in policy_rows:
        lines.append(
            f"| {int(row['review_capacity']):,} | "
            f"{int(row['baseline_captured_fraud']):,} | "
            f"{int(row['lstm_captured_fraud']):,} | "
            f"{int(row['captured_fraud_difference']):+,} | "
            f"GBP {float(row['baseline_net_expected_value']):,.0f} | "
            f"GBP {float(row['lstm_net_expected_value']):,.0f} | "
            f"GBP {float(row['net_expected_value_difference']):+,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The compact recurrent models did not provide sufficient "
                "incremental fraud-prioritisation value relative to the "
                "existing tabular XGBoost model and its engineered "
                "point-in-time behavioural features."
            ),
            "",
            (
                "This result does not imply that sequence models are generally "
                "ineffective for fraud detection. It applies to this compact, "
                "two-input, card1-history experiment under this chronological "
                "IEEE-CIS benchmark and documented policy assumptions."
            ),
            "",
            (
                "All financial values use illustrative portfolio assumptions "
                "and do not represent real financial-institution savings."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Create the Phase 6 compact sequence-model decision artifacts."""
    baseline_report = load_json(BASELINE_POLICY_PATH)
    lstm_report = load_json(LSTM_POLICY_PATH)
    gru_metrics = load_json(GRU_METRICS_PATH)
    lstm_metrics = load_json(LSTM_METRICS_PATH)

    validate_policy_reports(
        baseline_report=baseline_report,
        lstm_report=lstm_report,
    )
    validate_sequence_metrics(
        gru_metrics=gru_metrics,
        lstm_metrics=lstm_metrics,
    )

    model_rows = build_model_comparison_rows(
        gru_metrics=gru_metrics,
        lstm_metrics=lstm_metrics,
    )
    policy_rows = build_policy_comparison_rows(
        baseline_report=baseline_report,
        lstm_report=lstm_report,
    )

    save_csv(model_rows, MODEL_COMPARISON_CSV_PATH)
    save_csv(policy_rows, POLICY_COMPARISON_CSV_PATH)

    report = build_markdown_report(
        baseline_report=baseline_report,
        lstm_report=lstm_report,
        gru_metrics=gru_metrics,
        lstm_metrics=lstm_metrics,
        model_rows=model_rows,
        policy_rows=policy_rows,
    )

    DECISION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DECISION_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    baseline_operational = get_capacity_result(
        baseline_report,
        OPERATIONAL_CAPACITY,
    )
    lstm_operational = get_capacity_result(
        lstm_report,
        OPERATIONAL_CAPACITY,
    )

    print("\nPhase 6 sequence ablation decision completed.")
    print("Decision: REJECT GRU and LSTM")
    print(f"Operational capacity: {OPERATIONAL_CAPACITY:,}")
    print(
        "Best sequence candidate captured-fraud difference: "
        f"{int(lstm_operational['captured_fraud_count']) - int(baseline_operational['captured_fraud_count']):+,}"
    )
    print(
        "Best sequence candidate net-value difference: "
        f"GBP {float(lstm_operational['net_expected_value']) - float(baseline_operational['net_expected_value']):+,.0f}"
    )
    print("Final test split loaded: False")
    print(f"Saved sequence decision report to: {DECISION_REPORT_PATH}")


if __name__ == "__main__":
    main()
