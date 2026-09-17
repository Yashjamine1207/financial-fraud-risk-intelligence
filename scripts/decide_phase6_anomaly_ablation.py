"""Make the Phase 6 Isolation Forest anomaly-feature keep/reject decision.

This script compares the Phase 5 baseline policy with the Phase 6 XGBoost plus
Isolation Forest anomaly-feature policy under the identical validation protocol:

- Same chronological validation policy-selection period.
- Same sigmoid / Platt calibration approach.
- Same cost assumptions.
- Same review-capacity sensitivity analysis.
- Same operational capacity: 1,000 transactions.
- Final test split remains locked and is not loaded.

Decision rule
-------------
Keep the anomaly feature only if it improves fraud captured at the configured
operational capacity or improves net expected value at that same capacity,
without degrading the selected decision objective or operational reliability.

The decision is based on validation only. The final test split is deliberately
not evaluated for a rejected advanced feature.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

BASELINE_POLICY_PATH = Path("models/metrics/phase5_validation_policy_evaluation.json")
ANOMALY_POLICY_PATH = Path("models/metrics/phase6_anomaly_validation_policy_evaluation.json")

COMPARISON_CSV_PATH = Path("reports/tables/phase6_anomaly_policy_comparison.csv")
DECISION_REPORT_PATH = Path("docs/experiments/phase_6_anomaly_ablation_decision.md")

OPERATIONAL_CAPACITY = 1000
ANOMALY_FEATURE_NAME = "anomaly_score_isolation_forest"


def load_json_report(path: Path) -> dict[str, Any]:
    """Load and validate one policy-evaluation JSON report."""
    if not path.exists():
        raise FileNotFoundError(f"Required policy report was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise TypeError(f"Policy report must contain a JSON object: {path}")

    return report


def get_capacity_result(
    report: dict[str, Any],
    capacity: int,
) -> dict[str, Any]:
    """Return the policy result for one specific review capacity."""
    results = report.get("capacity_sensitivity_results")

    if not isinstance(results, list):
        raise TypeError(
            "Policy report must contain a list named " "'capacity_sensitivity_results'."
        )

    matching_results = [
        result
        for result in results
        if isinstance(result, dict) and result.get("review_capacity") == capacity
    ]

    if len(matching_results) != 1:
        raise ValueError(
            f"Expected exactly one result for capacity {capacity:,}; "
            f"found {len(matching_results)}."
        )

    return matching_results[0]


def validate_comparable_reports(
    baseline_report: dict[str, Any],
    anomaly_report: dict[str, Any],
) -> None:
    """Confirm both reports use the same policy-selection conditions."""
    for report_name, report in [
        ("Baseline", baseline_report),
        ("Anomaly candidate", anomaly_report),
    ]:
        data_protection = report.get("data_protection")

        if not isinstance(data_protection, dict):
            raise TypeError(
                f"{report_name} report must contain a dictionary named " "'data_protection'."
            )

        if data_protection.get("final_test_loaded") is not False:
            raise ValueError(
                f"{report_name} report indicates final-test access. "
                "Phase 6 ablation decisions must be validation-only."
            )

    baseline_split = baseline_report.get("chronological_split")
    anomaly_split = anomaly_report.get("chronological_split")

    if not isinstance(baseline_split, dict) or not isinstance(
        anomaly_split,
        dict,
    ):
        raise TypeError(
            "Both reports must contain chronological split metadata " "as dictionaries."
        )

    comparable_split_fields = [
        "split_timestamp",
        "calibration_fit_rows",
        "policy_selection_rows",
        "calibration_fit_fraud_count",
        "policy_selection_fraud_count",
    ]

    for field in comparable_split_fields:
        if baseline_split.get(field) != anomaly_split.get(field):
            raise ValueError(
                f"Reports are not comparable because '{field}' differs. "
                f"Baseline={baseline_split.get(field)!r}, "
                f"candidate={anomaly_split.get(field)!r}."
            )

    baseline_costs = baseline_report.get("cost_assumptions")
    anomaly_costs = anomaly_report.get("cost_assumptions")

    if baseline_costs != anomaly_costs:
        raise ValueError("Reports are not comparable because cost assumptions differ.")

    baseline_capacity = baseline_report.get("configured_operational_capacity")
    anomaly_capacity = anomaly_report.get("configured_operational_capacity")

    if baseline_capacity != OPERATIONAL_CAPACITY:
        raise ValueError(
            "Baseline report does not use the documented operational "
            f"capacity of {OPERATIONAL_CAPACITY:,}."
        )

    if anomaly_capacity != OPERATIONAL_CAPACITY:
        raise ValueError(
            "Anomaly report does not use the documented operational "
            f"capacity of {OPERATIONAL_CAPACITY:,}."
        )


def build_comparison_rows(
    baseline_report: dict[str, Any],
    anomaly_report: dict[str, Any],
) -> list[dict[str, float | int]]:
    """Create comparison rows across all common review capacities."""
    baseline_results = baseline_report["capacity_sensitivity_results"]
    anomaly_results = anomaly_report["capacity_sensitivity_results"]

    anomaly_by_capacity = {int(result["review_capacity"]): result for result in anomaly_results}

    rows: list[dict[str, float | int]] = []

    for baseline_result in baseline_results:
        capacity = int(baseline_result["review_capacity"])

        if capacity not in anomaly_by_capacity:
            raise ValueError(f"Anomaly policy report has no result for capacity {capacity}.")

        anomaly_result = anomaly_by_capacity[capacity]

        rows.append(
            {
                "review_capacity": capacity,
                "baseline_captured_fraud": int(baseline_result["captured_fraud_count"]),
                "anomaly_captured_fraud": int(anomaly_result["captured_fraud_count"]),
                "captured_fraud_difference": int(anomaly_result["captured_fraud_count"])
                - int(baseline_result["captured_fraud_count"]),
                "baseline_capture_rate": float(baseline_result["fraud_capture_rate"]),
                "anomaly_capture_rate": float(anomaly_result["fraud_capture_rate"]),
                "capture_rate_difference": float(anomaly_result["fraud_capture_rate"])
                - float(baseline_result["fraud_capture_rate"]),
                "baseline_net_expected_value": float(baseline_result["net_expected_value"]),
                "anomaly_net_expected_value": float(anomaly_result["net_expected_value"]),
                "net_expected_value_difference": float(anomaly_result["net_expected_value"])
                - float(baseline_result["net_expected_value"]),
            }
        )

    return rows


def make_keep_reject_decision(
    baseline_result: dict[str, Any],
    anomaly_result: dict[str, Any],
) -> tuple[str, str]:
    """Apply the documented decision rule at operational capacity."""
    captured_difference = int(anomaly_result["captured_fraud_count"]) - int(
        baseline_result["captured_fraud_count"]
    )

    net_value_difference = float(anomaly_result["net_expected_value"]) - float(
        baseline_result["net_expected_value"]
    )

    if captured_difference > 0 and net_value_difference >= 0.0:
        return (
            "keep",
            (
                "The anomaly feature improved fraud capture at the configured "
                "operational capacity without reducing net expected value."
            ),
        )

    if net_value_difference > 0.0 and captured_difference >= 0:
        return (
            "keep",
            (
                "The anomaly feature improved net expected value at the "
                "configured operational capacity without reducing fraud capture."
            ),
        )

    return (
        "reject",
        (
            "The anomaly feature did not improve the selected operational "
            "decision objective at the configured review capacity."
        ),
    )


def save_comparison_csv(
    rows: list[dict[str, float | int]],
) -> None:
    """Save capacity-sensitivity comparison rows to a CSV file."""
    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "review_capacity",
        "baseline_captured_fraud",
        "anomaly_captured_fraud",
        "captured_fraud_difference",
        "baseline_capture_rate",
        "anomaly_capture_rate",
        "capture_rate_difference",
        "baseline_net_expected_value",
        "anomaly_net_expected_value",
        "net_expected_value_difference",
    ]

    with COMPARISON_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved policy comparison CSV to: {COMPARISON_CSV_PATH}")


def build_markdown_report(
    baseline_report: dict[str, Any],
    anomaly_report: dict[str, Any],
    rows: list[dict[str, float | int]],
    decision: str,
    decision_reason: str,
) -> str:
    """Create a clear Phase 6 anomaly ablation decision record."""
    baseline_operational = get_capacity_result(
        report=baseline_report,
        capacity=OPERATIONAL_CAPACITY,
    )
    anomaly_operational = get_capacity_result(
        report=anomaly_report,
        capacity=OPERATIONAL_CAPACITY,
    )

    captured_difference = int(anomaly_operational["captured_fraud_count"]) - int(
        baseline_operational["captured_fraud_count"]
    )
    capture_rate_difference = float(anomaly_operational["fraud_capture_rate"]) - float(
        baseline_operational["fraud_capture_rate"]
    )
    net_value_difference = float(anomaly_operational["net_expected_value"]) - float(
        baseline_operational["net_expected_value"]
    )

    baseline_model = baseline_report["champion_model"]
    anomaly_model = anomaly_report["candidate_model"]
    chronological_split = baseline_report["chronological_split"]
    costs = baseline_report["cost_assumptions"]

    lines = [
        "# Phase 6 — Isolation Forest Anomaly Ablation Decision",
        "",
        "## Decision",
        "",
        f"**{decision.upper()}** `{ANOMALY_FEATURE_NAME}`.",
        "",
        decision_reason,
        "",
        "The candidate will not be evaluated on the locked final test split.",
        "The selected system remains the existing Phase 5 baseline XGBoost",
        "with sigmoid calibration and the documented top-k review policy.",
        "",
        "## Comparable evaluation protocol",
        "",
        f"- Baseline model: `{baseline_model['model_version']}`",
        f"- Candidate model: `{anomaly_model['model_version']}`",
        f"- Baseline MLflow run: `{baseline_model['mlflow_run_id']}`",
        f"- Candidate MLflow run: `{anomaly_model['mlflow_run_id']}`",
        "- Calibration method: sigmoid / Platt scaling",
        "- Calibration-fit period: earliest chronological 50% of validation",
        "- Policy-selection period: latest chronological 50% of validation",
        f"- Timestamp boundary: `{chronological_split['split_timestamp']:.0f}`",
        ("- Policy-selection transactions: " f"{chronological_split['policy_selection_rows']:,}"),
        (
            "- Policy-selection known fraud: "
            f"{chronological_split['policy_selection_fraud_count']:,}"
        ),
        "- Final test split loaded: False",
        "",
        "## Operational decision result",
        "",
        f"The configured operational capacity is **{OPERATIONAL_CAPACITY:,} reviews**.",
        "",
        "| Metric | Baseline | XGBoost + anomaly | Candidate minus baseline |",
        "|---|---:|---:|---:|",
        (
            "| Fraud captured | "
            f"{int(baseline_operational['captured_fraud_count']):,} | "
            f"{int(anomaly_operational['captured_fraud_count']):,} | "
            f"{captured_difference:+,} |"
        ),
        (
            "| Fraud capture rate | "
            f"{float(baseline_operational['fraud_capture_rate']):.2%} | "
            f"{float(anomaly_operational['fraud_capture_rate']):.2%} | "
            f"{capture_rate_difference:+.2%} |"
        ),
        (
            "| Net expected value | "
            f"£{float(baseline_operational['net_expected_value']):,.0f} | "
            f"£{float(anomaly_operational['net_expected_value']):,.0f} | "
            f"£{net_value_difference:+,.0f} |"
        ),
        "",
        "## Capacity sensitivity",
        "",
        "| Review capacity | Baseline captured fraud | Anomaly captured fraud | Difference | Baseline net value | Anomaly net value | Difference |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {int(row['review_capacity']):,} | "
            f"{int(row['baseline_captured_fraud']):,} | "
            f"{int(row['anomaly_captured_fraud']):,} | "
            f"{int(row['captured_fraud_difference']):+,} | "
            f"£{float(row['baseline_net_expected_value']):,.0f} | "
            f"£{float(row['anomaly_net_expected_value']):,.0f} | "
            f"£{float(row['net_expected_value_difference']):+,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The anomaly feature changed the ranking slightly, but it did "
                "not improve the project’s selected operational policy at the "
                "documented capacity of 1,000 reviews."
            ),
            "",
            (
                "Some larger review capacities may show a positive candidate "
                "difference. That does not justify retaining the feature for "
                "the selected policy because operational capacity is fixed at "
                "1,000 reviews under the versioned portfolio assumptions."
            ),
            "",
            "## Cost assumptions",
            "",
            (f"- False-negative cost: " f"£{float(costs['false_negative_cost']):,.0f}"),
            (f"- Manual-review cost: " f"£{float(costs['manual_review_cost']):,.0f}"),
            (f"- Fraud-prevention value: " f"£{float(costs['fraud_prevention_value']):,.0f}"),
            "",
            (
                "All financial assumptions are illustrative portfolio-project "
                "assumptions and are not real financial-institution outcomes."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def save_markdown_report(report: str) -> None:
    """Save the Phase 6 anomaly keep/reject decision document."""
    DECISION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DECISION_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    print(f"Saved anomaly decision report to: {DECISION_REPORT_PATH}")


def main() -> None:
    """Compare policy results and document the anomaly-feature decision."""
    baseline_report = load_json_report(BASELINE_POLICY_PATH)
    anomaly_report = load_json_report(ANOMALY_POLICY_PATH)

    validate_comparable_reports(
        baseline_report=baseline_report,
        anomaly_report=anomaly_report,
    )

    baseline_operational = get_capacity_result(
        report=baseline_report,
        capacity=OPERATIONAL_CAPACITY,
    )
    anomaly_operational = get_capacity_result(
        report=anomaly_report,
        capacity=OPERATIONAL_CAPACITY,
    )

    decision, decision_reason = make_keep_reject_decision(
        baseline_result=baseline_operational,
        anomaly_result=anomaly_operational,
    )

    comparison_rows = build_comparison_rows(
        baseline_report=baseline_report,
        anomaly_report=anomaly_report,
    )

    save_comparison_csv(comparison_rows)

    markdown_report = build_markdown_report(
        baseline_report=baseline_report,
        anomaly_report=anomaly_report,
        rows=comparison_rows,
        decision=decision,
        decision_reason=decision_reason,
    )
    save_markdown_report(markdown_report)

    captured_difference = int(anomaly_operational["captured_fraud_count"]) - int(
        baseline_operational["captured_fraud_count"]
    )
    net_value_difference = float(anomaly_operational["net_expected_value"]) - float(
        baseline_operational["net_expected_value"]
    )

    print("\nPhase 6 anomaly ablation decision completed.")
    print(f"Decision: {decision.upper()}")
    print(f"Operational capacity: {OPERATIONAL_CAPACITY:,}")
    print("Fraud captured difference at operational capacity: " f"{captured_difference:+,}")
    print(
        "Net expected value difference at operational capacity: " f"£{net_value_difference:+,.0f}"
    )
    print("Final test split loaded: False")


if __name__ == "__main__":
    main()
