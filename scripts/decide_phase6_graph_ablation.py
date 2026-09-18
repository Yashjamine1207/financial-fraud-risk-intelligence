"""Make the Phase 6 time-safe graph-feature keep/reject decision.

This script compares the Phase 5 baseline policy with the Phase 6 XGBoost plus
time-safe graph-feature policy under identical validation conditions:

- Same chronological validation policy-selection period.
- Same sigmoid / Platt calibration approach.
- Same cost assumptions.
- Same review-capacity sensitivity analysis.
- Same operational capacity: 1,000 transactions.
- Final test split remains locked and is not loaded.

Decision rule
-------------
Keep graph features only if they improve fraud captured at the configured
operational capacity or improve net expected value at that same capacity,
without degrading the selected decision objective.

The decision is based on validation only. A retained graph candidate remains
eligible for later final-model selection, but it is not evaluated on the final
test split until all Phase 6 advanced-method choices are complete.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

BASELINE_POLICY_PATH = Path("models/metrics/phase5_validation_policy_evaluation.json")
GRAPH_POLICY_PATH = Path("models/metrics/phase6_graph_validation_policy_evaluation.json")

COMPARISON_CSV_PATH = Path("reports/tables/phase6_graph_policy_comparison.csv")
DECISION_REPORT_PATH = Path("docs/experiments/phase_6_graph_ablation_decision.md")

OPERATIONAL_CAPACITY = 1000
GRAPH_FEATURE_VERSION = "graph-features-v1.0.0"
GRAPH_FEATURE_COUNT = 10


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
    """Return exactly one policy result for the requested review capacity."""
    results = report.get("capacity_sensitivity_results")

    if not isinstance(results, list):
        raise TypeError(
            "Policy report must contain a list named "
            "'capacity_sensitivity_results'."
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
    graph_report: dict[str, Any],
) -> None:
    """Confirm baseline and graph reports use the same policy conditions."""
    for report_name, report in [
        ("Baseline", baseline_report),
        ("Graph candidate", graph_report),
    ]:
        data_protection = report.get("data_protection")

        if not isinstance(data_protection, dict):
            raise TypeError(
                f"{report_name} report must contain a dictionary named "
                "'data_protection'."
            )

        if data_protection.get("final_test_loaded") is not False:
            raise ValueError(
                f"{report_name} report indicates final-test access. "
                "Phase 6 ablation decisions must be validation-only."
            )

    baseline_split = baseline_report.get("chronological_split")
    graph_split = graph_report.get("chronological_split")

    if not isinstance(baseline_split, dict) or not isinstance(graph_split, dict):
        raise TypeError(
            "Both reports must contain chronological split metadata as dictionaries."
        )

    comparable_split_fields = [
        "split_timestamp",
        "calibration_fit_rows",
        "policy_selection_rows",
        "calibration_fit_fraud_count",
        "policy_selection_fraud_count",
    ]

    for field in comparable_split_fields:
        if baseline_split.get(field) != graph_split.get(field):
            raise ValueError(
                f"Reports are not comparable because '{field}' differs. "
                f"Baseline={baseline_split.get(field)!r}, "
                f"graph_candidate={graph_split.get(field)!r}."
            )

    baseline_costs = baseline_report.get("cost_assumptions")
    graph_costs = graph_report.get("cost_assumptions")

    if baseline_costs != graph_costs:
        raise ValueError("Reports are not comparable because cost assumptions differ.")

    baseline_capacity = baseline_report.get("configured_operational_capacity")
    graph_capacity = graph_report.get("configured_operational_capacity")

    if baseline_capacity != OPERATIONAL_CAPACITY:
        raise ValueError(
            "Baseline report does not use the documented operational capacity "
            f"of {OPERATIONAL_CAPACITY:,}."
        )

    if graph_capacity != OPERATIONAL_CAPACITY:
        raise ValueError(
            "Graph report does not use the documented operational capacity "
            f"of {OPERATIONAL_CAPACITY:,}."
        )

    graph_features = graph_report.get("graph_features")

    if not isinstance(graph_features, dict):
        raise TypeError("Graph report must contain graph feature metadata.")

    if graph_features.get("feature_version") != GRAPH_FEATURE_VERSION:
        raise ValueError(
            "Graph report uses an unexpected graph feature version. "
            f"Expected '{GRAPH_FEATURE_VERSION}'."
        )

    if graph_features.get("feature_count") != GRAPH_FEATURE_COUNT:
        raise ValueError(
            f"Graph report must contain exactly {GRAPH_FEATURE_COUNT} graph features."
        )


def build_comparison_rows(
    baseline_report: dict[str, Any],
    graph_report: dict[str, Any],
) -> list[dict[str, float | int]]:
    """Create capacity-sensitivity comparison rows."""
    baseline_results = baseline_report["capacity_sensitivity_results"]
    graph_results = graph_report["capacity_sensitivity_results"]

    graph_by_capacity = {
        int(result["review_capacity"]): result
        for result in graph_results
    }

    rows: list[dict[str, float | int]] = []

    for baseline_result in baseline_results:
        capacity = int(baseline_result["review_capacity"])

        if capacity not in graph_by_capacity:
            raise ValueError(
                f"Graph policy report has no result for capacity {capacity}."
            )

        graph_result = graph_by_capacity[capacity]

        rows.append(
            {
                "review_capacity": capacity,
                "baseline_captured_fraud": int(
                    baseline_result["captured_fraud_count"]
                ),
                "graph_captured_fraud": int(
                    graph_result["captured_fraud_count"]
                ),
                "captured_fraud_difference": int(
                    graph_result["captured_fraud_count"]
                )
                - int(baseline_result["captured_fraud_count"]),
                "baseline_capture_rate": float(
                    baseline_result["fraud_capture_rate"]
                ),
                "graph_capture_rate": float(
                    graph_result["fraud_capture_rate"]
                ),
                "capture_rate_difference": float(
                    graph_result["fraud_capture_rate"]
                )
                - float(baseline_result["fraud_capture_rate"]),
                "baseline_net_expected_value": float(
                    baseline_result["net_expected_value"]
                ),
                "graph_net_expected_value": float(
                    graph_result["net_expected_value"]
                ),
                "net_expected_value_difference": float(
                    graph_result["net_expected_value"]
                )
                - float(baseline_result["net_expected_value"]),
            }
        )

    return rows


def make_keep_reject_decision(
    baseline_result: dict[str, Any],
    graph_result: dict[str, Any],
) -> tuple[str, str]:
    """Apply the documented Phase 6 decision rule at capacity 1,000."""
    captured_difference = int(graph_result["captured_fraud_count"]) - int(
        baseline_result["captured_fraud_count"]
    )

    net_value_difference = float(graph_result["net_expected_value"]) - float(
        baseline_result["net_expected_value"]
    )

    if captured_difference > 0 and net_value_difference >= 0.0:
        return (
            "keep",
            (
                "The graph features improved fraud capture at the configured "
                "operational capacity without reducing net expected value."
            ),
        )

    if net_value_difference > 0.0 and captured_difference >= 0:
        return (
            "keep",
            (
                "The graph features improved net expected value at the "
                "configured operational capacity without reducing fraud capture."
            ),
        )

    return (
        "reject",
        (
            "The graph features did not improve the selected operational "
            "decision objective at the configured review capacity."
        ),
    )


def save_comparison_csv(rows: list[dict[str, float | int]]) -> None:
    """Save the graph policy-comparison table."""
    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "review_capacity",
        "baseline_captured_fraud",
        "graph_captured_fraud",
        "captured_fraud_difference",
        "baseline_capture_rate",
        "graph_capture_rate",
        "capture_rate_difference",
        "baseline_net_expected_value",
        "graph_net_expected_value",
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

    print(f"Saved graph policy comparison CSV to: {COMPARISON_CSV_PATH}")


def build_markdown_report(
    baseline_report: dict[str, Any],
    graph_report: dict[str, Any],
    rows: list[dict[str, float | int]],
    decision: str,
    decision_reason: str,
) -> str:
    """Create the Phase 6 graph-feature ablation decision record."""
    baseline_operational = get_capacity_result(
        report=baseline_report,
        capacity=OPERATIONAL_CAPACITY,
    )
    graph_operational = get_capacity_result(
        report=graph_report,
        capacity=OPERATIONAL_CAPACITY,
    )

    captured_difference = int(graph_operational["captured_fraud_count"]) - int(
        baseline_operational["captured_fraud_count"]
    )
    capture_rate_difference = float(graph_operational["fraud_capture_rate"]) - float(
        baseline_operational["fraud_capture_rate"]
    )
    net_value_difference = float(graph_operational["net_expected_value"]) - float(
        baseline_operational["net_expected_value"]
    )

    baseline_model = baseline_report["champion_model"]
    graph_model = graph_report["candidate_model"]
    chronological_split = baseline_report["chronological_split"]
    graph_features = graph_report["graph_features"]

    if decision == "keep":
        next_step = (
            "The graph candidate is retained as an eligible model candidate. "
            "It will not be evaluated on the locked final test split yet because "
            "the compact LSTM/GRU sequence ablation remains to be completed."
        )
    else:
        next_step = (
            "The graph candidate is rejected and will not be evaluated on the "
            "locked final test split. The selected system remains the existing "
            "Phase 5 baseline XGBoost while the compact LSTM/GRU sequence "
            "ablation is evaluated."
        )

    lines = [
        "# Phase 6 - Time-Safe Graph-Feature Ablation Decision",
        "",
        "## Decision",
        "",
        f"**{decision.upper()}** time-safe graph features.",
        "",
        decision_reason,
        "",
        next_step,
        "",
        "## Comparable evaluation protocol",
        "",
        f"- Baseline model: `{baseline_model['model_version']}`",
        f"- Candidate model: `{graph_model['model_version']}`",
        f"- Baseline MLflow run: `{baseline_model['mlflow_run_id']}`",
        f"- Candidate MLflow run: `{graph_model['mlflow_run_id']}`",
        f"- Graph feature version: `{graph_features['feature_version']}`",
        f"- Graph feature count: {graph_features['feature_count']}",
        "- Calibration method: sigmoid / Platt scaling",
        "- Calibration-fit period: earliest chronological 50% of validation",
        "- Policy-selection period: latest chronological 50% of validation",
        f"- Timestamp boundary: `{chronological_split['split_timestamp']:.0f}`",
        (
            "- Policy-selection transactions: "
            f"{chronological_split['policy_selection_rows']:,}"
        ),
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
        (
            "| Metric | Baseline | XGBoost + graph features | "
            "Candidate minus baseline |"
        ),
        "|---|---:|---:|---:|",
        (
            "| Fraud captured | "
            f"{int(baseline_operational['captured_fraud_count']):,} | "
            f"{int(graph_operational['captured_fraud_count']):,} | "
            f"{captured_difference:+,} |"
        ),
        (
            "| Fraud capture rate | "
            f"{float(baseline_operational['fraud_capture_rate']):.2%} | "
            f"{float(graph_operational['fraud_capture_rate']):.2%} | "
            f"{capture_rate_difference:+.2%} |"
        ),
        (
            "| Net expected value | "
            f"GBP {float(baseline_operational['net_expected_value']):,.0f} | "
            f"GBP {float(graph_operational['net_expected_value']):,.0f} | "
            f"GBP {net_value_difference:+,.0f} |"
        ),
        "",
        "## Capacity sensitivity",
        "",
        (
            "| Review capacity | Baseline captured fraud | Graph captured fraud | "
            "Difference | Baseline net value | Graph net value | Difference |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {int(row['review_capacity']):,} | "
            f"{int(row['baseline_captured_fraud']):,} | "
            f"{int(row['graph_captured_fraud']):,} | "
            f"{int(row['captured_fraud_difference']):+,} | "
            f"GBP {float(row['baseline_net_expected_value']):,.0f} | "
            f"GBP {float(row['graph_net_expected_value']):,.0f} | "
            f"GBP {float(row['net_expected_value_difference']):+,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Graph features represent historical entity-relationship "
                "structure. They contributed to model ranking but do not prove "
                "that a card, device, email domain, relationship, or graph "
                "component caused fraud."
            ),
            "",
            (
                "All financial figures use illustrative portfolio-project "
                "assumptions and do not represent real financial-institution "
                "outcomes or realised savings."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def save_markdown_report(report: str) -> None:
    """Save the human-readable graph decision report."""
    DECISION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DECISION_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    print(f"Saved graph decision report to: {DECISION_REPORT_PATH}")


def main() -> None:
    """Compare graph policy results and document the keep/reject decision."""
    baseline_report = load_json_report(BASELINE_POLICY_PATH)
    graph_report = load_json_report(GRAPH_POLICY_PATH)

    validate_comparable_reports(
        baseline_report=baseline_report,
        graph_report=graph_report,
    )

    baseline_operational = get_capacity_result(
        report=baseline_report,
        capacity=OPERATIONAL_CAPACITY,
    )
    graph_operational = get_capacity_result(
        report=graph_report,
        capacity=OPERATIONAL_CAPACITY,
    )

    decision, decision_reason = make_keep_reject_decision(
        baseline_result=baseline_operational,
        graph_result=graph_operational,
    )

    comparison_rows = build_comparison_rows(
        baseline_report=baseline_report,
        graph_report=graph_report,
    )

    save_comparison_csv(comparison_rows)

    markdown_report = build_markdown_report(
        baseline_report=baseline_report,
        graph_report=graph_report,
        rows=comparison_rows,
        decision=decision,
        decision_reason=decision_reason,
    )
    save_markdown_report(markdown_report)

    captured_difference = int(graph_operational["captured_fraud_count"]) - int(
        baseline_operational["captured_fraud_count"]
    )
    net_value_difference = float(graph_operational["net_expected_value"]) - float(
        baseline_operational["net_expected_value"]
    )

    print("\nPhase 6 graph-feature ablation decision completed.")
    print(f"Decision: {decision.upper()}")
    print(f"Operational capacity: {OPERATIONAL_CAPACITY:,}")
    print(
        "Fraud captured difference at operational capacity: "
        f"{captured_difference:+,}"
    )
    print(
        "Net expected value difference at operational capacity: "
        f"GBP {net_value_difference:+,.0f}"
    )
    print("Final test split loaded: False")


if __name__ == "__main__":
    main()
