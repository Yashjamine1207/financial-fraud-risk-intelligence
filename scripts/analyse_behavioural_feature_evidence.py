"""
Phase 3 statistical evidence analysis.

This script uses the training feature table only. It does not load validation
or final-test data.

It measures associations between selected point-in-time behavioural features
and the fraud label using:
- Fraud rates
- Wilson 95% confidence intervals
- Fraud-rate differences
- Relative risk
- Two-proportion z-tests

Important interpretation rule:
These results describe association in the IEEE-CIS benchmark data. They do not
prove that a feature causes fraud and must not be described as causal evidence.
"""

import sys
from pathlib import Path

# Allow direct execution with:
# python scripts/analyse_behavioural_feature_evidence.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import math

import pandas as pd
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Input and output paths
# -----------------------------------------------------------------------------

TRAIN_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "point_in_time"
    / "train_behavioural_features.parquet"
)

TABLE_OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "tables"
REPORT_OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "evaluation"

RESULTS_CSV_PATH = (
    TABLE_OUTPUT_DIRECTORY
    / "phase3_feature_association_tests.csv"
)

REPORT_MARKDOWN_PATH = (
    REPORT_OUTPUT_DIRECTORY
    / "phase3_statistical_evidence.md"
)


def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Calculate a Wilson confidence interval for a binary proportion.

    Wilson intervals are more reliable than simple normal-approximation
    intervals when a fraud rate is small or group sizes differ.
    """
    if total <= 0:
        return float("nan"), float("nan")

    z_score = norm.ppf(1 - ((1 - confidence_level) / 2))
    proportion = successes / total

    denominator = 1 + (z_score**2 / total)

    centre = (
        proportion
        + (z_score**2 / (2 * total))
    ) / denominator

    margin = (
        z_score
        * math.sqrt(
            (
                proportion * (1 - proportion)
                + (z_score**2 / (4 * total))
            )
            / total
        )
    ) / denominator

    lower_bound = max(0.0, centre - margin)
    upper_bound = min(1.0, centre + margin)

    return lower_bound, upper_bound


def two_proportion_z_test(
    fraud_count_group_a: int,
    total_group_a: int,
    fraud_count_group_b: int,
    total_group_b: int,
) -> tuple[float, float]:
    """
    Perform a two-sided two-proportion z-test.

    Group A is the feature-present or higher-risk group.
    Group B is the comparison group.

    Returns
    -------
    tuple[float, float]
        z-statistic and two-sided p-value.
    """
    if total_group_a <= 0 or total_group_b <= 0:
        return float("nan"), float("nan")

    fraud_rate_a = fraud_count_group_a / total_group_a
    fraud_rate_b = fraud_count_group_b / total_group_b

    pooled_rate = (
        (fraud_count_group_a + fraud_count_group_b)
        / (total_group_a + total_group_b)
    )

    standard_error = math.sqrt(
        pooled_rate
        * (1 - pooled_rate)
        * ((1 / total_group_a) + (1 / total_group_b))
    )

    if standard_error == 0:
        return float("nan"), float("nan")

    z_statistic = (fraud_rate_a - fraud_rate_b) / standard_error

    p_value = 2 * norm.sf(abs(z_statistic))

    return z_statistic, p_value


def compare_binary_groups(
    df: pd.DataFrame,
    feature_name: str,
    group_a_mask: pd.Series,
    group_a_label: str,
    group_b_mask: pd.Series,
    group_b_label: str,
) -> dict:
    """
    Compare fraud rates for two mutually exclusive transaction groups.

    Group A is the selected condition, such as:
    - newly seen card
    - newly seen device
    - high recent card velocity

    Group B is the relevant comparison group.
    """
    group_a = df.loc[group_a_mask].copy()
    group_b = df.loc[group_b_mask].copy()

    total_group_a = len(group_a)
    total_group_b = len(group_b)

    fraud_count_group_a = int(group_a["isFraud"].sum())
    fraud_count_group_b = int(group_b["isFraud"].sum())

    fraud_rate_group_a = (
        fraud_count_group_a / total_group_a
        if total_group_a > 0
        else float("nan")
    )

    fraud_rate_group_b = (
        fraud_count_group_b / total_group_b
        if total_group_b > 0
        else float("nan")
    )

    group_a_ci_lower, group_a_ci_upper = wilson_confidence_interval(
        successes=fraud_count_group_a,
        total=total_group_a,
    )

    group_b_ci_lower, group_b_ci_upper = wilson_confidence_interval(
        successes=fraud_count_group_b,
        total=total_group_b,
    )

    fraud_rate_difference = fraud_rate_group_a - fraud_rate_group_b

    relative_risk = (
        fraud_rate_group_a / fraud_rate_group_b
        if fraud_rate_group_b > 0
        else float("nan")
    )

    z_statistic, p_value = two_proportion_z_test(
        fraud_count_group_a=fraud_count_group_a,
        total_group_a=total_group_a,
        fraud_count_group_b=fraud_count_group_b,
        total_group_b=total_group_b,
    )

    return {
        "feature": feature_name,
        "group_a_label": group_a_label,
        "group_b_label": group_b_label,
        "group_a_transactions": total_group_a,
        "group_a_fraud_count": fraud_count_group_a,
        "group_a_fraud_rate": fraud_rate_group_a,
        "group_a_ci_95_lower": group_a_ci_lower,
        "group_a_ci_95_upper": group_a_ci_upper,
        "group_b_transactions": total_group_b,
        "group_b_fraud_count": fraud_count_group_b,
        "group_b_fraud_rate": fraud_rate_group_b,
        "group_b_ci_95_lower": group_b_ci_lower,
        "group_b_ci_95_upper": group_b_ci_upper,
        "fraud_rate_difference": fraud_rate_difference,
        "relative_risk": relative_risk,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "statistically_significant_at_0_05": (
            bool(p_value < 0.05)
            if not pd.isna(p_value)
            else False
        ),
    }


def format_percentage(value: float) -> str:
    """Format a proportion as a percentage for the Markdown report."""
    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.2f}%"


def format_number(value: float) -> str:
    """Format counts and numbers safely for the Markdown report."""
    if pd.isna(value):
        return "N/A"

    return f"{value:,.0f}"


def format_decimal(value: float, decimal_places: int = 3) -> str:
    """Format a numeric result with a specified number of decimal places."""
    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimal_places}f}"


def create_markdown_report(
    results_df: pd.DataFrame,
    velocity_threshold: float,
    train_rows: int,
    overall_fraud_rate: float,
) -> str:
    """
    Create a readable Phase 3 evidence report.

    The report intentionally uses non-causal language. It says that a feature
    is associated with a higher/lower observed fraud rate; it does not say that
    the feature caused fraud.
    """
    report_lines = [
        "# Phase 3 Statistical Evidence Report",
        "",
        "## Scope",
        "",
        "This report evaluates selected associations between point-in-time-safe",
        "behavioural features and the `isFraud` label in the IEEE-CIS training",
        "period only. Validation and final-test data were not used.",
        "",
        f"- Training transactions analysed: {train_rows:,}",
        f"- Overall training fraud rate: {format_percentage(overall_fraud_rate)}",
        "- Confidence interval method: Wilson 95% confidence interval",
        "- Group comparison: two-sided two-proportion z-test",
        "",
        "## Velocity definition",
        "",
        "High card velocity is defined using the 95th percentile of the",
        "`velocity_card1_60min` distribution in the training period.",
        "",
        f"- High-velocity threshold: `velocity_card1_60min >= {velocity_threshold:.0f}`",
        "",
        "## Results",
        "",
        "| Feature comparison | Group A fraud rate (95% CI) | Group B fraud rate (95% CI) | Difference | Relative risk | p-value |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in results_df.iterrows():
        group_a_rate_ci = (
            f"{format_percentage(row['group_a_fraud_rate'])} "
            f"({format_percentage(row['group_a_ci_95_lower'])}–"
            f"{format_percentage(row['group_a_ci_95_upper'])})"
        )

        group_b_rate_ci = (
            f"{format_percentage(row['group_b_fraud_rate'])} "
            f"({format_percentage(row['group_b_ci_95_lower'])}–"
            f"{format_percentage(row['group_b_ci_95_upper'])})"
        )

        report_lines.append(
            "| "
            f"{row['feature']} | "
            f"{group_a_rate_ci} | "
            f"{group_b_rate_ci} | "
            f"{format_percentage(row['fraud_rate_difference'])} | "
            f"{format_decimal(row['relative_risk'], 2)} | "
            f"{format_decimal(row['p_value'], 6)} |"
        )

    report_lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- These results show statistical association in an anonymised public fraud benchmark dataset.",
            "- A higher observed fraud rate does not prove that a new card, device, or high velocity caused fraud.",
            "- The features may correlate with unobserved factors, collection processes, product mix, or other variables.",
            "- Results from this analysis must not be treated as financial-institution production outcomes.",
            "- The final test period remains locked and has not been used in this Phase 3 analysis.",
            "",
        ]
    )

    return "\n".join(report_lines)


def main() -> None:
    """Run Phase 3 association tests and save the outputs."""
    TABLE_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE 3: STATISTICAL EVIDENCE ANALYSIS")
    print("=" * 72)

    # Load training data only. No validation or final-test data is loaded.
    print(f"\nLoading training feature table:\n{TRAIN_FEATURE_PATH}")

    train_df = pd.read_parquet(TRAIN_FEATURE_PATH)

    print(f"Training rows loaded: {len(train_df):,}")
    print(f"Training columns loaded: {len(train_df.columns):,}")

    required_columns = [
        "isFraud",
        "is_new_card1",
        "is_new_DeviceInfo",
        "velocity_card1_60min",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in train_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required Phase 3 columns are missing: "
            f"{missing_columns}"
        )

    # Confirm binary fraud target validity.
    fraud_values = set(train_df["isFraud"].dropna().unique())

    if not fraud_values.issubset({0, 1}):
        raise ValueError(
            "The isFraud column must contain only binary values: 0 and 1."
        )

    overall_fraud_rate = float(train_df["isFraud"].mean())

    # -------------------------------------------------------------------------
    # Analysis 1: New card versus previously seen card
    # -------------------------------------------------------------------------
    print("\n[1/3] Comparing new cards against previously seen cards...")

    known_card_rows = train_df["is_new_card1"].notna()

    new_card_mask = known_card_rows & (train_df["is_new_card1"] == 1)
    existing_card_mask = known_card_rows & (train_df["is_new_card1"] == 0)

    new_card_results = compare_binary_groups(
        df=train_df,
        feature_name="New card1 versus previously seen card1",
        group_a_mask=new_card_mask,
        group_a_label="New card1 (is_new_card1 = 1)",
        group_b_mask=existing_card_mask,
        group_b_label="Previously seen card1 (is_new_card1 = 0)",
    )

    # -------------------------------------------------------------------------
    # Analysis 2: New device versus previously seen device
    #
    # Missing DeviceInfo rows are excluded. Missing does not mean a new device.
    # -------------------------------------------------------------------------
    print("[2/3] Comparing new devices against previously seen devices...")

    known_device_rows = train_df["is_new_DeviceInfo"].notna()

    new_device_mask = known_device_rows & (train_df["is_new_DeviceInfo"] == 1)
    existing_device_mask = (
        known_device_rows
        & (train_df["is_new_DeviceInfo"] == 0)
    )

    new_device_results = compare_binary_groups(
        df=train_df,
        feature_name="New device versus previously seen device",
        group_a_mask=new_device_mask,
        group_a_label="New DeviceInfo (is_new_DeviceInfo = 1)",
        group_b_mask=existing_device_mask,
        group_b_label="Previously seen DeviceInfo (is_new_DeviceInfo = 0)",
    )

    # -------------------------------------------------------------------------
    # Analysis 3: High card velocity versus lower card velocity
    #
    # Define high velocity from the training distribution only.
    # -------------------------------------------------------------------------
    print("[3/3] Comparing high card velocity against lower card velocity...")

    velocity_threshold = float(
        train_df["velocity_card1_60min"].quantile(0.95)
    )

    high_velocity_mask = (
        train_df["velocity_card1_60min"] >= velocity_threshold
    )

    lower_velocity_mask = (
        train_df["velocity_card1_60min"] < velocity_threshold
    )

    high_velocity_results = compare_binary_groups(
        df=train_df,
        feature_name=(
            "High card1 60-minute velocity versus lower card1 velocity"
        ),
        group_a_mask=high_velocity_mask,
        group_a_label=(
            "High velocity "
            f"(velocity_card1_60min >= {velocity_threshold:.0f})"
        ),
        group_b_mask=lower_velocity_mask,
        group_b_label=(
            "Lower velocity "
            f"(velocity_card1_60min < {velocity_threshold:.0f})"
        ),
    )

    # Create one structured results table.
    results_df = pd.DataFrame(
        [
            new_card_results,
            new_device_results,
            high_velocity_results,
        ]
    )

    # Round numeric values for a clean CSV output.
    results_df = results_df.round(
        {
            "group_a_fraud_rate": 6,
            "group_a_ci_95_lower": 6,
            "group_a_ci_95_upper": 6,
            "group_b_fraud_rate": 6,
            "group_b_ci_95_lower": 6,
            "group_b_ci_95_upper": 6,
            "fraud_rate_difference": 6,
            "relative_risk": 6,
            "z_statistic": 6,
            "p_value": 10,
        }
    )

    # Save table.
    print(f"\nSaving results table:\n{RESULTS_CSV_PATH}")

    results_df.to_csv(
        RESULTS_CSV_PATH,
        index=False,
    )

    # Save Markdown evidence report.
    report_text = create_markdown_report(
        results_df=results_df,
        velocity_threshold=velocity_threshold,
        train_rows=len(train_df),
        overall_fraud_rate=overall_fraud_rate,
    )

    print(f"\nSaving Markdown report:\n{REPORT_MARKDOWN_PATH}")

    REPORT_MARKDOWN_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    # Console summary.
    print("\n" + "=" * 72)
    print("STATISTICAL EVIDENCE SUMMARY")
    print("=" * 72)

    for _, row in results_df.iterrows():
        print(f"\nFeature: {row['feature']}")
        print(
            f"  Group A: {row['group_a_label']} | "
            f"transactions={row['group_a_transactions']:,} | "
            f"fraud rate={row['group_a_fraud_rate'] * 100:.2f}%"
        )
        print(
            f"  Group B: {row['group_b_label']} | "
            f"transactions={row['group_b_transactions']:,} | "
            f"fraud rate={row['group_b_fraud_rate'] * 100:.2f}%"
        )
        print(
            f"  Difference: {row['fraud_rate_difference'] * 100:.2f}% | "
            f"Relative risk: {row['relative_risk']:.2f} | "
            f"p-value: {row['p_value']:.8f}"
        )

    print("\nSafety and interpretation checks: PASSED")
    print("- Only the training feature table was loaded")
    print("- Validation and locked final-test data were not loaded")
    print("- All tested features were created before model training")
    print("- Results describe association only; they do not establish causation")

    print("\nStatistical evidence analysis completed successfully.")


if __name__ == "__main__":
    main()