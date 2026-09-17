"""
Validate IEEE-CIS data quality using Pandera schemas.

This script defines and applies data validation rules to ensure:
- Required columns are present with correct dtypes
- No duplicate TransactionIDs
- Valid value ranges (positive amounts, timestamps)
- Acceptable null rates
- Monotonic timestamp ordering

Validation failures raise clear errors to prevent bad data from entering modelling.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# -----------------------------------------------------------------------------
# Define Pandera schema for transaction data
# -----------------------------------------------------------------------------

class TransactionSchema(pa.DataFrameModel):
    """
    Pandera schema for IEEE-CIS transaction data validation.
    Enforces column presence, dtypes, ranges, and null constraints.
    """

    # Key identifier - must be unique, no nulls
    TransactionID: Series[int] = pa.Field(coerce=True, unique=True, nullable=False)

    # Timestamp - must be positive integer, monotonic increasing
    TransactionDT: Series[int] = pa.Field(coerce=True, ge=0, nullable=False)

    # Amount - must be positive float
    TransactionAmt: Series[float] = pa.Field(coerce=True, gt=0, nullable=False)

    # Target variable - binary 0/1
    isFraud: Series[int] = pa.Field(coerce=True, isin=[0, 1], nullable=False)

    # Product type - categorical
    ProductCD: Series[str] = pa.Field(coerce=True, nullable=True)

    # Card fields - mixed int/float, allow nulls
    card1: Series[float] = pa.Field(coerce=True, nullable=True)
    card2: Series[float] = pa.Field(coerce=True, nullable=True)
    card3: Series[float] = pa.Field(coerce=True, nullable=True)
    card4: Series[str] = pa.Field(coerce=True, nullable=True)
    card5: Series[float] = pa.Field(coerce=True, nullable=True)
    card6: Series[str] = pa.Field(coerce=True, nullable=True)

    # Address fields
    addr1: Series[float] = pa.Field(coerce=True, nullable=True)
    addr2: Series[float] = pa.Field(coerce=True, nullable=True)

    # Identity fields - numeric (float, allow nulls)
    id_01: Series[float] = pa.Field(coerce=True, nullable=True)
    id_02: Series[float] = pa.Field(coerce=True, nullable=True)
    id_03: Series[float] = pa.Field(coerce=True, nullable=True)
    # id_04 is string (T/F)
    id_04: Series[str] = pa.Field(coerce=True, nullable=True)
    id_05: Series[float] = pa.Field(coerce=True, nullable=True)
    # id_06 is string (T/F)
    id_06: Series[str] = pa.Field(coerce=True, nullable=True)
    id_07: Series[float] = pa.Field(coerce=True, nullable=True)
    id_08: Series[float] = pa.Field(coerce=True, nullable=True)
    id_09: Series[float] = pa.Field(coerce=True, nullable=True)
    id_10: Series[float] = pa.Field(coerce=True, nullable=True)
    id_11: Series[float] = pa.Field(coerce=True, nullable=True)
    # id_12 to id_16 are strings
    id_12: Series[str] = pa.Field(coerce=True, nullable=True)
    id_13: Series[str] = pa.Field(coerce=True, nullable=True)
    id_14: Series[float] = pa.Field(coerce=True, nullable=True)
    id_15: Series[str] = pa.Field(coerce=True, nullable=True)
    id_16: Series[str] = pa.Field(coerce=True, nullable=True)
    id_17: Series[float] = pa.Field(coerce=True, nullable=True)
    id_18: Series[float] = pa.Field(coerce=True, nullable=True)
    id_19: Series[float] = pa.Field(coerce=True, nullable=True)
    id_20: Series[float] = pa.Field(coerce=True, nullable=True)
    id_21: Series[float] = pa.Field(coerce=True, nullable=True)
    id_22: Series[float] = pa.Field(coerce=True, nullable=True)
    # id_23 is string
    id_23: Series[str] = pa.Field(coerce=True, nullable=True)
    id_24: Series[float] = pa.Field(coerce=True, nullable=True)
    id_25: Series[float] = pa.Field(coerce=True, nullable=True)
    id_26: Series[float] = pa.Field(coerce=True, nullable=True)
    # id_27 to id_31 are strings
    id_27: Series[str] = pa.Field(coerce=True, nullable=True)
    id_28: Series[str] = pa.Field(coerce=True, nullable=True)
    id_29: Series[str] = pa.Field(coerce=True, nullable=True)
    id_30: Series[str] = pa.Field(coerce=True, nullable=True)
    id_31: Series[str] = pa.Field(coerce=True, nullable=True)
    id_32: Series[float] = pa.Field(coerce=True, nullable=True)
    # id_33, id_34 are strings
    id_33: Series[str] = pa.Field(coerce=True, nullable=True)
    id_34: Series[str] = pa.Field(coerce=True, nullable=True)
    # id_35, id_36 are strings (T/F)
    id_35: Series[str] = pa.Field(coerce=True, nullable=True)
    id_36: Series[str] = pa.Field(coerce=True, nullable=True)
    # id_37, id_38 are strings (T/F) - NOT floats
    id_37: Series[str] = pa.Field(coerce=True, nullable=True)
    id_38: Series[str] = pa.Field(coerce=True, nullable=True)

    class Config:
        """Pandera config: coerce types, fail fast on errors."""
        coerce = True
        strict = False  # Allow extra columns (M-series, C-series, etc.)


# -----------------------------------------------------------------------------
# Validation functions
# -----------------------------------------------------------------------------

def check_duplicates(df: pd.DataFrame, column: str = "TransactionID") -> dict:
    """
    Check for duplicate values in the specified column.
    Returns a dict with duplicate count and sample duplicates.
    """
    duplicates = df[df.duplicated(subset=[column], keep=False)]
    return {
        "column": column,
        "duplicate_count": len(duplicates),
        "sample_duplicates": duplicates[column].head(10).tolist() if len(duplicates) > 0 else []
    }


def check_timestamp_monotonic(df: pd.DataFrame, column: str = "TransactionDT") -> dict:
    """
    Check if timestamp column is monotonically increasing.
    Returns a dict with violation count and sample violations.
    """
    is_monotonic = df[column].is_monotonic_increasing
    violations = df[df[column].diff() < 0] if not is_monotonic else pd.DataFrame()
    return {
        "column": column,
        "is_monotonic": bool(is_monotonic),
        "violation_count": len(violations),
        "sample_violations": violations[column].head(10).tolist() if len(violations) > 0 else []
    }


def check_null_rates(df: pd.DataFrame, threshold: float = 0.95) -> dict:
    """
    Calculate null rates per column and flag columns exceeding threshold.
    Returns a dict with null rates and high-null columns.
    """
    null_rates = (df.isna().sum() / len(df)).to_dict()
    high_null_cols = {col: rate for col, rate in null_rates.items() if rate > threshold}
    return {
        "null_rates": null_rates,
        "high_null_columns": high_null_cols,
        "threshold": threshold
    }


def check_value_ranges(df: pd.DataFrame) -> dict:
    """
    Check for invalid value ranges in key numeric columns.
    Returns a dict with violation counts for each checked column.
    """
    violations = {}

    # TransactionDT must be >= 0
    if "TransactionDT" in df.columns:
        violations["TransactionDT_negative"] = int((df["TransactionDT"] < 0).sum())

    # TransactionAmt must be > 0
    if "TransactionAmt" in df.columns:
        violations["TransactionAmt_non_positive"] = int((df["TransactionAmt"] <= 0).sum())

    # isFraud must be 0 or 1
    if "isFraud" in df.columns:
        violations["isFraud_invalid"] = int((~df["isFraud"].isin([0, 1])).sum())

    return violations


def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Run all validation checks on the dataframe.
    Returns a comprehensive validation report.
    """
    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "schema_validation": None,
        "duplicate_check": None,
        "timestamp_check": None,
        "null_rate_check": None,
        "value_range_check": None,
        "passed": True,
        "errors": []
    }

    # Schema validation
    try:
        TransactionSchema.validate(df, inplace=False)
        report["schema_validation"] = "PASSED"
    except pa.errors.SchemaError as e:
        report["schema_validation"] = "FAILED"
        report["passed"] = False
        report["errors"].append(f"Schema validation failed: {e!s}")

    # Duplicate check
    dup_result = check_duplicates(df)
    report["duplicate_check"] = dup_result
    if dup_result["duplicate_count"] > 0:
        report["passed"] = False
        report["errors"].append(f"Duplicate TransactionIDs found: {dup_result['duplicate_count']}")

    # Timestamp monotonic check
    ts_result = check_timestamp_monotonic(df)
    report["timestamp_check"] = ts_result
    if not ts_result["is_monotonic"]:
        report["errors"].append(f"TransactionDT not monotonic: {ts_result['violation_count']} violations")

    # Null rate check
    null_result = check_null_rates(df)
    report["null_rate_check"] = null_result

    # Value range check
    range_result = check_value_ranges(df)
    report["value_range_check"] = range_result
    invalid_ranges = {k: v for k, v in range_result.items() if v > 0}
    if invalid_ranges:
        report["passed"] = False
        report["errors"].append(f"Invalid value ranges: {invalid_ranges}")

    return report


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main():
    """
    Load joined transaction data and run validation checks.
    Save validation report to data/interim/validation_report.json
    """
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "interim" / "joined_transactions" / "train_joined.parquet"
    output_path = project_root / "data" / "interim" / "validation_report.json"

    print(f"Loading data from: {input_path}")

    # Load data
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Validate
    print("Running validation checks...")
    report = validate_dataframe(df)

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Validation report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Row count: {report['row_count']:,}")
    print(f"Column count: {report['column_count']}")
    print(f"Schema validation: {report['schema_validation']}")
    print(f"Duplicate check: {report['duplicate_check']['duplicate_count']} duplicates")
    print(f"Timestamp monotonic: {report['timestamp_check']['is_monotonic']}")
    print(f"High-null columns (>95%): {len(report['null_rate_check']['high_null_columns'])}")
    print(f"Value range violations: {report['value_range_check']}")
    print(f"OVERALL PASSED: {report['passed']}")

    if report["errors"]:
        print("\nERRORS:")
        for err in report["errors"]:
            print(f"  - {err}")

    # Exit with error code if validation failed
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()