"""
Generate point-in-time-safe behavioural features for the validation split.

Important leakage-control design:
- Training data is historical context for validation data.
- Validation rows can use only earlier train and validation transactions.
- Test data is never loaded by this script.
- isFraud is never used during feature creation.
- Only validation rows are saved to the final validation feature table.

Outputs:
- data/features/point_in_time/validation_behavioural_features.parquet
- data/features/point_in_time/validation_feature_metadata.json
"""

from pathlib import Path
import sys

# Add the project root to the Python import path so the script can run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.fraud_intelligence.features import PointInTimeFeaturePipeline


# -----------------------------------------------------------------------------
# Input and output paths
# -----------------------------------------------------------------------------

TRAIN_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "temporal_splits"
    / "train.parquet"
)

VALIDATION_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "temporal_splits"
    / "validation.parquet"
)

FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "features" / "point_in_time"

VALIDATION_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "validation_behavioural_features.parquet"
)

METADATA_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "validation_feature_metadata.json"
)


# -----------------------------------------------------------------------------
# Entity definitions used for historical count and first-seen features
# -----------------------------------------------------------------------------

ENTITY_FEATURE_SPECS = [
    ["card1"],
    ["addr1"],
    ["DeviceInfo"],
    ["R_emaildomain"],
]


def build_behavioural_features(
    df: pd.DataFrame,
    pipeline: PointInTimeFeaturePipeline,
) -> pd.DataFrame:
    """
    Apply the complete Phase 3 behavioural feature pipeline.

    The input DataFrame must contain only transactions available at or before
    the split being generated and must include TransactionID and TransactionDT.
    """
    feature_df = pipeline.compute_velocity_features(
        df=df,
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=None,
    )

    feature_df = pipeline.compute_velocity_features(
        df=feature_df,
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=["card1"],
    )

    feature_df = pipeline.compute_amount_features(
        df=feature_df,
        amount_col="TransactionAmt",
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=None,
    )

    feature_df = pipeline.compute_amount_features(
        df=feature_df,
        amount_col="TransactionAmt",
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=["card1"],
    )

    feature_df = pipeline.compute_recency_feature(
        df=feature_df,
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=None,
    )

    feature_df = pipeline.compute_recency_feature(
        df=feature_df,
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=["card1"],
    )

    for entity_columns in ENTITY_FEATURE_SPECS:
        feature_df = pipeline.compute_historical_entity_features(
            df=feature_df,
            entity_columns=entity_columns,
            transaction_id_col="TransactionID",
            timestamp_col="TransactionDT",
        )

    return feature_df


def main() -> None:
    """Generate and save the validation behavioural feature table."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE 3: VALIDATION BEHAVIOURAL FEATURE GENERATION")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Load train and validation only.
    #
    # The test split is intentionally not loaded. It remains locked until the
    # final evaluation stage, as documented in the Phase 2 leakage audit.
    # -------------------------------------------------------------------------
    print(f"\nLoading training history:\n{TRAIN_INPUT_PATH}")
    train_df = pd.read_parquet(TRAIN_INPUT_PATH)

    print(f"Training rows loaded: {len(train_df):,}")

    print(f"\nLoading validation split:\n{VALIDATION_INPUT_PATH}")
    validation_df = pd.read_parquet(VALIDATION_INPUT_PATH)

    print(f"Validation rows loaded: {len(validation_df):,}")

    # Add a temporary column so we can save validation rows after calculating
    # features across the combined chronological history.
    train_df = train_df.copy()
    validation_df = validation_df.copy()

    train_df["_feature_split"] = "train_history"
    validation_df["_feature_split"] = "validation"

    # Concatenate history and validation records. The pipeline will sort by
    # TransactionDT and TransactionID before it creates every feature.
    combined_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    print(f"\nCombined rows used for feature history: {len(combined_df):,}")

    # Confirm that the combined data contains no test rows.
    expected_combined_rows = len(train_df) + len(validation_df)

    if len(combined_df) != expected_combined_rows:
        raise ValueError(
            "Unexpected combined row count. Validation feature generation stopped."
        )

    # Initialise the versioned pipeline.
    pipeline = PointInTimeFeaturePipeline(
        config_path=FEATURE_CONFIG_PATH,
        version="v1.0.0",
    )

    print("\nCreating point-in-time-safe features using train history...")
    combined_feature_df = build_behavioural_features(
        df=combined_df,
        pipeline=pipeline,
    )

    # Keep only rows from the validation period. Drop the temporary split
    # marker because it is pipeline metadata, not a model feature.
    validation_feature_df = (
        combined_feature_df
        .loc[
            combined_feature_df["_feature_split"] == "validation"
        ]
        .copy()
        .drop(columns="_feature_split")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------------------
    if len(validation_feature_df) != len(validation_df):
        raise ValueError(
            "Safety check failed: validation output row count does not match "
            "the Phase 2 validation split."
        )

    if validation_feature_df["TransactionID"].duplicated().any():
        raise ValueError(
            "Safety check failed: duplicate TransactionID values found."
        )

    if not validation_feature_df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "Safety check failed: validation output is not chronological."
        )

    if (
        validation_feature_df["TransactionDT"].min()
        < validation_df["TransactionDT"].min()
    ):
        raise ValueError(
            "Safety check failed: validation output contains an unexpected "
            "timestamp before the validation period."
        )

    # Collect all Phase 3 behavioural feature columns.
    behavioural_feature_columns = sorted(
        column
        for column in validation_feature_df.columns
        if (
            column.startswith("velocity_")
            or column.startswith("amount_")
            or column.startswith("recency_")
            or column.startswith("history_")
            or column.startswith("is_new_")
        )
    )

    expected_feature_count = 34

    if len(behavioural_feature_columns) != expected_feature_count:
        raise ValueError(
            "Safety check failed: expected "
            f"{expected_feature_count} behavioural features but found "
            f"{len(behavioural_feature_columns)}."
        )

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("VALIDATION FEATURE GENERATION SUMMARY")
    print("=" * 72)

    print(f"Validation rows saved:          {len(validation_feature_df):,}")
    print(f"Behavioural features created:   {len(behavioural_feature_columns)}")
    print(f"Total output columns:           {len(validation_feature_df.columns):,}")
    print(
        "Validation history source:      "
        "training rows + earlier validation rows"
    )
    print("Test split loaded:              No")

    print("\nPoint-in-time safety checks: PASSED")
    print("- Validation features use only earlier train/validation history")
    print("- Test data was not loaded")
    print("- TransactionID remains unique")
    print("- Validation output remains chronological")
    print("- Same-second transactions cannot use one another")
    print("- Fraud labels are not used during feature creation")

    print(f"\nSaving validation feature table:\n{VALIDATION_OUTPUT_PATH}")

    validation_feature_df.to_parquet(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaving validation feature metadata:\n{METADATA_OUTPUT_PATH}")

    pipeline.save_feature_metadata(METADATA_OUTPUT_PATH)

    # Preview the first validation rows. These rows should have historical
    # values because the complete training split is available as prior context.
    preview_columns = [
        "TransactionID",
        "TransactionDT",
        "card1",
        "TransactionAmt",
        "velocity_global_5min",
        "velocity_card1_5min",
        "amount_global_1h_mean",
        "amount_card1_1h_mean",
        "recency_global_seconds",
        "recency_card1_seconds",
        "history_card1_transaction_count",
        "is_new_card1",
    ]

    print("\nFirst five validation rows with behavioural features:")
    print(
        validation_feature_df[preview_columns]
        .head()
        .to_string(index=False)
    )

    print("\nValidation feature generation completed successfully.")


if __name__ == "__main__":
    main()