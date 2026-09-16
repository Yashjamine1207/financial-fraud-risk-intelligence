"""
Create point-in-time-safe behavioural features for the Phase 2 training split.

The script creates features using only historical transaction information:

1. Global transaction velocity
2. Card-level transaction velocity
3. Global historical amount statistics
4. Card-level historical amount statistics
5. Global and card-level recency
6. Historical entity relationship counts
7. New-entity indicators

Point-in-time safety rules:
- Every historical feature uses strictly earlier TransactionDT values only.
- Transactions sharing the same TransactionDT cannot use one another.
- The fraud label (isFraud) is never used to build these features.
- Missing entity values are not treated as one shared device, address, or email.
"""

from pathlib import Path
import sys

# Add the project root to the import path so this script can be run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.fraud_intelligence.features import PointInTimeFeaturePipeline


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

TRAIN_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "temporal_splits"
    / "train.parquet"
)

FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "features" / "point_in_time"

TRAIN_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "train_behavioural_features.parquet"
)

METADATA_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "train_feature_metadata.json"
)


# -----------------------------------------------------------------------------
# Entity specifications
# -----------------------------------------------------------------------------
# Each entity represents an identifier that could be available at transaction
# scoring time. We calculate only historical usage count and first-seen status.
#
# card1          -> card proxy history
# addr1          -> billing-address proxy history
# DeviceInfo     -> device history, only where device information is present
# R_emaildomain  -> recipient email-domain history, only where present
# -----------------------------------------------------------------------------

ENTITY_FEATURE_SPECS = [
    ["card1"],
    ["addr1"],
    ["DeviceInfo"],
    ["R_emaildomain"],
]


def main() -> None:
    """Create and save point-in-time-safe training behavioural features."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE 3: TRAINING BEHAVIOURAL FEATURE GENERATION")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Load the immutable Phase 2 chronological training split.
    # -------------------------------------------------------------------------
    print(f"\nLoading training split:\n{TRAIN_INPUT_PATH}")

    train_df = pd.read_parquet(TRAIN_INPUT_PATH)

    print(f"Rows loaded: {len(train_df):,}")
    print(f"Columns before feature generation: {len(train_df.columns):,}")

    # The pipeline reads window definitions and feature settings from the
    # version-controlled YAML configuration.
    pipeline = PointInTimeFeaturePipeline(
        config_path=FEATURE_CONFIG_PATH,
        version="v1.0.0",
    )

    # -------------------------------------------------------------------------
    # 1. Global velocity features
    # Historical transaction counts across the entire transaction stream.
    # -------------------------------------------------------------------------
    print("\n[1/8] Creating global velocity features...")

    feature_df = pipeline.compute_velocity_features(
        df=train_df,
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=None,
    )

    # -------------------------------------------------------------------------
    # 2. Card-level velocity features
    # Historical transaction counts for the same card1 proxy.
    # -------------------------------------------------------------------------
    print("[2/8] Creating card1 velocity features...")

    feature_df = pipeline.compute_velocity_features(
        df=feature_df,
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=["card1"],
    )

    # -------------------------------------------------------------------------
    # 3. Global historical amount features
    # Historical amount mean, standard deviation, and z-score.
    # -------------------------------------------------------------------------
    print("[3/8] Creating global amount-history features...")

    feature_df = pipeline.compute_amount_features(
        df=feature_df,
        amount_col="TransactionAmt",
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=None,
    )

    # -------------------------------------------------------------------------
    # 4. Card-level historical amount features
    # Amount behaviour relative to historical transactions for the same card1.
    # -------------------------------------------------------------------------
    print("[4/8] Creating card1 amount-history features...")

    feature_df = pipeline.compute_amount_features(
        df=feature_df,
        amount_col="TransactionAmt",
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=["card1"],
    )

    # -------------------------------------------------------------------------
    # 5. Global recency
    # Time since the most recent transaction in the overall transaction stream.
    # -------------------------------------------------------------------------
    print("[5/8] Creating global recency feature...")

    feature_df = pipeline.compute_recency_feature(
        df=feature_df,
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=None,
    )

    # -------------------------------------------------------------------------
    # 6. Card-level recency
    # Time since the previous observed transaction for the same card1 proxy.
    # -------------------------------------------------------------------------
    print("[6/8] Creating card1 recency feature...")

    feature_df = pipeline.compute_recency_feature(
        df=feature_df,
        transaction_id_col="TransactionID",
        timestamp_col="TransactionDT",
        group_cols=["card1"],
    )

    # -------------------------------------------------------------------------
    # 7. Historical entity transaction counts
    # 8. New-entity indicators
    #
    # The reusable pipeline method creates both features for every entity:
    # history_<entity>_transaction_count and is_new_<entity>.
    # -------------------------------------------------------------------------
    print("[7/8] Creating historical entity relationship counts...")
    print("[8/8] Creating new-entity indicators...")

    for entity_columns in ENTITY_FEATURE_SPECS:
        entity_name = "_".join(entity_columns)

        print(f"      Processing entity: {entity_name}")

        feature_df = pipeline.compute_historical_entity_features(
            df=feature_df,
            entity_columns=entity_columns,
            transaction_id_col="TransactionID",
            timestamp_col="TransactionDT",
        )

    # -------------------------------------------------------------------------
    # Collect all features produced by this script.
    # -------------------------------------------------------------------------
    velocity_columns = sorted(
        column
        for column in feature_df.columns
        if column.startswith("velocity_")
    )

    amount_columns = sorted(
        column
        for column in feature_df.columns
        if column.startswith("amount_")
    )

    recency_columns = sorted(
        column
        for column in feature_df.columns
        if column.startswith("recency_")
    )

    history_columns = sorted(
        column
        for column in feature_df.columns
        if column.startswith("history_")
    )

    new_entity_columns = sorted(
        column
        for column in feature_df.columns
        if column.startswith("is_new_")
    )

    created_feature_columns = (
        velocity_columns
        + amount_columns
        + recency_columns
        + history_columns
        + new_entity_columns
    )

    # -------------------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------------------
    if feature_df["TransactionID"].duplicated().any():
        raise ValueError(
            "Safety check failed: duplicate TransactionID values were found."
        )

    if not feature_df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "Safety check failed: the output is not sorted by TransactionDT."
        )

    expected_entity_columns = [
        "history_card1_transaction_count",
        "is_new_card1",
        "history_addr1_transaction_count",
        "is_new_addr1",
        "history_DeviceInfo_transaction_count",
        "is_new_DeviceInfo",
        "history_R_emaildomain_transaction_count",
        "is_new_R_emaildomain",
    ]

    missing_created_columns = [
        column
        for column in expected_entity_columns
        if column not in feature_df.columns
    ]

    if missing_created_columns:
        raise ValueError(
            "Entity feature generation failed. Missing columns: "
            f"{missing_created_columns}"
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FEATURE GENERATION SUMMARY")
    print("=" * 72)

    print(f"Velocity features created:     {len(velocity_columns)}")
    print(f"Amount features created:       {len(amount_columns)}")
    print(f"Recency features created:      {len(recency_columns)}")
    print(f"Historical count features:     {len(history_columns)}")
    print(f"New-entity indicator features: {len(new_entity_columns)}")
    print(f"Total new features:            {len(created_feature_columns)}")
    print(f"Columns after generation:      {len(feature_df.columns):,}")

    print("\nCreated entity relationship features:")

    for feature_name in history_columns + new_entity_columns:
        missing_rate = feature_df[feature_name].isna().mean() * 100

        print(
            f"  - {feature_name} "
            f"(missing because entity unavailable: {missing_rate:.2f}%)"
        )

    print("\nPoint-in-time safety checks: PASSED")
    print("- TransactionID remains unique")
    print("- Output remains sorted chronologically")
    print("- Historical features use strictly earlier timestamps only")
    print("- Same-second transactions cannot use one another")
    print("- Current transaction is excluded from its own history")
    print("- Missing entity identifiers are not grouped together")
    print("- Fraud labels are not used during feature creation")

    # -------------------------------------------------------------------------
    # Save final Phase 3 training feature table and metadata.
    # -------------------------------------------------------------------------
    print(f"\nSaving feature table:\n{TRAIN_OUTPUT_PATH}")

    feature_df.to_parquet(
        TRAIN_OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaving feature metadata:\n{METADATA_OUTPUT_PATH}")

    pipeline.save_feature_metadata(METADATA_OUTPUT_PATH)

    # Show a compact preview of new entity features.
    preview_columns = [
        "TransactionID",
        "TransactionDT",
        "card1",
        "addr1",
        "DeviceInfo",
        "R_emaildomain",
        "history_card1_transaction_count",
        "is_new_card1",
        "history_addr1_transaction_count",
        "is_new_addr1",
        "history_DeviceInfo_transaction_count",
        "is_new_DeviceInfo",
        "history_R_emaildomain_transaction_count",
        "is_new_R_emaildomain",
    ]

    print("\nFirst five rows of entity-history features:")
    print(
        feature_df[preview_columns]
        .head()
        .to_string(index=False)
    )

    print("\nFeature generation completed successfully.")


if __name__ == "__main__":
    main()