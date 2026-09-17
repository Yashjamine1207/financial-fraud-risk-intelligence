"""
Generate point-in-time-safe behavioural and target-encoded features for the final test split.


Important leakage-control design:
- Train and validation data are historical context for test data.
- Test rows can use only earlier train, validation, and test transactions.
- isFraud is never used during feature creation.
- Target encoding uses mappings fitted on the complete training period only.
- Test labels are never used for encoding.


Outputs:
- data/features/point_in_time/test_behavioural_features.parquet
- data/features/point_in_time/test_feature_metadata.json
"""


import json
import sys
from pathlib import Path

# Allow direct execution with: python scripts/generate_final_test_features.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
import yaml

from src.fraud_intelligence.features import (
    PointInTimeFeaturePipeline,
    TimeSafeTargetEncoder,
)

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


TEST_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "temporal_splits"
    / "test.parquet"
)


FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"


BEHAVIOURAL_FEATURE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "point_in_time"
    / "test_behavioural_features.parquet"
)


TARGET_ENCODING_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "point_in_time"
    / "target_encoding_metadata.json"
)


FINAL_TEST_FEATURE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "point_in_time"
    / "test_behavioural_and_encoded_features.parquet"
)


FINAL_TEST_METADATA_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "point_in_time"
    / "test_feature_metadata.json"
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


def load_target_encoding_config() -> dict:
    """Load the target-encoding settings from configs/features.yaml."""
    with FEATURE_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if "target_encoding" not in config:
        raise ValueError(
            "configs/features.yaml is missing the target_encoding section."
        )

    target_encoding_config = config["target_encoding"]

    required_keys = {
        "smoothing",
        "min_samples",
        "oof_folds",
        "columns",
    }

    missing_keys = required_keys.difference(target_encoding_config)

    if missing_keys:
        raise ValueError(
            "Target encoding configuration is missing keys: "
            f"{sorted(missing_keys)}"
        )

    return target_encoding_config


def main() -> None:
    """Generate and save the final test behavioural and target-encoded features."""
    print("=" * 72)
    print("PHASE 3 EXTENSION: FINAL TEST FEATURE GENERATION")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Load train, validation, and final test.
    #
    # This script is explicitly allowed to load the test split because its
    # only purpose is to create a feature table for the one-time holdout
    # evaluation. No labels are used during feature creation or encoding.
    # -------------------------------------------------------------------------
    print(f"\nLoading training history:\n{TRAIN_INPUT_PATH}")
    train_df = pd.read_parquet(TRAIN_INPUT_PATH)
    print(f"Training rows loaded: {len(train_df):,}")

    print(f"\nLoading validation history:\n{VALIDATION_INPUT_PATH}")
    validation_df = pd.read_parquet(VALIDATION_INPUT_PATH)
    print(f"Validation rows loaded: {len(validation_df):,}")

    print(f"\nLoading final test split:\n{TEST_INPUT_PATH}")
    test_df = pd.read_parquet(TEST_INPUT_PATH)
    print(f"Test rows loaded: {len(test_df):,}")

    # Add a temporary column so we can save only test rows after feature creation.
    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["_feature_split"] = "train_history"
    validation_df["_feature_split"] = "validation_history"
    test_df["_feature_split"] = "test"

    combined_df = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True,
    )

    print(f"\nCombined rows used for feature history: {len(combined_df):,}")

    expected_combined_rows = len(train_df) + len(validation_df) + len(test_df)

    if len(combined_df) != expected_combined_rows:
        raise ValueError(
            "Unexpected combined row count. Test feature generation stopped."
        )

    # -------------------------------------------------------------------------
    # Behavioural feature pipeline
    # -------------------------------------------------------------------------
    pipeline = PointInTimeFeaturePipeline(
        config_path=FEATURE_CONFIG_PATH,
        version="v1.0.0",
    )

    print("\nCreating point-in-time-safe behavioural features...")
    combined_feature_df = build_behavioural_features(
        df=combined_df,
        pipeline=pipeline,
    )

    test_feature_df = (
        combined_feature_df
        .loc[combined_feature_df["_feature_split"] == "test"]
        .copy()
        .drop(columns="_feature_split")
        .reset_index(drop=True)
    )

    if len(test_feature_df) != len(test_df):
        raise ValueError(
            "Safety check failed: test output row count does not match "
            "the Phase 2 test split."
        )

    if test_feature_df["TransactionID"].duplicated().any():
        raise ValueError(
            "Safety check failed: duplicate TransactionID values in test output."
        )

    if not test_feature_df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "Safety check failed: test output is not chronological."
        )

    behavioural_feature_columns = sorted(
        column
        for column in test_feature_df.columns
        if column.startswith(
            (
                "velocity_",
                "amount_",
                "recency_",
                "history_",
                "is_new_",
            )
        )
    )

    expected_behavioural_feature_count = 34

    if len(behavioural_feature_columns) != expected_behavioural_feature_count:
        raise ValueError(
            "Safety check failed: expected "
            f"{expected_behavioural_feature_count} behavioural features but found "
            f"{len(behavioural_feature_columns)}."
        )

    print(f"Behavioural features created: {len(behavioural_feature_columns)}")

    # -------------------------------------------------------------------------
    # Target encoding using train-fitted mappings only.
    # -------------------------------------------------------------------------
    print("\nLoading target-encoding configuration...")
    target_encoding_config = load_target_encoding_config()

    encoding_columns = target_encoding_config["columns"]
    smoothing = float(target_encoding_config["smoothing"])
    min_samples = int(target_encoding_config["min_samples"])

    print(f"Target-encoded columns: {encoding_columns}")

    # Fit a fresh encoder on the complete training period to reproduce the same
    # mapping that was used for validation. This encoder will then be applied
    # to the test period.
    print("\nFitting target encoder on the complete training period...")

    encoder = TimeSafeTargetEncoder(
        columns=encoding_columns,
        smoothing=smoothing,
        min_samples=min_samples,
    )

    # Fit on the full training feature table using its isFraud labels.
    # This reproduces the same mapping used in the original Phase 3 run.
    encoder.fit(
        df=train_df,
        target=train_df["isFraud"],
    )

    print(f"Training global fraud rate: {encoder.global_fraud_rate_:.6f}")

    print("\nApplying train-fitted target encoding to test features...")
    test_encoded_features = encoder.transform(test_feature_df)

    test_feature_df = pd.concat(
        [
            test_feature_df.reset_index(drop=True),
            test_encoded_features.reset_index(drop=True),
        ],
        axis=1,
    )

    encoded_columns = [
        f"{column}_target_encoded"
        for column in encoding_columns
    ]

    missing_encoded_columns = [
        column
        for column in encoded_columns
        if column not in test_feature_df.columns
    ]

    if missing_encoded_columns:
        raise ValueError(
            "Safety check failed: test encoded columns missing: "
            f"{missing_encoded_columns}"
        )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL TEST FEATURE GENERATION SUMMARY")
    print("=" * 72)

    print(f"Test rows saved: {len(test_feature_df):,}")
    print(f"Behavioural features: {len(behavioural_feature_columns)}")
    print(f"Target-encoded features: {len(encoded_columns)}")
    print(f"Total output columns: {len(test_feature_df.columns):,}")
    print("Test feature source: train + validation + earlier test history")
    print("Test labels used in encoding: No")

    print(f"\nSaving test behavioural and encoded features:\n{FINAL_TEST_FEATURE_OUTPUT_PATH}")

    test_feature_df.to_parquet(
        FINAL_TEST_FEATURE_OUTPUT_PATH,
        index=False,
    )

    metadata = {
        "feature_pipeline_version": "v1.0.0",
        "encoding_type": "train-fitted target encoding applied to test",
        "source_config": str(FEATURE_CONFIG_PATH),
        "target_column": "isFraud",
        "encoded_columns": encoded_columns,
        "source_columns": encoding_columns,
        "smoothing": smoothing,
        "min_samples": min_samples,
        "test_rows": len(test_feature_df),
        "training_global_fraud_rate": encoder.global_fraud_rate_,
        "test_labels_used_in_encoding": False,
        "encoding_rule": (
            "Test is transformed using mappings fitted on all training labels only."
        ),
    }

    print(f"\nSaving test feature metadata:\n{FINAL_TEST_METADATA_OUTPUT_PATH}")

    with FINAL_TEST_METADATA_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    preview_columns = [
        "TransactionID",
        "TransactionDT",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "velocity_global_5min",
        "amount_global_1h_mean",
        "recency_global_seconds",
        "history_card1_transaction_count",
        "is_new_card1",
        *encoded_columns,
    ]

    print("\nFirst five test rows with features:")
    print(test_feature_df[preview_columns].head().to_string(index=False))

    print("\nFinal test feature generation completed successfully.")


if __name__ == "__main__":
    main()