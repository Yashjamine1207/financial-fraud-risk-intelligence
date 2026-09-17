"""
Apply leakage-safe target encoding to Phase 3 train and validation features.

Train:
- Uses chronological out-of-fold target encoding.
- Every row is encoded using fraud labels from strictly earlier time folds only.
- Same-timestamp transactions cannot use one another's fraud labels.

Validation:
- Uses mappings fitted on the complete training feature table only.
- Validation labels are never used for validation target encoding.

Test:
- Never loaded by this script.
- Remains locked for final evaluation.
"""

import json
import sys
from pathlib import Path

# Allow direct execution with: python scripts/apply_target_encoding.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.fraud_intelligence.features import TimeSafeTargetEncoder

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

FEATURE_CONFIG_PATH = PROJECT_ROOT / "configs" / "features.yaml"

TRAIN_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "point_in_time"
    / "train_behavioural_features.parquet"
)

VALIDATION_FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "point_in_time"
    / "validation_behavioural_features.parquet"
)

TARGET_ENCODING_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "point_in_time"
    / "target_encoding_metadata.json"
)


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


def validate_required_columns(
    df: pd.DataFrame,
    target_encoding_columns: list[str],
    dataset_name: str,
) -> None:
    """Check that every required encoding and timeline column is present."""
    required_columns = [
        "TransactionID",
        "TransactionDT",
        "isFraud",
        *target_encoding_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )


def main() -> None:
    """Create chronological train and training-fitted validation encodings."""
    print("=" * 72)
    print("PHASE 3: LEAKAGE-SAFE TARGET ENCODING")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------
    target_encoding_config = load_target_encoding_config()

    encoding_columns = target_encoding_config["columns"]
    smoothing = float(target_encoding_config["smoothing"])
    min_samples = int(target_encoding_config["min_samples"])
    oof_folds = int(target_encoding_config["oof_folds"])

    print("\nTarget-encoding configuration:")
    print(f"Columns:     {encoding_columns}")
    print(f"Smoothing:   {smoothing}")
    print(f"Min samples: {min_samples}")
    print(f"OOF folds:   {oof_folds}")

    # -------------------------------------------------------------------------
    # Load only Phase 3 train and validation feature tables.
    #
    # Test data is deliberately not loaded. It remains locked until final
    # model, calibration, and threshold-policy selection are complete.
    # -------------------------------------------------------------------------
    print(f"\nLoading train feature table:\n{TRAIN_FEATURE_PATH}")
    train_df = pd.read_parquet(TRAIN_FEATURE_PATH)

    print(f"Train rows loaded: {len(train_df):,}")
    print(f"Train columns before encoding: {len(train_df.columns):,}")

    print(f"\nLoading validation feature table:\n{VALIDATION_FEATURE_PATH}")
    validation_df = pd.read_parquet(VALIDATION_FEATURE_PATH)

    print(f"Validation rows loaded: {len(validation_df):,}")
    print(f"Validation columns before encoding: {len(validation_df.columns):,}")

    validate_required_columns(
        df=train_df,
        target_encoding_columns=encoding_columns,
        dataset_name="Training feature table",
    )

    validate_required_columns(
        df=validation_df,
        target_encoding_columns=encoding_columns,
        dataset_name="Validation feature table",
    )

    # -------------------------------------------------------------------------
    # Training target encoding: chronological out-of-fold.
    # -------------------------------------------------------------------------
    print("\nCreating chronological out-of-fold encodings for training data...")

    encoder = TimeSafeTargetEncoder(
        columns=encoding_columns,
        smoothing=smoothing,
        min_samples=min_samples,
    )

    train_encoded_features = encoder.fit_transform_oof(
        df=train_df,
        target=train_df["isFraud"],
        timestamp_col="TransactionDT",
        transaction_id_col="TransactionID",
        n_splits=oof_folds,
    )

    # The encoder returns only new columns. Join them onto the existing
    # behavioural-feature table using the original row index.
    train_df = pd.concat(
        [
            train_df.reset_index(drop=True),
            train_encoded_features.reset_index(drop=True),
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Validation target encoding: train-fitted mapping only.
    #
    # encoder was fitted on the full training data at the end of
    # fit_transform_oof(). Validation labels are not passed here.
    # -------------------------------------------------------------------------
    print(
        "Creating validation encodings using the completed training-period "
        "mapping only..."
    )

    validation_encoded_features = encoder.transform(
        validation_df
    )

    validation_df = pd.concat(
        [
            validation_df.reset_index(drop=True),
            validation_encoded_features.reset_index(drop=True),
        ],
        axis=1,
    )

    encoded_columns = [
        f"{column}_target_encoded"
        for column in encoding_columns
    ]

    # -------------------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------------------
    if len(train_df) != 413_378:
        raise ValueError(
            "Safety check failed: unexpected training row count."
        )

    if len(validation_df) != 88_581:
        raise ValueError(
            "Safety check failed: unexpected validation row count."
        )

    if train_df["TransactionID"].duplicated().any():
        raise ValueError(
            "Safety check failed: duplicate TransactionID values in train."
        )

    if validation_df["TransactionID"].duplicated().any():
        raise ValueError(
            "Safety check failed: duplicate TransactionID values in validation."
        )

    if not train_df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "Safety check failed: train feature table is not chronological."
        )

    if not validation_df["TransactionDT"].is_monotonic_increasing:
        raise ValueError(
            "Safety check failed: validation feature table is not chronological."
        )

    missing_encoded_columns_train = [
        column
        for column in encoded_columns
        if column not in train_df.columns
    ]

    missing_encoded_columns_validation = [
        column
        for column in encoded_columns
        if column not in validation_df.columns
    ]

    if missing_encoded_columns_train:
        raise ValueError(
            "Safety check failed: train encoded columns missing: "
            f"{missing_encoded_columns_train}"
        )

    if missing_encoded_columns_validation:
        raise ValueError(
            "Safety check failed: validation encoded columns missing: "
            f"{missing_encoded_columns_validation}"
        )

    # The earliest OOF fold has no previous labelled history. Missing values
    # are expected there and will later be handled by training-only imputation.
    train_missing_rates = {
        column: round(float(train_df[column].isna().mean() * 100), 2)
        for column in encoded_columns
    }

    validation_missing_rates = {
        column: round(float(validation_df[column].isna().mean() * 100), 2)
        for column in encoded_columns
    }

    # Validation encodings should have no missing values because unknown
    # categories use the global training fraud rate.
    validation_columns_with_missing_values = [
        column
        for column, missing_rate in validation_missing_rates.items()
        if missing_rate > 0
    ]

    if validation_columns_with_missing_values:
        raise ValueError(
            "Safety check failed: validation target encodings contain missing "
            "values: "
            f"{validation_columns_with_missing_values}"
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TARGET ENCODING SUMMARY")
    print("=" * 72)

    print(f"Target-encoded columns added: {len(encoded_columns)}")
    print(f"Train columns after encoding:  {len(train_df.columns):,}")
    print(f"Validation columns after encoding: {len(validation_df.columns):,}")
    print(f"Training global fraud rate: {encoder.global_fraud_rate_:.6f}")

    print("\nCreated columns:")

    for column in encoded_columns:
        print(
            f"  - {column} | "
            f"train missing: {train_missing_rates[column]:.2f}% | "
            f"validation missing: {validation_missing_rates[column]:.2f}%"
        )

    print("\nLeakage safety checks: PASSED")
    print("- Training uses chronological out-of-fold label history")
    print("- Same-timestamp transactions cannot use one another's labels")
    print("- Validation uses training-period mappings only")
    print("- Validation fraud labels were not used")
    print("- Test split was not loaded")
    print("- All target-encoding settings are stored in configs/features.yaml")

    # -------------------------------------------------------------------------
    # Save updated train and validation feature tables.
    # -------------------------------------------------------------------------
    print(f"\nSaving updated train feature table:\n{TRAIN_FEATURE_PATH}")

    train_df.to_parquet(
        TRAIN_FEATURE_PATH,
        index=False,
    )

    print(
        f"\nSaving updated validation feature table:\n"
        f"{VALIDATION_FEATURE_PATH}"
    )

    validation_df.to_parquet(
        VALIDATION_FEATURE_PATH,
        index=False,
    )

    metadata = {
        "feature_pipeline_version": "v1.0.0",
        "encoding_type": "chronological_out_of_fold_target_encoding",
        "source_config": str(FEATURE_CONFIG_PATH),
        "target_column": "isFraud",
        "encoded_columns": encoded_columns,
        "source_columns": encoding_columns,
        "smoothing": smoothing,
        "min_samples": min_samples,
        "oof_folds": oof_folds,
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "training_global_fraud_rate": encoder.global_fraud_rate_,
        "test_split_loaded": False,
        "train_encoding_rule": (
            "Each chronological fold uses only labels from strictly earlier "
            "timestamp folds."
        ),
        "validation_encoding_rule": (
            "Validation is transformed using mappings fitted on all training "
            "labels only."
        ),
    }

    print(
        f"\nSaving target-encoding metadata:\n"
        f"{TARGET_ENCODING_METADATA_PATH}"
    )

    with TARGET_ENCODING_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    preview_columns = [
        "TransactionID",
        "TransactionDT",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        *encoded_columns,
    ]

    print("\nFirst five validation rows with target encodings:")
    print(
        validation_df[preview_columns]
        .head()
        .to_string(index=False)
    )

    print("\nTarget encoding completed successfully.")


if __name__ == "__main__":
    main()