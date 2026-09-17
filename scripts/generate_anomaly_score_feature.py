"""Generate leakage-safe Isolation Forest anomaly features for Phase 6.

Protocol
--------
1. Training anomaly scores:
   Use chronological out-of-fold scoring. Each fold is scored by an Isolation
   Forest fitted only on earlier training folds. The first fold has no earlier
   history, so its anomaly score is intentionally missing.

2. Validation and final-test anomaly scores:
   Fit one Isolation Forest on the complete chronological training period only,
   then transform validation and final-test data.

3. Fraud labels are never used by Isolation Forest.

The locked final test split is not used to fit the anomaly model. It is only
transformed because Phase 5 has already produced the locked test feature table.
No model selection occurs in this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from fraud_intelligence.features.anomaly import (
    ANOMALY_SCORE_COLUMN,
    AnomalyScorerConfig,
    TimeSafeAnomalyScorer,
)
from fraud_intelligence.models.data_contract import load_modelling_data

OUTPUT_DIRECTORY = Path("data/features/anomaly")
TEST_FEATURES_PATH = Path(
    "data/features/point_in_time/test_behavioural_and_encoded_features.parquet"
)

N_CHRONOLOGICAL_FOLDS = 5


def create_chronological_fold_ids(
    transaction_timestamps: pd.Series,
    n_folds: int,
) -> pd.Series:
    """Create timestamp-safe chronological fold IDs.

    Rows with the same TransactionDT always receive the same fold ID. This
    prevents same-second rows from appearing in both fit and score partitions.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    unique_timestamps = sorted(transaction_timestamps.unique())

    if len(unique_timestamps) < n_folds:
        raise ValueError(
            "The number of distinct timestamps must be at least the number "
            "of chronological folds."
        )

    timestamp_to_fold: dict[int | float, int] = {}

    for position, timestamp in enumerate(unique_timestamps):
        fold_id = min((position * n_folds) // len(unique_timestamps), n_folds - 1)
        timestamp_to_fold[timestamp] = fold_id

    return transaction_timestamps.map(timestamp_to_fold).astype("int8")


def generate_training_oof_anomaly_scores(
    X_train: pd.DataFrame,
    transaction_timestamps: pd.Series,
    config: AnomalyScorerConfig,
    n_folds: int,
) -> tuple[pd.Series, pd.Series]:
    """Generate chronological out-of-fold anomaly scores for training rows.

    Fold 0 has no earlier observations on which to fit an anomaly model.
    Therefore, its scores remain missing. The Phase 4 preprocessing pipeline
    will handle this safely with its existing numeric median imputation plus
    missingness indicators.
    """
    fold_ids = create_chronological_fold_ids(
        transaction_timestamps=transaction_timestamps,
        n_folds=n_folds,
    )

    scores = pd.Series(
        data=float("nan"),
        index=X_train.index,
        name=ANOMALY_SCORE_COLUMN,
        dtype="float64",
    )

    for fold_id in range(1, n_folds):
        fit_mask = fold_ids < fold_id
        score_mask = fold_ids == fold_id

        scorer = TimeSafeAnomalyScorer(config=config)
        scorer.fit(X_train.loc[fit_mask])
        scores.loc[score_mask] = scorer.transform(X_train.loc[score_mask])

        print(
            f"Completed chronological anomaly fold {fold_id + 1}/{n_folds}: "
            f"fit rows={int(fit_mask.sum()):,}, "
            f"scored rows={int(score_mask.sum()):,}"
        )

    return scores, fold_ids


def load_test_feature_matrix(
    feature_columns: list[str],
) -> pd.DataFrame | None:
    """Load and align final-test features only if the Phase 5 file exists."""
    if not TEST_FEATURES_PATH.exists():
        return None

    test_dataframe = pd.read_parquet(TEST_FEATURES_PATH)

    missing_columns = set(feature_columns).difference(test_dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Final-test feature table is missing Phase 4 model feature column(s): " f"{missing}"
        )

    return test_dataframe.loc[:, feature_columns].copy()


def main() -> None:
    """Generate and save train, validation, and optional test anomaly scores."""
    load_dotenv()

    modelling_data = load_modelling_data()
    feature_columns = modelling_data.feature_columns

    # The data contract deliberately excludes TransactionID, TransactionDT,
    # isFraud, and high-missing columns from X_train / X_validation.
    X_train = modelling_data.X_train.loc[:, feature_columns].copy()
    X_validation = modelling_data.X_validation.loc[:, feature_columns].copy()

    # TransactionDT is loaded separately only to make timestamp-safe training
    # folds. It is never supplied to Isolation Forest as a model feature.
    train_full_dataframe = pd.read_parquet(modelling_data.config["data"]["train_path"])
    train_timestamps = train_full_dataframe[
        modelling_data.config["data"]["timestamp_column"]
    ].copy()

    if len(train_timestamps) != len(X_train):
        raise ValueError("Training timestamp count does not match the modelling feature count.")

    scorer_config = AnomalyScorerConfig()

    print("Generating chronological out-of-fold training anomaly scores...")
    train_scores, train_fold_ids = generate_training_oof_anomaly_scores(
        X_train=X_train,
        transaction_timestamps=train_timestamps,
        config=scorer_config,
        n_folds=N_CHRONOLOGICAL_FOLDS,
    )

    print("Fitting final Isolation Forest on the complete training period...")
    final_train_scorer = TimeSafeAnomalyScorer(config=scorer_config)
    final_train_scorer.fit(X_train)

    print("Scoring chronological validation data with train-fitted model...")
    validation_scores = final_train_scorer.transform(X_validation)

    test_scores: pd.Series | None = None
    X_test = load_test_feature_matrix(feature_columns=feature_columns)

    if X_test is not None:
        print("Scoring final test data with train-fitted model...")
        test_scores = final_train_scorer.transform(X_test)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    train_scores.to_frame().to_parquet(
        OUTPUT_DIRECTORY / "train_anomaly_scores.parquet",
        index=False,
    )
    validation_scores.to_frame().to_parquet(
        OUTPUT_DIRECTORY / "val_anomaly_scores.parquet",
        index=False,
    )

    if test_scores is not None:
        test_scores.to_frame().to_parquet(
            OUTPUT_DIRECTORY / "test_anomaly_scores.parquet",
            index=False,
        )

    metadata = {
        "feature_name": ANOMALY_SCORE_COLUMN,
        "model": "IsolationForest",
        "anomaly_feature_version": "isolation-forest-v1.0.0",
        "fit_protocol": {
            "training": (
                "chronological out-of-fold scoring; each scoring fold uses only "
                "earlier training folds; first fold intentionally missing"
            ),
            "validation": "complete training period only",
            "test": "complete training period only",
            "fraud_labels_used": False,
            "same_timestamp_split_across_folds": False,
        },
        "chronological_training_folds": N_CHRONOLOGICAL_FOLDS,
        "config": {
            "n_estimators": scorer_config.n_estimators,
            "contamination": scorer_config.contamination,
            "max_samples": scorer_config.max_samples,
            "random_state": scorer_config.random_state,
            "n_jobs": scorer_config.n_jobs,
        },
        "train_rows": len(train_scores),
        "validation_rows": len(validation_scores),
        "test_rows": len(test_scores) if test_scores is not None else None,
        "train_missing_score_count": int(train_scores.isna().sum()),
        "train_fold_row_counts": {
            str(fold_id): int((train_fold_ids == fold_id).sum())
            for fold_id in range(N_CHRONOLOGICAL_FOLDS)
        },
    }

    metadata_path = OUTPUT_DIRECTORY / "anomaly_feature_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nLeakage-safe anomaly feature generation completed.")
    print(f"Train scores: {len(train_scores):,}")
    print(f"Validation scores: {len(validation_scores):,}")
    print(
        f"Test scores: {len(test_scores):,}"
        if test_scores is not None
        else "Test scores: not generated"
    )
    print(
        f"Training rows with intentionally missing early-history score: {train_scores.isna().sum():,}"
    )
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
