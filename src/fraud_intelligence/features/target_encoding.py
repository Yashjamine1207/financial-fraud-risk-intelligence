"""
Time-safe target encoding for fraud-risk features.

Target encoding replaces a category with a smoothed historical fraud rate.

Leakage-control rules:
- Validation data must only use mappings learned from training data.
- Training data must use out-of-fold chronological encodings.
- A row must never use its own isFraud label in its target-encoded value.
- Same-timestamp rows must not use one another's labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY = "__UNKNOWN__"


@dataclass
class TimeSafeTargetEncoder:
    """
    Create smoothed, leakage-safe target encodings for categorical variables.

    Parameters
    ----------
    columns:
        Categorical columns to encode.
    smoothing:
        Higher values shrink rare-category fraud rates more strongly towards
        the overall historical fraud rate.
    min_samples:
        Categories with fewer than this number of historical observations use
        the global historical fraud rate instead of a category-specific rate.
    """

    columns: list[str]
    smoothing: float = 10.0
    min_samples: int = 20

    global_fraud_rate_: float | None = field(default=None, init=False)
    mappings_: dict[str, pd.Series] = field(default_factory=dict, init=False)

    @staticmethod
    def _prepare_category_values(values: pd.Series) -> pd.Series:
        """
        Convert category values to safe strings.

        Missing source values become a dedicated category because missingness
        itself is an available transaction-time signal. Unseen categories at
        transform time use the global fraud rate.
        """
        return (
            values.astype("string")
            .fillna(MISSING_CATEGORY)
            .replace("", MISSING_CATEGORY)
        )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Ensure every requested target-encoding column exists."""
        missing_columns = [
            column
            for column in self.columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Target-encoding columns missing from input data: "
                f"{missing_columns}"
            )

    @staticmethod
    def _validate_target(target: pd.Series) -> None:
        """Validate that the target contains binary fraud labels only."""
        valid_target_values = set(target.dropna().unique())

        if not valid_target_values.issubset({0, 1}):
            raise ValueError(
                "Target encoding requires binary isFraud values: 0 and 1."
            )

    def fit(
        self,
        df: pd.DataFrame,
        target: pd.Series,
    ) -> TimeSafeTargetEncoder:
        """
        Learn category fraud-rate mappings from labelled training data only.

        This method must never be fitted using validation or final-test labels.
        """
        self._validate_columns(df)
        self._validate_target(target)

        if len(df) != len(target):
            raise ValueError(
                "Input features and target must contain the same number of rows."
            )

        target = target.reset_index(drop=True).astype(float)
        input_df = df.reset_index(drop=True)

        self.global_fraud_rate_ = float(target.mean())
        self.mappings_ = {}

        for column in self.columns:
            category_values = self._prepare_category_values(input_df[column])

            statistics_df = pd.DataFrame(
                {
                    "category": category_values,
                    "isFraud": target,
                }
            )

            category_stats = statistics_df.groupby(
                "category",
                dropna=False,
            )["isFraud"].agg(
                transaction_count="count",
                fraud_sum="sum",
            )

            raw_fraud_rate = (
                category_stats["fraud_sum"]
                / category_stats["transaction_count"]
            )

            # Smoothed fraud rate:
            # rare categories move towards the global historical fraud rate.
            smoothed_fraud_rate = (
                (
                    category_stats["transaction_count"] * raw_fraud_rate
                    + self.smoothing * self.global_fraud_rate_
                )
                / (category_stats["transaction_count"] + self.smoothing)
            )

            # Very rare categories receive the global historical rate.
            smoothed_fraud_rate = smoothed_fraud_rate.where(
                category_stats["transaction_count"] >= self.min_samples,
                self.global_fraud_rate_,
            )

            self.mappings_[column] = smoothed_fraud_rate

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mappings learned previously from training data.

        Unseen categories use the global training fraud rate. This prevents
        missing values during model scoring while avoiding use of future labels.
        """
        if self.global_fraud_rate_ is None or not self.mappings_:
            raise RuntimeError(
                "Target encoder is not fitted. Call fit() before transform()."
            )

        self._validate_columns(df)

        output = pd.DataFrame(index=df.index)

        for column in self.columns:
            category_values = self._prepare_category_values(df[column])

            encoded_values = category_values.map(self.mappings_[column])

            # An unseen validation category has no training fraud history.
            # Use the training-period global fraud rate only.
            output[f"{column}_target_encoded"] = encoded_values.fillna(
                self.global_fraud_rate_
            ).astype(float)

        return output

    def fit_transform_oof(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        timestamp_col: str = "TransactionDT",
        transaction_id_col: str = "TransactionID",
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """
        Create chronological out-of-fold target encodings for training data.

        For each time block:
        - The block's labels are never used to encode that block.
        - Only strictly earlier timestamp blocks provide target history.
        - Rows sharing a timestamp remain in the same fold, so they cannot
          use one another's fraud labels.

        The earliest chronological fold has no previous labels available.
        Its encoding remains NaN and will later be handled by the model's
        preprocessing pipeline using a training-only imputer.
        """
        self._validate_columns(df)
        self._validate_target(target)

        if timestamp_col not in df.columns:
            raise ValueError(
                f"Input data is missing required timestamp column: {timestamp_col}"
            )

        if transaction_id_col not in df.columns:
            raise ValueError(
                "Input data is missing required transaction ID column: "
                f"{transaction_id_col}"
            )

        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        if len(df) != len(target):
            raise ValueError(
                "Input features and target must contain the same number of rows."
            )

        working_df = df.copy()
        working_df["_target_encoding_target"] = (
            target.reset_index(drop=True).astype(float)
        )

        # Stable sorting makes behaviour reproducible for equal timestamps.
        working_df = working_df.sort_values(
            by=[timestamp_col, transaction_id_col],
            kind="mergesort",
        ).reset_index()

        original_index_column = "index"

        output = pd.DataFrame(
            index=working_df[original_index_column],
            columns=[
                f"{column}_target_encoded"
                for column in self.columns
            ],
            dtype=float,
        )

        # Split unique timestamps, not individual rows. This ensures no
        # same-second transactions are split between training and encoding sets.
        unique_timestamps = working_df[timestamp_col].drop_duplicates().to_numpy()

        timestamp_folds = np.array_split(
            unique_timestamps,
            min(n_splits, len(unique_timestamps)),
        )

        for fold_number, fold_timestamps in enumerate(timestamp_folds):
            if len(fold_timestamps) == 0:
                continue

            current_fold_start_time = fold_timestamps[0]

            validation_mask = working_df[timestamp_col].isin(fold_timestamps)

            # Strictly earlier timestamps only. Same-timestamp data never
            # contributes fraud labels to the encoding of the current fold.
            historical_mask = (
                working_df[timestamp_col] < current_fold_start_time
            )

            validation_rows = working_df.loc[validation_mask]
            historical_rows = working_df.loc[historical_mask]

            # The first fold has no historical labels. Leave it as NaN.
            if historical_rows.empty:
                continue

            fold_encoder = TimeSafeTargetEncoder(
                columns=self.columns,
                smoothing=self.smoothing,
                min_samples=self.min_samples,
            )

            fold_encoder.fit(
                df=historical_rows,
                target=historical_rows["_target_encoding_target"],
            )

            encoded_validation_rows = fold_encoder.transform(validation_rows)

            encoded_validation_rows.index = validation_rows[
                original_index_column
            ]

            output.loc[
                encoded_validation_rows.index,
                encoded_validation_rows.columns,
            ] = encoded_validation_rows

        # Fit the final mapping on all training rows. This final fitted encoder
        # is used later to encode validation data, using training labels only.
        self.fit(
            df=df,
            target=target,
        )

        return output.sort_index()