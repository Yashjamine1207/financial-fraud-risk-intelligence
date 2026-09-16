"""
Point-in-time-safe behavioural feature engineering.

Every calculation in this module uses only transactions that occurred before
the transaction currently being scored. It must never use future rows or the
fraud label to create behavioural features.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class PointInTimeFeaturePipeline:
    """
    Build leakage-safe behavioural features from historical transactions.

    The input data is sorted by TransactionDT before calculation. For each
    transaction, the current transaction is excluded from every historical
    count, mean, standard deviation, and recency calculation.
    """

    def __init__(self, config_path: Path, version: str = "v1.0.0") -> None:
        """
        Initialise the pipeline from the YAML feature configuration.

        Parameters
        ----------
        config_path:
            Path to configs/features.yaml.
        version:
            Version stored with generated feature metadata.
        """
        self.config_path = Path(config_path)
        self.version = version
        self.config = self._load_config()
        self.feature_metadata: list[dict[str, Any]] = []

    def _load_config(self) -> dict[str, Any]:
        """Load and validate the required feature configuration."""
        with self.config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("Feature configuration must contain a YAML mapping.")

        required_keys = {"velocity_windows", "amount_windows", "entity_groups"}
        missing_keys = required_keys.difference(config)

        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Feature configuration is missing required keys: {missing}")

        return config

    @staticmethod
    def _validate_input(
        df: pd.DataFrame,
        required_columns: list[str],
    ) -> None:
        """Check that all required columns are present before feature creation."""
        missing_columns = [column for column in required_columns if column not in df.columns]

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Input data is missing required columns: {missing}")

    @staticmethod
    def _entity_label(group_cols: list[str] | None) -> str:
        """
        Create a stable entity label for feature names.

        Examples
        --------
        None -> global
        ['card1'] -> card1
        ['card1', 'card2'] -> card1_card2
        """
        return "global" if group_cols is None else "_".join(group_cols)

    @staticmethod
    def _sort_transactions(
        df: pd.DataFrame,
        timestamp_col: str,
        transaction_id_col: str,
    ) -> pd.DataFrame:
        """
        Sort deterministically by timestamp and transaction ID.

        TransactionID is used only as a tie-breaker when two transactions have
        the same TransactionDT. It is not used as a model feature.
        """
        return df.sort_values(
            by=[timestamp_col, transaction_id_col],
            kind="mergesort",
        ).reset_index(drop=True)

    @staticmethod
    def _get_group_positions(
        df: pd.DataFrame,
        group_cols: list[str] | None,
    ) -> list[np.ndarray]:
        """
        Return row positions for each entity group.

        If group_cols is None, all rows form one global group.
        dropna=False ensures rows with missing entity values are still handled
        consistently rather than being silently excluded.
        """
        if group_cols is None:
            return [np.arange(len(df), dtype=np.int64)]

        grouped = df.groupby(group_cols, dropna=False, sort=False)
        return [
            np.asarray(row_positions, dtype=np.int64)
            for row_positions in grouped.indices.values()
        ]

    def compute_velocity_features(
        self,
        df: pd.DataFrame,
        transaction_id_col: str = "TransactionID",
        timestamp_col: str = "TransactionDT",
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Create historical transaction-count features.
        For every configured time window, count only transactions that occurred
        at a strictly earlier TransactionDT value.
        Transactions that share the same TransactionDT are processed as one
        batch. They receive identical historical counts and never use one
        another as prior history.
        Examples of created columns:
        - velocity_global_5min
        - velocity_card1_60min
        """
        required_columns = [transaction_id_col, timestamp_col]
        if group_cols is not None:
            required_columns.extend(group_cols)
        self._validate_input(df, required_columns)
        output = self._sort_transactions(
            df=df,
            timestamp_col=timestamp_col,
            transaction_id_col=transaction_id_col,
        )
        entity_label = self._entity_label(group_cols)
        timestamps = output[timestamp_col].to_numpy(dtype=np.int64)
        group_positions = self._get_group_positions(
            df=output,
            group_cols=group_cols,
        )
        feature_names: list[str] = []
        for window_minutes in self.config["velocity_windows"]:
            window_seconds = int(window_minutes) * 60
            feature_name = f"velocity_{entity_label}_{int(window_minutes)}min"
            feature_values = np.zeros(len(output), dtype=np.int32)
            for positions in group_positions:
                group_timestamps = timestamps[positions]
                # left marks the first historical transaction still inside
                # the configured lookback window.
                left = 0
                batch_start = 0
                while batch_start < len(positions):
                    batch_end = batch_start + 1
                    # Find every transaction for this entity that shares the
                    # same timestamp. These rows must not use each other.
                    while (
                        batch_end < len(positions)
                        and group_timestamps[batch_end] == group_timestamps[batch_start]
                    ):
                        batch_end += 1
                    current_timestamp = group_timestamps[batch_start]
                    # Remove historical transactions outside the time window.
                    while (
                        left < batch_start
                        and current_timestamp - group_timestamps[left] > window_seconds
                    ):
                        left += 1
                    # Count transactions from strictly earlier timestamps only.
                    historical_count = batch_start - left
                    # Every transaction in the same timestamp batch receives
                    # the same prior-history count.
                    batch_positions = positions[batch_start:batch_end]
                    feature_values[batch_positions] = historical_count
                    batch_start = batch_end
            output[feature_name] = feature_values
            feature_names.append(feature_name)
        self._record_metadata(
            feature_names=feature_names,
            feature_type="velocity",
            entity=entity_label,
            lookback_windows=[
                f"{int(window_minutes)} minutes"
                for window_minutes in self.config["velocity_windows"]
            ],
        )
        return output

    def compute_amount_features(
        self,
        df: pd.DataFrame,
        amount_col: str = "TransactionAmt",
        transaction_id_col: str = "TransactionID",
        timestamp_col: str = "TransactionDT",
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Create historical amount mean, standard deviation, and z-score features.
        Every feature uses transactions from strictly earlier timestamps only.
        Transactions sharing the same TransactionDT are processed as one batch,
        so same-second transactions never use one another as history.
        This implementation uses a deque and running sum statistics. It is
        efficient enough for the full IEEE-CIS training split.
        """
        required_columns = [
            transaction_id_col,
            timestamp_col,
            amount_col,
        ]
        if group_cols is not None:
            required_columns.extend(group_cols)
        self._validate_input(df, required_columns)
        output = self._sort_transactions(
            df=df,
            timestamp_col=timestamp_col,
            transaction_id_col=transaction_id_col,
        )
        entity_label = self._entity_label(group_cols)
        timestamps = output[timestamp_col].to_numpy(dtype=np.int64)
        amounts = output[amount_col].to_numpy(dtype=np.float64)
        group_positions = self._get_group_positions(
            df=output,
            group_cols=group_cols,
        )
        feature_names: list[str] = []
        for window_hours in self.config["amount_windows"]:
            window_seconds = int(window_hours) * 60 * 60
            mean_name = f"amount_{entity_label}_{int(window_hours)}h_mean"
            std_name = f"amount_{entity_label}_{int(window_hours)}h_std"
            zscore_name = f"amount_{entity_label}_{int(window_hours)}h_zscore"
            means = np.full(len(output), np.nan, dtype=np.float64)
            stds = np.full(len(output), np.nan, dtype=np.float64)
            zscores = np.full(len(output), np.nan, dtype=np.float64)
            for positions in group_positions:
                group_timestamps = timestamps[positions]
                # Stores positions from earlier timestamp batches that are
                # still inside the configured lookback window.
                history = deque()
                # Running statistics make the method O(n), rather than O(n²).
                running_sum = 0.0
                running_sum_squares = 0.0
                batch_start = 0
                while batch_start < len(positions):
                    batch_end = batch_start + 1
                    # Find all entity transactions with this same timestamp.
                    while (
                        batch_end < len(positions)
                        and group_timestamps[batch_end] == group_timestamps[batch_start]
                    ):
                        batch_end += 1
                    current_timestamp = group_timestamps[batch_start]
                    # Remove transactions that are outside this lookback window.
                    while (
                        history
                        and current_timestamp - timestamps[history[0]] > window_seconds
                    ):
                        expired_position = history.popleft()
                        expired_amount = amounts[expired_position]
                        running_sum -= expired_amount
                        running_sum_squares -= expired_amount ** 2
                    prior_count = len(history)
                    batch_positions = positions[batch_start:batch_end]
                    # Calculate statistics from strictly earlier transactions.
                    if prior_count > 0:
                        historical_mean = running_sum / prior_count
                        means[batch_positions] = historical_mean
                        # At least two historical values are needed for a
                        # meaningful standard deviation and z-score.
                        if prior_count > 1:
                            variance = (
                                running_sum_squares / prior_count
                            ) - (historical_mean ** 2)
                            # Protect against tiny negative values caused by
                            # floating-point precision.
                            historical_std = np.sqrt(max(variance, 0.0))
                            stds[batch_positions] = historical_std
                            if historical_std > 0:
                                zscores[batch_positions] = (
                                    amounts[batch_positions] - historical_mean
                                ) / historical_std
                    # Add the whole current timestamp batch only after every
                    # row in it has received its historical features.
                    for row_position in batch_positions:
                        current_amount = amounts[row_position]
                        history.append(row_position)
                        running_sum += current_amount
                        running_sum_squares += current_amount ** 2
                    batch_start = batch_end
            output[mean_name] = means
            output[std_name] = stds
            output[zscore_name] = zscores
            feature_names.extend([
                mean_name,
                std_name,
                zscore_name,
            ])
        self._record_metadata(
            feature_names=feature_names,
            feature_type="amount",
            entity=entity_label,
            lookback_windows=[
                f"{int(window_hours)} hours"
                for window_hours in self.config["amount_windows"]
            ],
        )
        return output

    def compute_recency_feature(
        self,
        df: pd.DataFrame,
        transaction_id_col: str = "TransactionID",
        timestamp_col: str = "TransactionDT",
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Create time since the most recent transaction at a strictly earlier time.
        All rows that share the same TransactionDT receive the same recency
        value. They never use another same-second transaction as the previous
        transaction.
        The first observed timestamp for an entity has a missing value because
        no earlier transaction is available.
        """
        required_columns = [transaction_id_col, timestamp_col]
        if group_cols is not None:
            required_columns.extend(group_cols)
        self._validate_input(df, required_columns)
        output = self._sort_transactions(
            df=df,
            timestamp_col=timestamp_col,
            transaction_id_col=transaction_id_col,
        )
        entity_label = self._entity_label(group_cols)
        feature_name = f"recency_{entity_label}_seconds"
        timestamps = output[timestamp_col].to_numpy(dtype=np.int64)
        recency_values = np.full(len(output), np.nan, dtype=np.float64)
        group_positions = self._get_group_positions(
            df=output,
            group_cols=group_cols,
        )
        for positions in group_positions:
            group_timestamps = timestamps[positions]
            previous_timestamp: int | None = None
            batch_start = 0
            while batch_start < len(positions):
                batch_end = batch_start + 1
                # Identify all transactions with the same timestamp.
                while (
                    batch_end < len(positions)
                    and group_timestamps[batch_end] == group_timestamps[batch_start]
                ):
                    batch_end += 1
                current_timestamp = group_timestamps[batch_start]
                batch_positions = positions[batch_start:batch_end]
                # The first timestamp batch has no strictly earlier history.
                if previous_timestamp is not None:
                    recency_values[batch_positions] = (
                        current_timestamp - previous_timestamp
                    )
                # Update history only after assigning the complete batch.
                previous_timestamp = current_timestamp
                batch_start = batch_end
        output[feature_name] = recency_values
        self._record_metadata(
            feature_names=[feature_name],
            feature_type="recency",
            entity=entity_label,
            lookback_windows=["Time since previous transaction at an earlier timestamp"],
        )
        return output

    def compute_historical_entity_features(
        self,
        df: pd.DataFrame,
        entity_columns: list[str],
        transaction_id_col: str = "TransactionID",
        timestamp_col: str = "TransactionDT",
    ) -> pd.DataFrame:
        """
        Create historical entity-count and new-entity features.

        For each valid entity value, the method creates:

        - history_<entity>_transaction_count:
          Number of transactions previously observed for the same entity.

        - is_new_<entity>:
          1 when the current transaction is the first observed transaction
          for that entity, otherwise 0.

        Point-in-time safety rules:
        - Only transactions with strictly earlier TransactionDT values count
          as historical information.
        - Transactions sharing the same TransactionDT do not use one another.
        - Missing entity values are not treated as one shared entity.

        For rows where an entity value is missing:
        - history_<entity>_transaction_count is NaN
        - is_new_<entity> is NaN

        Parameters
        ----------
        df:
            Transaction-level input data.
        entity_columns:
            One or more columns defining the entity.

            Examples:
            ["card1"]
            ["DeviceInfo"]
            ["R_emaildomain"]
            ["card1", "card2"]
        transaction_id_col:
            Unique transaction identifier used as a deterministic tie-breaker.
        timestamp_col:
            Timestamp representing transaction order.

        Returns
        -------
        pd.DataFrame
            Chronologically sorted input data with historical entity features.
        """
        required_columns = [
            transaction_id_col,
            timestamp_col,
            *entity_columns,
        ]
        self._validate_input(df, required_columns)

        # Sort all records before calculating any history.
        output = self._sort_transactions(
            df=df,
            timestamp_col=timestamp_col,
            transaction_id_col=transaction_id_col,
        )

        entity_label = self._entity_label(entity_columns)

        count_feature_name = f"history_{entity_label}_transaction_count"
        new_entity_feature_name = f"is_new_{entity_label}"

        timestamps = output[timestamp_col].to_numpy(dtype=np.int64)

        # A relationship feature cannot be computed where one or more entity
        # identifiers are missing. For example, a missing DeviceInfo value is
        # not evidence that two transactions used the same device.
        valid_entity_mask = output[entity_columns].notna().all(axis=1)

        # Use floating-point arrays because NaN is meaningful for unknown
        # entity identities.
        historical_counts = np.full(
            shape=len(output),
            fill_value=np.nan,
            dtype=np.float64,
        )
        is_new_entity = np.full(
            shape=len(output),
            fill_value=np.nan,
            dtype=np.float64,
        )

        # Work only with rows whose entity identifier is available.
        valid_output = output.loc[valid_entity_mask].copy()

        if not valid_output.empty:
            valid_grouped = valid_output.groupby(
                entity_columns,
                sort=False,
                dropna=False,
            )

            for _, group_df in valid_grouped:
                # group_df preserves the original chronological row order.
                group_positions = group_df.index.to_numpy(dtype=np.int64)
                group_timestamps = timestamps[group_positions]

                batch_start = 0

                while batch_start < len(group_positions):
                    batch_end = batch_start + 1

                    # Build one batch containing every transaction for the
                    # entity that shares the same timestamp.
                    while (
                        batch_end < len(group_positions)
                        and group_timestamps[batch_end]
                        == group_timestamps[batch_start]
                    ):
                        batch_end += 1

                    batch_positions = group_positions[batch_start:batch_end]

                    # The number of prior rows is the number of transactions
                    # from strictly earlier timestamps in this entity group.
                    historical_counts[batch_positions] = batch_start

                    # If no earlier timestamp exists for this entity, this
                    # timestamp batch represents its first observed use.
                    if batch_start == 0:
                        is_new_entity[batch_positions] = 1.0
                    else:
                        is_new_entity[batch_positions] = 0.0

                    batch_start = batch_end

        output[count_feature_name] = historical_counts
        output[new_entity_feature_name] = is_new_entity

        self._record_metadata(
            feature_names=[
                count_feature_name,
                new_entity_feature_name,
            ],
            feature_type="historical_entity",
            entity=entity_label,
            lookback_windows=[
                "All transactions strictly before scoring time"
            ],
        )

        return output

    def _record_metadata(
        self,
        feature_names: list[str],
        feature_type: str,
        entity: str,
        lookback_windows: list[str],
    ) -> None:
        """Store feature definitions for the Phase 3 feature catalogue."""
        created_at = datetime.now(timezone.utc).isoformat()

        for feature_name in feature_names:
            self.feature_metadata.append(
                {
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "entity": entity,
                    "lookback_windows": lookback_windows,
                    "availability": "Historical transactions strictly before scoring time",
                    "uses_target": False,
                    "version": self.version,
                    "created_at_utc": created_at,
                }
            )

    def get_feature_metadata(self) -> list[dict[str, Any]]:
        """Return metadata for all features produced by this pipeline instance."""
        return self.feature_metadata.copy()

    def save_feature_metadata(self, output_path: Path) -> None:
        """Write the recorded feature metadata to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "feature_pipeline_version": self.version,
            "config_path": str(self.config_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "features": self.feature_metadata,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)