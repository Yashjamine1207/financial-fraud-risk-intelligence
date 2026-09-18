"""Point-in-time transaction sequence dataset construction."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceFeatureConfig:
    """Configuration for compact historical transaction sequences."""

    transaction_id_column: str = "TransactionID"
    timestamp_column: str = "TransactionDT"
    target_column: str = "isFraud"
    entity_column: str = "card1"
    amount_column: str = "TransactionAmt"
    max_sequence_length: int = 10
    padding_value: float = 0.0


@dataclass(frozen=True)
class SequenceDataset:
    """Model-ready tensors and traceability metadata."""

    sequences: np.ndarray
    labels: np.ndarray
    transaction_ids: np.ndarray
    timestamps: np.ndarray
    entity_history_lengths: np.ndarray


class PointInTimeSequenceBuilder:
    """Build sequences using only current and strictly prior entity events."""

    def __init__(self, config: SequenceFeatureConfig | None = None) -> None:
        self.config = config or SequenceFeatureConfig()
        self._history: dict[Hashable, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.config.max_sequence_length - 1)
        )
        self._last_timestamp_by_entity: dict[Hashable, float] = {}

    def reset_state(self) -> None:
        """Clear all stored historical entity state."""
        self._history.clear()
        self._last_timestamp_by_entity.clear()

    def _validate_input(self, dataframe: pd.DataFrame) -> None:
        required_columns = {
            self.config.transaction_id_column,
            self.config.timestamp_column,
            self.config.target_column,
            self.config.entity_column,
            self.config.amount_column,
        }
        missing_columns = sorted(required_columns.difference(dataframe.columns))

        if missing_columns:
            raise ValueError(
                "Cannot build sequence features because required columns are missing: "
                f"{missing_columns}"
            )

        if dataframe.empty:
            raise ValueError("Cannot build sequence features from an empty dataframe.")

        if dataframe[self.config.transaction_id_column].isna().any():
            raise ValueError("TransactionID must not contain missing values.")

        if dataframe[self.config.transaction_id_column].duplicated().any():
            raise ValueError("TransactionID must be unique.")

        if dataframe[self.config.timestamp_column].isna().any():
            raise ValueError("TransactionDT must not contain missing values.")

        if dataframe[self.config.amount_column].isna().any():
            raise ValueError("TransactionAmt must not contain missing values.")

        if (dataframe[self.config.amount_column] < 0).any():
            raise ValueError("TransactionAmt must not contain negative values.")

        valid_labels = set(dataframe[self.config.target_column].dropna().unique())
        if not valid_labels.issubset({0, 1}):
            raise ValueError("isFraud must contain only binary values 0 and 1.")

        if self.config.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2.")

    @staticmethod
    def _normalise_entity(value: object) -> str | None:
        """Return a stable entity key or None for unavailable entity values."""
        if pd.isna(value):
            return None
        return f"card1::{value}"

    def _event_features(
        self,
        transaction_amount: float,
        timestamp: float,
        entity: str | None,
    ) -> np.ndarray:
        """Create current-event features using only available prior state."""
        log_amount = float(np.log1p(transaction_amount))

        if entity is None or entity not in self._last_timestamp_by_entity:
            log_time_gap = 0.0
        else:
            time_gap = max(
                timestamp - self._last_timestamp_by_entity[entity],
                0.0,
            )
            log_time_gap = float(np.log1p(time_gap))

        return np.asarray([log_amount, log_time_gap], dtype=np.float32)

    def _make_padded_sequence(
        self,
        entity: str | None,
        current_event_features: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Create one left-padded sequence ending with the current event."""
        sequence = np.full(
            (
                self.config.max_sequence_length,
                len(current_event_features),
            ),
            self.config.padding_value,
            dtype=np.float32,
        )

        historical_events = [] if entity is None else list(self._history[entity])
        combined_events = [*historical_events, current_event_features]
        combined_events = combined_events[-self.config.max_sequence_length :]

        sequence[-len(combined_events) :] = np.asarray(
            combined_events,
            dtype=np.float32,
        )

        return sequence, len(historical_events)

    def _update_history(
        self,
        entity: str | None,
        timestamp: float,
        event_features: np.ndarray,
    ) -> None:
        """Update history only after every row in a timestamp batch was scored."""
        if entity is None:
            return

        self._history[entity].append(event_features)
        self._last_timestamp_by_entity[entity] = timestamp

    def transform(
        self,
        dataframe: pd.DataFrame,
        reset_state: bool = True,
    ) -> SequenceDataset:
        """Create sequence tensors in original dataframe row order.

        Every transaction is scored with:
        - its own amount and its own prior-entity time gap;
        - up to max_sequence_length - 1 strictly earlier events for card1;
        - no other same-timestamp event information;
        - no future events;
        - no fraud labels as model inputs.

        Set reset_state=False only when processing a later chronological period
        immediately after an earlier period with this same builder instance.
        """
        self._validate_input(dataframe)

        if reset_state:
            self.reset_state()

        original_order_column = "__sequence_original_order__"
        working = dataframe.copy()
        working[original_order_column] = np.arange(len(working))

        working = working.sort_values(
            self.config.timestamp_column,
            kind="mergesort",
        )

        records: list[tuple[int, np.ndarray, int, int, float]] = []

        for _, timestamp_batch in working.groupby(
            self.config.timestamp_column,
            sort=False,
        ):
            pending_updates: list[tuple[str | None, float, np.ndarray]] = []

            for _, row in timestamp_batch.iterrows():
                entity = self._normalise_entity(row[self.config.entity_column])
                timestamp = float(row[self.config.timestamp_column])
                amount = float(row[self.config.amount_column])

                current_event_features = self._event_features(
                    transaction_amount=amount,
                    timestamp=timestamp,
                    entity=entity,
                )

                sequence, history_length = self._make_padded_sequence(
                    entity=entity,
                    current_event_features=current_event_features,
                )

                records.append(
                    (
                        int(row[original_order_column]),
                        sequence,
                        int(row[self.config.target_column]),
                        int(row[self.config.transaction_id_column]),
                        timestamp,
                        history_length,
                    )
                )

                pending_updates.append(
                    (
                        entity,
                        timestamp,
                        current_event_features,
                    )
                )

            for entity, timestamp, event_features in pending_updates:
                self._update_history(
                    entity=entity,
                    timestamp=timestamp,
                    event_features=event_features,
                )

        records.sort(key=lambda record: record[0])

        sequences = np.stack([record[1] for record in records]).astype(np.float32)
        labels = np.asarray([record[2] for record in records], dtype=np.int8)
        transaction_ids = np.asarray([record[3] for record in records])
        timestamps = np.asarray([record[4] for record in records], dtype=np.float64)
        history_lengths = np.asarray([record[5] for record in records], dtype=np.int16)

        return SequenceDataset(
            sequences=sequences,
            labels=labels,
            transaction_ids=transaction_ids,
            timestamps=timestamps,
            entity_history_lengths=history_lengths,
        )