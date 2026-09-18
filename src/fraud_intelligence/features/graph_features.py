"""Leakage-safe, point-in-time graph features for fraud transactions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

GRAPH_FEATURE_COLUMNS = [
    "graph_card1_prior_degree",
    "graph_deviceinfo_prior_card_degree",
    "graph_remail_prior_card_degree",
    "graph_card1_device_prior_edge_count",
    "graph_card1_remail_prior_edge_count",
    "graph_card1_prior_component_size",
    "graph_deviceinfo_prior_component_size",
    "graph_remail_prior_component_size",
    "is_new_card1_device_relationship",
    "is_new_card1_remail_relationship",
]


class UnionFind:
    """Maintain connected-component membership and size incrementally."""

    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}
        self.size: dict[Hashable, int] = {}

    def contains(self, node: Hashable) -> bool:
        return node in self.parent

    def add(self, node: Hashable) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def find(self, node: Hashable) -> Hashable:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, first: Hashable, second: Hashable) -> None:
        self.add(first)
        self.add(second)

        first_root = self.find(first)
        second_root = self.find(second)

        if first_root == second_root:
            return

        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root

        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]

    def component_size(self, node: Hashable) -> int:
        if node not in self.parent:
            return 0
        return self.size[self.find(node)]


@dataclass(frozen=True)
class GraphFeatureConfig:
    """Column names used to construct point-in-time graph features."""

    timestamp_column: str = "TransactionDT"
    card_column: str = "card1"
    device_column: str = "DeviceInfo"
    recipient_email_column: str = "R_emaildomain"


class PointInTimeGraphFeatureBuilder:
    """Build historical entity-relationship features without future leakage."""

    def __init__(self, config: GraphFeatureConfig | None = None) -> None:
        self.config = config or GraphFeatureConfig()
        self._reset_state()

    def _reset_state(self) -> None:
        self.graph = UnionFind()
        self.neighbours: dict[str, set[str]] = defaultdict(set)
        self.edge_counts: Counter[tuple[str, str]] = Counter()

    @staticmethod
    def _is_present(value: object) -> bool:
        return not pd.isna(value)

    @staticmethod
    def _node(prefix: str, value: object) -> str | None:
        if not PointInTimeGraphFeatureBuilder._is_present(value):
            return None
        return f"{prefix}::{value}"

    def _validate_input(self, transactions: pd.DataFrame) -> None:
        required_columns = [
            self.config.timestamp_column,
            self.config.card_column,
            self.config.device_column,
            self.config.recipient_email_column,
        ]
        missing_columns = [
            column for column in required_columns if column not in transactions.columns
        ]

        if missing_columns:
            raise ValueError(
                "Cannot build graph features because required columns are missing: "
                f"{missing_columns}"
            )

        if transactions[self.config.timestamp_column].isna().any():
            raise ValueError(
                f"{self.config.timestamp_column} contains missing values. "
                "A chronological graph cannot be built without transaction time."
            )

    def _relationship_features(
        self,
        card_node: str | None,
        entity_node: str | None,
    ) -> tuple[float, float, float, float]:
        if card_node is None or entity_node is None:
            return (np.nan, np.nan, np.nan, np.nan)

        edge_count = float(self.edge_counts[(card_node, entity_node)])
        is_new_relationship = float(edge_count == 0)

        return (
            float(len(self.neighbours[entity_node])),
            edge_count,
            float(self.graph.component_size(entity_node)),
            is_new_relationship,
        )

    def _score_row(self, row: pd.Series) -> dict[str, float]:
        card_node = self._node("card1", row[self.config.card_column])
        device_node = self._node("device", row[self.config.device_column])
        remail_node = self._node(
            "remail",
            row[self.config.recipient_email_column],
        )

        card_degree = (
            float(len(self.neighbours[card_node])) if card_node is not None else np.nan
        )
        card_component_size = (
            float(self.graph.component_size(card_node))
            if card_node is not None
            else np.nan
        )

        (
            device_prior_card_degree,
            card_device_edge_count,
            device_component_size,
            is_new_card_device,
        ) = self._relationship_features(card_node, device_node)

        (
            remail_prior_card_degree,
            card_remail_edge_count,
            remail_component_size,
            is_new_card_remail,
        ) = self._relationship_features(card_node, remail_node)

        return {
            "graph_card1_prior_degree": card_degree,
            "graph_deviceinfo_prior_card_degree": device_prior_card_degree,
            "graph_remail_prior_card_degree": remail_prior_card_degree,
            "graph_card1_device_prior_edge_count": card_device_edge_count,
            "graph_card1_remail_prior_edge_count": card_remail_edge_count,
            "graph_card1_prior_component_size": card_component_size,
            "graph_deviceinfo_prior_component_size": device_component_size,
            "graph_remail_prior_component_size": remail_component_size,
            "is_new_card1_device_relationship": is_new_card_device,
            "is_new_card1_remail_relationship": is_new_card_remail,
        }

    def _update_graph(self, rows: Iterable[pd.Series]) -> None:
        for row in rows:
            card_node = self._node("card1", row[self.config.card_column])
            device_node = self._node("device", row[self.config.device_column])
            remail_node = self._node(
                "remail",
                row[self.config.recipient_email_column],
            )

            if card_node is None:
                continue

            for entity_node in (device_node, remail_node):
                if entity_node is None:
                    continue

                self.graph.union(card_node, entity_node)
                self.neighbours[card_node].add(entity_node)
                self.neighbours[entity_node].add(card_node)
                self.edge_counts[(card_node, entity_node)] += 1

    def transform(
        self,
        transactions: pd.DataFrame,
        reset_state: bool = True,
    ) -> pd.DataFrame:
        """Return input rows with features based exclusively on prior graph state.

        Parameters
        ----------
        transactions:
            Transaction-level data containing `TransactionDT`, `card1`,
            `DeviceInfo`, and `R_emaildomain`.
        reset_state:
            If True, start with an empty historical graph. Use False only when
            sequentially scoring later chronological periods after earlier periods
            have already been transformed by this same builder.

        Returns
        -------
        pd.DataFrame
            The original transactions in their original row order, plus the
            configured point-in-time graph feature columns.
        """
        self._validate_input(transactions)

        if reset_state:
            self._reset_state()

        if transactions.empty:
            output = transactions.copy()
            for column in GRAPH_FEATURE_COLUMNS:
                output[column] = pd.Series(dtype="float64")
            return output

        original_index_name = "__graph_original_row_order__"
        working = transactions.copy()
        working[original_index_name] = np.arange(len(working))

        working = working.sort_values(
            self.config.timestamp_column,
            kind="mergesort",
        )

        feature_records: list[dict[str, float]] = []
        record_positions: list[int] = []

        timestamp_column = self.config.timestamp_column

        for _, timestamp_batch in working.groupby(timestamp_column, sort=False):
            batch_rows = [
                row
                for _, row in timestamp_batch.drop(columns=[original_index_name]).iterrows()
            ]

            for position, row in zip(
                timestamp_batch[original_index_name].to_numpy(),
                batch_rows,
                strict=True,
            ):
                feature_records.append(self._score_row(row))
                record_positions.append(int(position))

            self._update_graph(batch_rows)

        feature_frame = pd.DataFrame(feature_records, columns=GRAPH_FEATURE_COLUMNS)
        feature_frame[original_index_name] = record_positions
        feature_frame = feature_frame.sort_values(original_index_name)
        feature_frame = feature_frame.drop(columns=[original_index_name]).reset_index(
            drop=True
        )

        output = transactions.reset_index(drop=True).copy()

        for column in GRAPH_FEATURE_COLUMNS:
            output[column] = feature_frame[column].astype("float64")

        return output