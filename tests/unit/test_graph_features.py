import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.features.graph_features import (
    GRAPH_FEATURE_COLUMNS,
    PointInTimeGraphFeatureBuilder,
)


def test_same_timestamp_rows_cannot_update_each_other() -> None:
    transactions = pd.DataFrame(
        {
            "TransactionDT": [10, 20, 20, 30],
            "card1": [1001, 1001, 2002, 2002],
            "DeviceInfo": ["device_a", "device_a", "device_a", "device_a"],
            "R_emaildomain": ["gmail.com", "gmail.com", "yahoo.com", "yahoo.com"],
        }
    )

    result = PointInTimeGraphFeatureBuilder().transform(transactions)

    assert result.loc[0, "graph_card1_prior_degree"] == 0.0
    assert result.loc[0, "graph_card1_device_prior_edge_count"] == 0.0
    assert result.loc[0, "is_new_card1_device_relationship"] == 1.0

    assert result.loc[1, "graph_card1_prior_degree"] == 2.0
    assert result.loc[1, "graph_card1_device_prior_edge_count"] == 1.0
    assert result.loc[1, "is_new_card1_device_relationship"] == 0.0

    assert result.loc[2, "graph_deviceinfo_prior_card_degree"] == 1.0
    assert result.loc[2, "graph_card1_device_prior_edge_count"] == 0.0
    assert result.loc[2, "is_new_card1_device_relationship"] == 1.0

    assert result.loc[3, "graph_deviceinfo_prior_card_degree"] == 2.0
    assert result.loc[3, "graph_card1_device_prior_edge_count"] == 1.0
    assert result.loc[3, "is_new_card1_device_relationship"] == 0.0


def test_missing_entities_never_create_shared_graph_nodes() -> None:
    transactions = pd.DataFrame(
        {
            "TransactionDT": [10, 20],
            "card1": [1001, 2002],
            "DeviceInfo": [np.nan, np.nan],
            "R_emaildomain": [np.nan, np.nan],
        }
    )

    result = PointInTimeGraphFeatureBuilder().transform(transactions)

    assert np.isnan(result.loc[0, "graph_deviceinfo_prior_card_degree"])
    assert np.isnan(result.loc[1, "graph_deviceinfo_prior_card_degree"])
    assert np.isnan(result.loc[0, "graph_remail_prior_card_degree"])
    assert np.isnan(result.loc[1, "graph_remail_prior_card_degree"])

    assert result.loc[0, "graph_card1_prior_degree"] == 0.0
    assert result.loc[1, "graph_card1_prior_degree"] == 0.0


def test_later_period_can_use_only_prior_period_graph_history() -> None:
    train_transactions = pd.DataFrame(
        {
            "TransactionDT": [10],
            "card1": [1001],
            "DeviceInfo": ["device_a"],
            "R_emaildomain": ["gmail.com"],
        }
    )

    validation_transactions = pd.DataFrame(
        {
            "TransactionDT": [20],
            "card1": [1001],
            "DeviceInfo": ["device_a"],
            "R_emaildomain": ["gmail.com"],
        }
    )

    builder = PointInTimeGraphFeatureBuilder()

    builder.transform(train_transactions, reset_state=True)
    validation_result = builder.transform(validation_transactions, reset_state=False)

    assert validation_result.loc[0, "graph_card1_prior_degree"] == 2.0
    assert validation_result.loc[0, "graph_card1_device_prior_edge_count"] == 1.0
    assert validation_result.loc[0, "graph_card1_remail_prior_edge_count"] == 1.0
    assert validation_result.loc[0, "is_new_card1_device_relationship"] == 0.0
    assert validation_result.loc[0, "is_new_card1_remail_relationship"] == 0.0


def test_output_preserves_original_input_order() -> None:
    transactions = pd.DataFrame(
        {
            "TransactionDT": [30, 10, 20],
            "card1": [1001, 1001, 1001],
            "DeviceInfo": ["device_a", "device_a", "device_a"],
            "R_emaildomain": ["gmail.com", "gmail.com", "gmail.com"],
        }
    )

    result = PointInTimeGraphFeatureBuilder().transform(transactions)

    assert result["TransactionDT"].tolist() == [30, 10, 20]
    assert result.loc[0, "graph_card1_device_prior_edge_count"] == 2.0
    assert result.loc[1, "graph_card1_device_prior_edge_count"] == 0.0
    assert result.loc[2, "graph_card1_device_prior_edge_count"] == 1.0


def test_missing_required_columns_raise_meaningful_error() -> None:
    transactions = pd.DataFrame(
        {
            "TransactionDT": [10],
            "card1": [1001],
            "DeviceInfo": ["device_a"],
        }
    )

    with pytest.raises(ValueError, match="R_emaildomain"):
        PointInTimeGraphFeatureBuilder().transform(transactions)


def test_empty_input_returns_all_graph_feature_columns() -> None:
    transactions = pd.DataFrame(
        {
            "TransactionDT": pd.Series(dtype="float64"),
            "card1": pd.Series(dtype="float64"),
            "DeviceInfo": pd.Series(dtype="object"),
            "R_emaildomain": pd.Series(dtype="object"),
        }
    )

    result = PointInTimeGraphFeatureBuilder().transform(transactions)

    assert result.empty
    assert all(column in result.columns for column in GRAPH_FEATURE_COLUMNS)