"""
Tests for point-in-time-safe behavioural feature engineering.

These tests use a tiny controlled dataset to prove that:
- Current transactions never use themselves as history.
- Same-second transactions do not use one another as history.
- Only strictly earlier timestamps contribute to historical features.
- Missing entity identifiers are not grouped together.
- isFraud is never used to create behavioural features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fraud_intelligence.features import PointInTimeFeaturePipeline


@pytest.fixture
def feature_config_path(tmp_path: Path) -> Path:
    """
    Create a temporary minimal feature configuration for tests.

    The production configuration remains in configs/features.yaml.
    This small temporary configuration makes the unit tests fast and focused.
    """
    config_path = tmp_path / "features_test.yaml"

    config_path.write_text(
        """
velocity_windows:
  - 5

amount_windows:
  - 1

entity_groups:
  - card1
""".strip(),
        encoding="utf-8",
    )

    return config_path


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """
    Return a small chronological transaction dataset.

    Important test cases:
    - TransactionID 1 and 2 have the same timestamp: 100 seconds.
    - They must not use each other as history.
    - TransactionID 3 occurs later at 110 seconds.
    - Rows 5 and 6 have missing DeviceInfo values.
    """
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [100, 100, 110, 130, 150, 160],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "card1": [1111, 1111, 1111, 2222, 1111, 1111],
            "DeviceInfo": [
                "device_a",
                "device_a",
                "device_a",
                "device_b",
                np.nan,
                np.nan,
            ],
            "isFraud": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def pipeline(feature_config_path: Path) -> PointInTimeFeaturePipeline:
    """Create a test version of the point-in-time feature pipeline."""
    return PointInTimeFeaturePipeline(
        config_path=feature_config_path,
        version="test-v1.0.0",
    )


def test_global_velocity_excludes_same_timestamp_transactions(
    pipeline: PointInTimeFeaturePipeline,
    sample_transactions: pd.DataFrame,
) -> None:
    """
    Same-second transactions must not count one another as prior history.

    TransactionID 1 and 2 both occur at timestamp 100. Therefore both must
    have global velocity zero. TransactionID 3 occurs at timestamp 110 and
    can use both earlier transactions as history.
    """
    result = pipeline.compute_velocity_features(
        df=sample_transactions,
        group_cols=None,
    )

    velocity_by_id = result.set_index("TransactionID")["velocity_global_5min"]

    assert velocity_by_id.loc[1] == 0
    assert velocity_by_id.loc[2] == 0
    assert velocity_by_id.loc[3] == 2


def test_card_velocity_uses_only_prior_timestamps(
    pipeline: PointInTimeFeaturePipeline,
    sample_transactions: pd.DataFrame,
) -> None:
    """
    Card-level velocity must use only earlier transactions for the same card.

    Transactions 1 and 2 share card1=1111 and timestamp=100, but neither may
    count the other. Transaction 3 is later and may count both.
    """
    result = pipeline.compute_velocity_features(
        df=sample_transactions,
        group_cols=["card1"],
    )

    velocity_by_id = result.set_index("TransactionID")["velocity_card1_5min"]

    assert velocity_by_id.loc[1] == 0
    assert velocity_by_id.loc[2] == 0
    assert velocity_by_id.loc[3] == 2

    # TransactionID 4 has a different card1 value, so its history is zero.
    assert velocity_by_id.loc[4] == 0


def test_amount_features_exclude_current_and_same_second_transactions(
    pipeline: PointInTimeFeaturePipeline,
    sample_transactions: pd.DataFrame,
) -> None:
    """
    Historical amount statistics must exclude the current row and same-second rows.

    TransactionID 1 and 2 have no strictly earlier transactions, so their
    historical amount mean is missing.

    TransactionID 3 can use prior amounts of 10 and 20:
    - mean = 15
    - population standard deviation = 5
    - z-score for amount 30 = (30 - 15) / 5 = 3
    """
    result = pipeline.compute_amount_features(
        df=sample_transactions,
        group_cols=None,
    )

    amount_mean = result.set_index("TransactionID")["amount_global_1h_mean"]
    amount_std = result.set_index("TransactionID")["amount_global_1h_std"]
    amount_zscore = result.set_index("TransactionID")["amount_global_1h_zscore"]

    assert pd.isna(amount_mean.loc[1])
    assert pd.isna(amount_mean.loc[2])

    assert amount_mean.loc[3] == pytest.approx(15.0)
    assert amount_std.loc[3] == pytest.approx(5.0)
    assert amount_zscore.loc[3] == pytest.approx(3.0)


def test_recency_uses_previous_distinct_timestamp(
    pipeline: PointInTimeFeaturePipeline,
    sample_transactions: pd.DataFrame,
) -> None:
    """
    Same-second transactions must have missing recency, not zero seconds.

    TransactionID 1 and 2 are in the first timestamp batch, so neither has
    a prior distinct timestamp. TransactionID 3 occurs 10 seconds later.
    """
    result = pipeline.compute_recency_feature(
        df=sample_transactions,
        group_cols=None,
    )

    recency_by_id = result.set_index("TransactionID")[
        "recency_global_seconds"
    ]

    assert pd.isna(recency_by_id.loc[1])
    assert pd.isna(recency_by_id.loc[2])
    assert recency_by_id.loc[3] == pytest.approx(10.0)
    assert recency_by_id.loc[4] == pytest.approx(20.0)


def test_entity_history_and_new_entity_flags_are_point_in_time_safe(
    pipeline: PointInTimeFeaturePipeline,
    sample_transactions: pd.DataFrame,
) -> None:
    """
    Entity count and new-entity features must use only prior timestamps.

    TransactionID 1 and 2 are both first-time same-second observations for
    card1=1111. Both receive a count of zero and a new-card flag of one.

    TransactionID 3 occurs later for the same card, so it sees two earlier
    card transactions and is no longer new.
    """
    result = pipeline.compute_historical_entity_features(
        df=sample_transactions,
        entity_columns=["card1"],
    )

    result_by_id = result.set_index("TransactionID")

    assert (
        result_by_id.loc[1, "history_card1_transaction_count"]
        == pytest.approx(0.0)
    )
    assert result_by_id.loc[1, "is_new_card1"] == pytest.approx(1.0)

    assert (
        result_by_id.loc[2, "history_card1_transaction_count"]
        == pytest.approx(0.0)
    )
    assert result_by_id.loc[2, "is_new_card1"] == pytest.approx(1.0)

    assert (
        result_by_id.loc[3, "history_card1_transaction_count"]
        == pytest.approx(2.0)
    )
    assert result_by_id.loc[3, "is_new_card1"] == pytest.approx(0.0)


def test_missing_entity_values_are_not_grouped_together(
    pipeline: PointInTimeFeaturePipeline,
    sample_transactions: pd.DataFrame,
) -> None:
    """
    Missing entity values must remain missing in entity-history features.

    TransactionID 5 and 6 both have DeviceInfo missing. They must not be
    considered transactions from a shared device.
    """
    result = pipeline.compute_historical_entity_features(
        df=sample_transactions,
        entity_columns=["DeviceInfo"],
    )

    result_by_id = result.set_index("TransactionID")

    assert pd.isna(
        result_by_id.loc[5, "history_DeviceInfo_transaction_count"]
    )
    assert pd.isna(result_by_id.loc[5, "is_new_DeviceInfo"])

    assert pd.isna(
        result_by_id.loc[6, "history_DeviceInfo_transaction_count"]
    )
    assert pd.isna(result_by_id.loc[6, "is_new_DeviceInfo"])


def test_behavioural_features_do_not_depend_on_fraud_label(
    pipeline: PointInTimeFeaturePipeline,
    sample_transactions: pd.DataFrame,
) -> None:
    """
    Changing isFraud values must not change velocity or history features.

    This proves that behavioural features are built without target encoding
    and without direct target leakage.
    """
    changed_labels = sample_transactions.copy()

    # Reverse the labels while keeping every transaction attribute unchanged.
    changed_labels["isFraud"] = 1 - changed_labels["isFraud"]

    original_result = pipeline.compute_velocity_features(
        df=sample_transactions,
        group_cols=["card1"],
    )

    changed_result = pipeline.compute_velocity_features(
        df=changed_labels,
        group_cols=["card1"],
    )

    assert original_result["velocity_card1_5min"].equals(
        changed_result["velocity_card1_5min"]
    )