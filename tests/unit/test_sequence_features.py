import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.features.sequence_features import (
    PointInTimeSequenceBuilder,
    SequenceFeatureConfig,
)


def make_builder() -> PointInTimeSequenceBuilder:
    return PointInTimeSequenceBuilder(
        config=SequenceFeatureConfig(max_sequence_length=3)
    )


def test_first_entity_event_has_only_current_transaction() -> None:
    dataframe = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [10],
            "card1": [1001],
            "TransactionAmt": [99.0],
            "isFraud": [0],
        }
    )

    result = make_builder().transform(dataframe)

    assert result.sequences.shape == (1, 3, 2)
    assert result.entity_history_lengths.tolist() == [0]
    assert np.all(result.sequences[0, 0:2] == 0.0)
    assert result.sequences[0, 2, 0] == pytest.approx(np.log1p(99.0), abs=1e-6)
    assert result.sequences[0, 2, 1] == 0.0


def test_later_entity_event_uses_only_prior_entity_history() -> None:
    dataframe = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 30],
            "card1": [1001, 1001, 1001],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "isFraud": [0, 1, 0],
        }
    )

    result = make_builder().transform(dataframe)

    assert result.entity_history_lengths.tolist() == [0, 1, 2]

    np.testing.assert_allclose(
        result.sequences[2, :, 0],
        [np.log1p(10.0), np.log1p(20.0), np.log1p(30.0)],
    )

    np.testing.assert_allclose(
        result.sequences[2, :, 1],
        [0.0, np.log1p(10.0), np.log1p(10.0)],
    )


def test_same_timestamp_rows_cannot_enter_each_others_history() -> None:
    dataframe = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 20],
            "card1": [1001, 1001, 1001],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "isFraud": [0, 0, 1],
        }
    )

    result = make_builder().transform(dataframe)

    assert result.entity_history_lengths.tolist() == [0, 1, 1]

    expected_prior_amount = np.log1p(10.0)

    assert result.sequences[1, 1, 0] == pytest.approx(expected_prior_amount, abs=1e-6)
    assert result.sequences[2, 1, 0] == pytest.approx(expected_prior_amount, abs=1e-6)

    assert result.sequences[1, 2, 0] == pytest.approx(np.log1p(20.0), abs=1e-6)
    assert result.sequences[2, 2, 0] == pytest.approx(np.log1p(30.0), abs=1e-6)


def test_validation_can_use_training_history_but_not_future_validation_rows() -> None:
    train_dataframe = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [10],
            "card1": [1001],
            "TransactionAmt": [10.0],
            "isFraud": [0],
        }
    )

    validation_dataframe = pd.DataFrame(
        {
            "TransactionID": [2, 3],
            "TransactionDT": [20, 30],
            "card1": [1001, 1001],
            "TransactionAmt": [20.0, 30.0],
            "isFraud": [1, 0],
        }
    )

    builder = make_builder()

    builder.transform(train_dataframe, reset_state=True)
    validation_result = builder.transform(validation_dataframe, reset_state=False)

    assert validation_result.entity_history_lengths.tolist() == [1, 2]

    np.testing.assert_allclose(
        validation_result.sequences[0, :, 0],
        [0.0, np.log1p(10.0), np.log1p(20.0)],
    )

    np.testing.assert_allclose(
        validation_result.sequences[1, :, 0],
        [np.log1p(10.0), np.log1p(20.0), np.log1p(30.0)],
    )


def test_original_dataframe_order_is_preserved() -> None:
    dataframe = pd.DataFrame(
        {
            "TransactionID": [3, 1, 2],
            "TransactionDT": [30, 10, 20],
            "card1": [1001, 1001, 1001],
            "TransactionAmt": [30.0, 10.0, 20.0],
            "isFraud": [0, 0, 1],
        }
    )

    result = make_builder().transform(dataframe)

    assert result.transaction_ids.tolist() == [3, 1, 2]
    assert result.timestamps.tolist() == [30.0, 10.0, 20.0]
    assert result.entity_history_lengths.tolist() == [2, 0, 1]


def test_missing_card_proxy_has_no_shared_history() -> None:
    dataframe = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [10, 20],
            "card1": [np.nan, np.nan],
            "TransactionAmt": [10.0, 20.0],
            "isFraud": [0, 1],
        }
    )

    result = make_builder().transform(dataframe)

    assert result.entity_history_lengths.tolist() == [0, 0]
    assert result.sequences[0, 2, 1] == 0.0
    assert result.sequences[1, 2, 1] == 0.0

