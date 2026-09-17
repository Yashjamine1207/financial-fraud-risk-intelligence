"""
Tests for leakage-safe chronological target encoding.

These tests prove that:
- Same-timestamp transactions cannot use one another's fraud labels.
- A training row cannot use its own fraud label in its encoded value.
- Validation data uses mappings learned from training labels only.
- Unseen validation categories fall back to the training fraud rate.
"""

import pandas as pd
import pytest

from fraud_intelligence.features import TimeSafeTargetEncoder


@pytest.fixture
def training_data() -> pd.DataFrame:
    """
    Create a small chronological labelled training dataset.

    Rows 1 and 2 share timestamp 100. They must not use one another's labels.
    Row 3 occurs later at timestamp 200.
    Row 4 occurs later at timestamp 300.
    """
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100, 100, 200, 300],
            "ProductCD": ["A", "B", "A", "B"],
            "card4": ["visa", "mastercard", "visa", "mastercard"],
            "isFraud": [0, 1, 1, 0],
        }
    )


def test_first_timestamp_fold_has_no_target_history(
    training_data: pd.DataFrame,
) -> None:
    """
    The earliest timestamp has no earlier fraud labels available.

    Both transactions at timestamp 100 must receive NaN target encodings.
    They cannot use their own label or the other same-second row's label.
    """
    encoder = TimeSafeTargetEncoder(
        columns=["ProductCD"],
        smoothing=0.0,
        min_samples=1,
    )

    encoded = encoder.fit_transform_oof(
        df=training_data,
        target=training_data["isFraud"],
        timestamp_col="TransactionDT",
        transaction_id_col="TransactionID",
        n_splits=3,
    )

    assert pd.isna(encoded.loc[0, "ProductCD_target_encoded"])
    assert pd.isna(encoded.loc[1, "ProductCD_target_encoded"])


def test_oof_encoding_excludes_current_row_target(
    training_data: pd.DataFrame,
) -> None:
    """
    A training row must not use its own isFraud value.

    At timestamp 200, ProductCD A has one earlier observation:
    - timestamp 100, ProductCD A, isFraud = 0

    The current row at timestamp 200 has isFraud = 1, but its OOF encoding
    must still equal 0.0 because it can only use the earlier A observation.
    """
    encoder = TimeSafeTargetEncoder(
        columns=["ProductCD"],
        smoothing=0.0,
        min_samples=1,
    )

    encoded = encoder.fit_transform_oof(
        df=training_data,
        target=training_data["isFraud"],
        timestamp_col="TransactionDT",
        transaction_id_col="TransactionID",
        n_splits=3,
    )

    # Row index 2 is TransactionID 3 at timestamp 200.
    assert encoded.loc[2, "ProductCD_target_encoded"] == pytest.approx(0.0)


def test_changing_current_label_does_not_change_own_encoding(
    training_data: pd.DataFrame,
) -> None:
    """
    Changing a row's own fraud label must not change that row's OOF encoding.

    This directly verifies that target encoding does not leak the row's label.
    """
    original_encoder = TimeSafeTargetEncoder(
        columns=["ProductCD"],
        smoothing=0.0,
        min_samples=1,
    )

    original_encoded = original_encoder.fit_transform_oof(
        df=training_data,
        target=training_data["isFraud"],
        timestamp_col="TransactionDT",
        transaction_id_col="TransactionID",
        n_splits=3,
    )

    changed_labels = training_data["isFraud"].copy()

    # Change the label for TransactionID 3 at timestamp 200.
    changed_labels.loc[2] = 0

    changed_encoder = TimeSafeTargetEncoder(
        columns=["ProductCD"],
        smoothing=0.0,
        min_samples=1,
    )

    changed_encoded = changed_encoder.fit_transform_oof(
        df=training_data,
        target=changed_labels,
        timestamp_col="TransactionDT",
        transaction_id_col="TransactionID",
        n_splits=3,
    )

    # TransactionID 3's encoding must remain based on earlier timestamp 100.
    assert (
        original_encoded.loc[2, "ProductCD_target_encoded"]
        == changed_encoded.loc[2, "ProductCD_target_encoded"]
    )

    assert original_encoded.loc[2, "ProductCD_target_encoded"] == pytest.approx(
        0.0
    )


def test_validation_transform_uses_training_mapping_only(
    training_data: pd.DataFrame,
) -> None:
    """
    Validation categories must use a mapping fitted from training data only.

    ProductCD C is unseen in training, so it receives the global training
    fraud rate. ProductCD A receives the fraud rate learned from training A
    rows only.
    """
    encoder = TimeSafeTargetEncoder(
        columns=["ProductCD"],
        smoothing=0.0,
        min_samples=1,
    )

    encoder.fit(
        df=training_data,
        target=training_data["isFraud"],
    )

    validation_data = pd.DataFrame(
        {
            "ProductCD": ["A", "C"],
        }
    )

    encoded_validation = encoder.transform(validation_data)

    # Training ProductCD A fraud rate: (0 + 1) / 2 = 0.5
    assert encoded_validation.loc[
        0,
        "ProductCD_target_encoded",
    ] == pytest.approx(0.5)

    # Training global fraud rate: (0 + 1 + 1 + 0) / 4 = 0.5
    assert encoded_validation.loc[
        1,
        "ProductCD_target_encoded",
    ] == pytest.approx(0.5)