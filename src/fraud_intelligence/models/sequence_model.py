"""Compact TensorFlow/Keras sequence models for the Phase 6 ablation."""

from __future__ import annotations

from typing import Literal

import tensorflow as tf

SequenceArchitecture = Literal["gru", "lstm"]


def build_sequence_classifier(
    architecture: SequenceArchitecture,
    sequence_length: int,
    feature_count: int,
    hidden_units: int,
    dropout_rate: float,
    recurrent_dropout_rate: float,
    dense_units: int,
    learning_rate: float,
    padding_value: float = 0.0,
) -> tf.keras.Model:
    """Build a compact binary sequence classifier with a GRU or LSTM layer."""
    if architecture not in {"gru", "lstm"}:
        raise ValueError("architecture must be either 'gru' or 'lstm'.")

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")

    if feature_count < 1:
        raise ValueError("feature_count must be positive.")

    if hidden_units < 1 or dense_units < 1:
        raise ValueError("hidden_units and dense_units must be positive.")

    inputs = tf.keras.Input(
        shape=(sequence_length, feature_count),
        name="transaction_sequence",
    )

    masked_inputs = tf.keras.layers.Masking(
        mask_value=padding_value,
        name="padding_mask",
    )(inputs)

    recurrent_layer: tf.keras.layers.Layer

    if architecture == "gru":
        recurrent_layer = tf.keras.layers.GRU(
            units=hidden_units,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name="gru_encoder",
        )
    else:
        recurrent_layer = tf.keras.layers.LSTM(
            units=hidden_units,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name="lstm_encoder",
        )

    encoded_sequence = recurrent_layer(masked_inputs)

    dense_output = tf.keras.layers.Dense(
        units=dense_units,
        activation="relu",
        name="dense_representation",
    )(encoded_sequence)

    dropped_output = tf.keras.layers.Dropout(
        rate=dropout_rate,
        name="dense_dropout",
    )(dense_output)

    fraud_probability = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="fraud_probability",
    )(dropped_output)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=fraud_probability,
        name=f"compact_{architecture}_sequence_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            ),
            tf.keras.metrics.AUC(
                curve="ROC",
                name="roc_auc",
            ),
        ],
    )

    return model