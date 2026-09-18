import pytest
import tensorflow as tf

from fraud_intelligence.models.sequence_model import build_sequence_classifier


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_sequence_model_outputs_one_probability_per_transaction(
    architecture: str,
) -> None:
    model = build_sequence_classifier(
        architecture=architecture,
        sequence_length=10,
        feature_count=2,
        hidden_units=8,
        dropout_rate=0.0,
        recurrent_dropout_rate=0.0,
        dense_units=4,
        learning_rate=0.001,
    )

    sample_sequences = tf.zeros((3, 10, 2), dtype=tf.float32)
    probabilities = model(sample_sequences, training=False)

    assert probabilities.shape == (3, 1)
    assert model.output_shape == (None, 1)


def test_invalid_sequence_architecture_raises_meaningful_error() -> None:
    with pytest.raises(ValueError, match="gru.*lstm"):
        build_sequence_classifier(
            architecture="rnn",
            sequence_length=10,
            feature_count=2,
            hidden_units=8,
            dropout_rate=0.0,
            recurrent_dropout_rate=0.0,
            dense_units=4,
            learning_rate=0.001,
        )