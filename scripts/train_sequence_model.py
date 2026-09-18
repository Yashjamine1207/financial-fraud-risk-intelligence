"""Train a compact GRU or LSTM sequence-model ablation on chronological data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import tensorflow as tf
import yaml
from dotenv import load_dotenv

from fraud_intelligence.models.evaluation import evaluate_binary_classifier
from fraud_intelligence.models.sequence_model import build_sequence_classifier

CONFIG_PATH = Path("configs/sequence_model.yaml")
EXPERIMENT_NAME = "fraud-intelligence-phase-6-sequence-ablation"


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and validate the sequence-model configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Configuration must parse to a dictionary: {path}")

    return config


def load_sequence_dataset(path: Path, split_name: str) -> dict[str, np.ndarray]:
    """Load and validate one saved sequence dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"{split_name.title()} sequence dataset was not found: {path}. "
            "Run scripts/generate_sequence_datasets.py first."
        )

    with np.load(path) as dataset:
        required_arrays = {
            "sequences",
            "labels",
            "transaction_ids",
            "timestamps",
            "entity_history_lengths",
        }
        missing_arrays = required_arrays.difference(dataset.files)

        if missing_arrays:
            raise ValueError(
                f"{split_name.title()} sequence dataset is missing arrays: "
                f"{sorted(missing_arrays)}"
            )

        loaded = {
            array_name: dataset[array_name].copy()
            for array_name in required_arrays
        }

    sequences = loaded["sequences"]
    labels = loaded["labels"]
    timestamps = loaded["timestamps"]

    if sequences.ndim != 3:
        raise ValueError(
            f"{split_name.title()} sequences must have shape "
            "(rows, sequence_length, feature_count)."
        )

    if len(sequences) != len(labels) or len(sequences) != len(timestamps):
        raise ValueError(
            f"{split_name.title()} sequence arrays do not have matching row counts."
        )

    if not np.isfinite(sequences).all():
        raise ValueError(f"{split_name.title()} sequences contain non-finite values.")

    if not np.isfinite(timestamps).all():
        raise ValueError(f"{split_name.title()} timestamps contain non-finite values.")

    if np.any(np.diff(timestamps) < 0):
        raise ValueError(
            f"{split_name.title()} sequence timestamps are not chronologically sorted."
        )

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{split_name.title()} labels must be binary.")

    return loaded


def calculate_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Calculate inverse-frequency class weights from fitting labels only."""
    fraud_count = int(labels.sum())
    non_fraud_count = int(len(labels) - fraud_count)

    if fraud_count == 0 or non_fraud_count == 0:
        raise ValueError("Training fit period must contain both classes.")

    return {
        0: 1.0,
        1: non_fraud_count / fraud_count,
    }


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a Git-trackable metrics report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Saved metrics report to: {path}")


def main() -> None:
    """Train one compact GRU or LSTM model on the saved sequence arrays."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=["gru", "lstm"],
        required=True,
        help="Recurrent architecture to train.",
    )
    arguments = parser.parse_args()

    load_dotenv()
    config = load_yaml_config(CONFIG_PATH)

    if config["data"].get("final_test_locked") is not True:
        raise RuntimeError(
            "Final-test protection is disabled. "
            "Set data.final_test_locked to true before sequence training."
        )

    architecture = arguments.architecture
    model_config = config["model"]
    sequence_config = config["sequence"]
    output_config = config["outputs"]
    evaluation_config = config["evaluation"]
    reproducibility_config = config["reproducibility"]

    train_data = load_sequence_dataset(
        Path(output_config["train_sequence_path"]),
        "training",
    )
    validation_data = load_sequence_dataset(
        Path(output_config["validation_sequence_path"]),
        "validation",
    )

    X_train_full = train_data["sequences"].astype(np.float32)
    y_train_full = train_data["labels"].astype(np.float32)

    X_validation = validation_data["sequences"].astype(np.float32)
    y_validation = validation_data["labels"].astype(np.int8)

    if validation_data["timestamps"].min() <= train_data["timestamps"].max():
        raise ValueError(
            "Temporal split violation: validation sequence data must begin "
            "strictly after the training sequence data."
        )

    internal_validation_fraction = float(
        model_config["validation_split_within_train"]
    )
    split_index = int(len(X_train_full) * (1.0 - internal_validation_fraction))

    if split_index <= 0 or split_index >= len(X_train_full):
        raise ValueError("Unable to create chronological internal validation split.")

    X_train_fit = X_train_full[:split_index]
    y_train_fit = y_train_full[:split_index]

    X_train_early_stop = X_train_full[split_index:]
    y_train_early_stop = y_train_full[split_index:]

    if len(np.unique(y_train_fit)) != 2 or len(np.unique(y_train_early_stop)) != 2:
        raise ValueError(
            "Both chronological training-fit and early-stopping periods "
            "must contain fraud and non-fraud labels."
        )

    class_weights = calculate_class_weights(y_train_fit)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(model_config["random_seed"]))

    model = build_sequence_classifier(
        architecture=architecture,
        sequence_length=X_train_full.shape[1],
        feature_count=X_train_full.shape[2],
        hidden_units=int(model_config["hidden_units"]),
        dropout_rate=float(model_config["dropout_rate"]),
        recurrent_dropout_rate=float(model_config["recurrent_dropout_rate"]),
        dense_units=int(model_config["dense_units"]),
        learning_rate=float(model_config["learning_rate"]),
        padding_value=float(sequence_config["padding_value"]),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=int(model_config["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        )
    ]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_path = Path(
        f"models/artifacts/phase6_{architecture}_sequence_model.keras"
    )
    metrics_path = Path(
        f"reports/tables/phase6_{architecture}_sequence_validation_metrics.json"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = f"phase6-compact-{architecture}-sequence-v1.0.0"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project_phase": "6",
                "experiment_type": "controlled_sequence_model_ablation",
                "architecture": architecture,
                "model_version": (
                    f"{reproducibility_config['model_version_prefix']}"
                    f"-{architecture}-v1.0.0"
                ),
                "entity_sequence_key": config["data"]["entity_column"],
                "sequence_length": str(sequence_config["max_sequence_length"]),
                "same_timestamp_policy": sequence_config["same_timestamp_policy"],
                "sequence_training_protocol": "strictly_chronological",
                "shuffle_training_rows": "false",
                "fraud_labels_used_as_sequence_inputs": "false",
                "test_split_loaded": "false",
            }
        )

        mlflow.log_params(
            {
                "architecture": architecture,
                "train_rows": len(X_train_full),
                "validation_rows": len(X_validation),
                "training_fit_rows": len(X_train_fit),
                "early_stopping_rows": len(X_train_early_stop),
                "sequence_length": X_train_full.shape[1],
                "feature_count_per_event": X_train_full.shape[2],
                "hidden_units": model_config["hidden_units"],
                "dense_units": model_config["dense_units"],
                "dropout_rate": model_config["dropout_rate"],
                "learning_rate": model_config["learning_rate"],
                "batch_size": model_config["batch_size"],
                "max_epochs": model_config["max_epochs"],
                "class_weight_fraud": class_weights[1],
                "review_capacity": evaluation_config["review_capacity"],
            }
        )

        print(f"\nTraining compact {architecture.upper()} on chronological sequences...")
        print(f"Training-fit rows: {len(X_train_fit):,}")
        print(f"Early-stopping rows: {len(X_train_early_stop):,}")
        print(f"Validation rows: {len(X_validation):,}")
        print(f"Fraud class weight: {class_weights[1]:.2f}")
        print("Final test split loaded: False")

        training_start = time.perf_counter()

        history = model.fit(
            X_train_fit,
            y_train_fit,
            validation_data=(X_train_early_stop, y_train_early_stop),
            epochs=int(model_config["max_epochs"]),
            batch_size=int(model_config["batch_size"]),
            class_weight=class_weights,
            callbacks=callbacks,
            shuffle=False,
            verbose=2,
        )

        training_time_seconds = time.perf_counter() - training_start

        prediction_start = time.perf_counter()

        validation_probabilities = model.predict(
            X_validation,
            batch_size=int(model_config["batch_size"]),
            verbose=0,
        ).reshape(-1)

        prediction_time_seconds = time.perf_counter() - prediction_start
        prediction_latency_ms_per_row = (
            prediction_time_seconds / len(X_validation)
        ) * 1_000

        metrics = evaluate_binary_classifier(
            y_true=y_validation,
            fraud_probabilities=validation_probabilities,
            review_capacity=int(evaluation_config["review_capacity"]),
            reference_threshold=0.5,
            prediction_latency_ms_per_row=prediction_latency_ms_per_row,
        )

        metrics.update(
            {
                "phase": 6,
                "model_name": f"compact_{architecture}_sequence_classifier",
                "model_version": (
                    f"{reproducibility_config['model_version_prefix']}"
                    f"-{architecture}-v1.0.0"
                ),
                "architecture": architecture,
                "run_id": run.info.run_id,
                "training_time_seconds": float(training_time_seconds),
                "epochs_completed": len(history.history["loss"]),
                "best_internal_validation_pr_auc": float(
                    max(history.history["val_pr_auc"])
                ),
                "sequence_length": int(X_train_full.shape[1]),
                "feature_count_per_event": int(X_train_full.shape[2]),
                "entity_column": config["data"]["entity_column"],
                "dataset_version": reproducibility_config["dataset_version"],
                "feature_version": reproducibility_config[
                    "point_in_time_feature_version"
                ],
                "target_encoding_version": reproducibility_config[
                    "target_encoding_version"
                ],
                "test_split_loaded": False,
            }
        )

        numeric_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }

        mlflow.log_metrics(numeric_metrics)

        model.save(model_path)
        save_json(metrics, metrics_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(metrics_path)

        print(f"\n{architecture.upper()} validation results")
        print("-" * 56)
        print(f"PR-AUC: {metrics['pr_auc']:.6f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
        print(f"Precision@1,000: {metrics['precision_at_k']:.6f}")
        print(f"Recall@1,000: {metrics['recall_at_k']:.6f}")
        print(
            "Captured fraud at 1,000 reviews: "
            f"{metrics['captured_fraud_count_at_k']:,} / "
            f"{metrics['total_fraud_count']:,}"
        )
        print(f"Epochs completed: {metrics['epochs_completed']}")
        print(f"Training time: {metrics['training_time_seconds']:.2f} seconds")
        print("Final test split loaded: False")
        print(f"\nMLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()