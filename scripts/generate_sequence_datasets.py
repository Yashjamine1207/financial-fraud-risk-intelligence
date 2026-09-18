"""Generate leakage-safe train and validation transaction sequence datasets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fraud_intelligence.features.sequence_features import (
    PointInTimeSequenceBuilder,
    SequenceDataset,
    SequenceFeatureConfig,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and validate one YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Sequence configuration was not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Sequence configuration must parse to a dictionary: {path}")

    required_sections = {"data", "sequence", "outputs", "reproducibility"}

    missing_sections = required_sections.difference(config)

    if missing_sections:
        raise ValueError(
            "Sequence configuration is missing required section(s): "
            f"{sorted(missing_sections)}"
        )

    return config


def read_parquet(path: Path, split_name: str) -> pd.DataFrame:
    """Read one source feature table."""
    if not path.exists():
        raise FileNotFoundError(
            f"{split_name.title()} source table was not found: {path}\n"
            "Check configs/sequence_model.yaml."
        )

    dataframe = pd.read_parquet(path)

    if dataframe.empty:
        raise ValueError(f"{split_name.title()} source table is empty: {path}")

    LOGGER.info(
        "Loaded %s table: %s rows, %s columns.",
        split_name,
        f"{len(dataframe):,}",
        f"{len(dataframe.columns):,}",
    )

    return dataframe


def build_sequence_config(config: dict[str, Any]) -> SequenceFeatureConfig:
    """Map YAML settings to the feature-builder configuration."""
    data_config = config["data"]
    sequence_config = config["sequence"]

    return SequenceFeatureConfig(
        transaction_id_column=data_config["transaction_id_column"],
        timestamp_column=data_config["timestamp_column"],
        target_column=data_config["target_column"],
        entity_column=data_config["entity_column"],
        amount_column=data_config["amount_column"],
        max_sequence_length=int(sequence_config["max_sequence_length"]),
        padding_value=float(sequence_config["padding_value"]),
    )


def save_sequence_dataset(
    dataset: SequenceDataset,
    output_path: Path,
    split_name: str,
) -> None:
    """Save compressed model tensors using an atomic write."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".tmp.npz")

    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            sequences=dataset.sequences,
            labels=dataset.labels,
            transaction_ids=dataset.transaction_ids,
            timestamps=dataset.timestamps,
            entity_history_lengths=dataset.entity_history_lengths,
        )

    temporary_path.replace(output_path)

    LOGGER.info(
        "Wrote %s sequence dataset: shape=%s -> %s",
        split_name,
        dataset.sequences.shape,
        output_path,
    )


def build_split_summary(
    dataset: SequenceDataset,
    split_name: str,
    sequence_config: SequenceFeatureConfig,
) -> dict[str, int | float | str]:
    """Create traceable summary statistics for one sequence split."""
    history_lengths = dataset.entity_history_lengths

    return {
        "split": split_name,
        "row_count": len(dataset.labels),
        "fraud_count": int(dataset.labels.sum()),
        "fraud_rate": float(dataset.labels.mean()),
        "sequence_length": int(dataset.sequences.shape[1]),
        "feature_count_per_event": int(dataset.sequences.shape[2]),
        "mean_prior_entity_history_length": float(history_lengths.mean()),
        "median_prior_entity_history_length": float(np.median(history_lengths)),
        "rows_with_prior_entity_history": int((history_lengths > 0).sum()),
        "rows_with_prior_entity_history_rate": float((history_lengths > 0).mean()),
        "rows_at_max_history_length": int(
            (history_lengths >= sequence_config.max_sequence_length - 1).sum()
        ),
        "rows_at_max_history_length_rate": float(
            (history_lengths >= sequence_config.max_sequence_length - 1).mean()
        ),
        "minimum_timestamp": float(dataset.timestamps.min()),
        "maximum_timestamp": float(dataset.timestamps.max()),
    }


def save_summary_csv(
    summaries: list[dict[str, int | float | str]],
    output_path: Path,
) -> None:
    """Save compact train/validation sequence summary metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(summaries[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    LOGGER.info("Saved sequence summary CSV: %s", output_path)


def save_metadata_json(
    config: dict[str, Any],
    train_summary: dict[str, int | float | str],
    validation_summary: dict[str, int | float | str],
    output_path: Path,
) -> None:
    """Save dataset versioning and leakage-control metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "phase": 6,
        "purpose": "compact_time_safe_sequence_dataset_generation",
        "sequence_feature_version": config["project"]["sequence_feature_version"],
        "dataset_version": config["reproducibility"]["dataset_version"],
        "point_in_time_feature_version": config["reproducibility"][
            "point_in_time_feature_version"
        ],
        "entity_column": config["data"]["entity_column"],
        "timestamp_column": config["data"]["timestamp_column"],
        "target_column": config["data"]["target_column"],
        "sequence_length": config["sequence"]["max_sequence_length"],
        "input_features": config["sequence"]["input_features"],
        "padding": config["sequence"]["padding"],
        "padding_value": config["sequence"]["padding_value"],
        "leakage_controls": {
            "training_history": "strictly_prior_training_timestamps_only",
            "validation_history": (
                "training_history_plus_strictly_prior_validation_timestamps_only"
            ),
            "same_timestamp_policy": config["sequence"]["same_timestamp_policy"],
            "fraud_labels_used_as_sequence_inputs": False,
            "final_test_locked": True,
            "final_test_loaded": False,
        },
        "train_summary": train_summary,
        "validation_summary": validation_summary,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    LOGGER.info("Saved sequence metadata JSON: %s", output_path)


def generate_sequence_datasets(config: dict[str, Any]) -> None:
    """Generate train then validation sequences without reading final test data."""
    data_config = config["data"]
    output_config = config["outputs"]

    if data_config.get("final_test_locked") is not True:
        raise RuntimeError(
            "Final-test protection is disabled. "
            "Set data.final_test_locked to true before sequence generation."
        )

    train_input_path = Path(data_config["train_feature_path"])
    validation_input_path = Path(data_config["validation_feature_path"])

    train_output_path = Path(output_config["train_sequence_path"])
    validation_output_path = Path(output_config["validation_sequence_path"])

    sequence_config = build_sequence_config(config)
    builder = PointInTimeSequenceBuilder(config=sequence_config)

    train_dataframe = read_parquet(train_input_path, "training")
    validation_dataframe = read_parquet(validation_input_path, "validation")

    LOGGER.info(
        "Generating training sequences using current events plus strictly prior "
        "same-card1 transaction history."
    )
    train_sequences = builder.transform(
        dataframe=train_dataframe,
        reset_state=True,
    )

    LOGGER.info(
        "Generating validation sequences using training history plus strictly "
        "prior validation card1 history."
    )
    validation_sequences = builder.transform(
        dataframe=validation_dataframe,
        reset_state=False,
    )

    train_summary = build_split_summary(
        dataset=train_sequences,
        split_name="training",
        sequence_config=sequence_config,
    )
    validation_summary = build_split_summary(
        dataset=validation_sequences,
        split_name="validation",
        sequence_config=sequence_config,
    )

    save_sequence_dataset(
        dataset=train_sequences,
        output_path=train_output_path,
        split_name="training",
    )
    save_sequence_dataset(
        dataset=validation_sequences,
        output_path=validation_output_path,
        split_name="validation",
    )

    save_summary_csv(
        summaries=[train_summary, validation_summary],
        output_path=Path(output_config["sequence_summary_path"]),
    )
    save_metadata_json(
        config=config,
        train_summary=train_summary,
        validation_summary=validation_summary,
        output_path=Path(output_config["sequence_metadata_path"]),
    )

    LOGGER.info(
        "Final-test sequence data was intentionally not generated or loaded."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-safe train and validation transaction sequences. "
            "The final test split is never read."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sequence_model.yaml"),
        help="Path to the sequence-model YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Run sequence dataset generation."""
    configure_logging()
    arguments = parse_arguments()
    config = load_yaml_config(arguments.config)
    generate_sequence_datasets(config)


if __name__ == "__main__":
    main()