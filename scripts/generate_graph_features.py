"""Generate leakage-safe graph features for training and validation only."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fraud_intelligence.features.graph_features import (
    GRAPH_FEATURE_COLUMNS,
    GraphFeatureConfig,
    PointInTimeGraphFeatureBuilder,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Graph configuration file was not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise TypeError(f"Graph configuration is empty or invalid: {config_path}")

    return config


def read_parquet(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{split_name.title()} feature input was not found: {path}\n"
            "Confirm the configured path in configs/graph_features.yaml."
        )

    dataframe = pd.read_parquet(path)

    if dataframe.empty:
        raise ValueError(f"{split_name.title()} feature input is empty: {path}")

    LOGGER.info(
        "Loaded %s input: %s rows, %s columns.",
        split_name,
        f"{len(dataframe):,}",
        f"{len(dataframe.columns):,}",
    )
    return dataframe


def write_parquet(dataframe: pd.DataFrame, output_path: Path, split_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".tmp.parquet")
    dataframe.to_parquet(temporary_path, index=False)
    temporary_path.replace(output_path)

    LOGGER.info(
        "Wrote %s graph-feature dataset: %s rows, %s columns -> %s",
        split_name,
        f"{len(dataframe):,}",
        f"{len(dataframe.columns):,}",
        output_path,
    )


def log_feature_summary(dataframe: pd.DataFrame, split_name: str) -> None:
    available_features = [
        column for column in GRAPH_FEATURE_COLUMNS if column in dataframe.columns
    ]

    if len(available_features) != len(GRAPH_FEATURE_COLUMNS):
        missing_features = sorted(set(GRAPH_FEATURE_COLUMNS) - set(available_features))
        raise RuntimeError(
            f"{split_name.title()} output is missing expected graph features: "
            f"{missing_features}"
        )

    LOGGER.info("%s graph-feature summary:", split_name.title())

    for column in available_features:
        null_rate = dataframe[column].isna().mean()
        LOGGER.info(
            "  %-48s null_rate=%6.2f%% mean=%12.4f",
            column,
            null_rate * 100,
            dataframe[column].mean(),
        )


def build_feature_config(config: dict[str, Any]) -> GraphFeatureConfig:
    data_config = config["data"]
    graph_config = config["graph"]

    relationship_entities = graph_config["relationship_entities"]

    if relationship_entities != ["DeviceInfo", "R_emaildomain"]:
        raise ValueError(
            "This graph-feature implementation expects relationship_entities "
            "to be exactly ['DeviceInfo', 'R_emaildomain']."
        )

    return GraphFeatureConfig(
        timestamp_column=data_config["timestamp_column"],
        card_column=graph_config["primary_entity"],
        device_column=relationship_entities[0],
        recipient_email_column=relationship_entities[1],
    )


def generate_train_and_validation_features(config: dict[str, Any]) -> None:
    data_config = config["data"]

    if not data_config.get("final_test_locked", True):
        raise ValueError(
            "final_test_locked must remain true during graph feature generation."
        )

    train_input_path = Path(data_config["train_feature_path"])
    validation_input_path = Path(data_config["validation_feature_path"])
    train_output_path = Path(data_config["train_graph_output_path"])
    validation_output_path = Path(data_config["validation_graph_output_path"])

    graph_feature_config = build_feature_config(config)
    builder = PointInTimeGraphFeatureBuilder(config=graph_feature_config)

    train_dataframe = read_parquet(train_input_path, "training")
    validation_dataframe = read_parquet(validation_input_path, "validation")

    LOGGER.info(
        "Generating training graph features from earlier training transactions only."
    )
    train_with_graph = builder.transform(train_dataframe, reset_state=True)
    log_feature_summary(train_with_graph, "training")
    write_parquet(train_with_graph, train_output_path, "training")

    LOGGER.info(
        "Generating validation graph features using training history plus "
        "earlier validation transactions only."
    )
    validation_with_graph = builder.transform(validation_dataframe, reset_state=False)
    log_feature_summary(validation_with_graph, "validation")
    write_parquet(validation_with_graph, validation_output_path, "validation")

    LOGGER.info(
        "Final-test graph features were intentionally not generated. "
        "The locked final test split must not be used until the graph ablation "
        "has been selected or rejected using validation results."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate point-in-time graph features for train and validation data. "
            "The locked final test split is never read or written."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/graph_features.yaml"),
        help="Path to the graph-feature YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    arguments = parse_arguments()
    config = load_config(arguments.config)
    generate_train_and_validation_features(config)


if __name__ == "__main__":
    main()
