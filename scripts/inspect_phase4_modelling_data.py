"""Inspect and record the Phase 4 modelling data contract.

This script loads only the Phase 3 training and validation feature tables.
It validates the shared feature contract and writes a small, Git-trackable
JSON summary. The final test split is never loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fraud_intelligence.models.data_contract import load_modelling_data

OUTPUT_PATH = Path("reports/tables/phase4_modelling_data_summary.json")


def build_summary() -> dict[str, Any]:
    """Load the approved development data and return an auditable summary."""
    modelling_data = load_modelling_data()

    target_column = modelling_data.config["data"]["target_column"]
    timestamp_column = modelling_data.config["data"]["timestamp_column"]

    return {
        "phase": 4,
        "purpose": (
            "Validate the shared train/validation modelling data contract before "
            "training Logistic Regression and XGBoost."
        ),
        "test_split_loaded": False,
        "test_split_locked": modelling_data.config["data"]["test_locked"],
        "dataset_version": modelling_data.config["reproducibility"]["dataset_version"],
        "feature_version": modelling_data.config["reproducibility"]["feature_version"],
        "target_encoding_version": modelling_data.config["reproducibility"][
            "target_encoding_version"
        ],
        "train": {
            "rows": len(modelling_data.X_train),
            "features": int(modelling_data.X_train.shape[1]),
            "fraud_rate": float(modelling_data.y_train.mean()),
        },
        "validation": {
            "rows": len(modelling_data.X_validation),
            "features": int(modelling_data.X_validation.shape[1]),
            "fraud_rate": float(modelling_data.y_validation.mean()),
        },
        "feature_types": {
            "numeric_columns": len(modelling_data.numeric_columns),
            "categorical_columns": len(modelling_data.categorical_columns),
        },
        "temporal_boundaries": {
            "timestamp_column": timestamp_column,
            "target_column": target_column,
        },
        "excluded_columns": {
            "mandatory": modelling_data.config["feature_contract"]["mandatory_excluded_columns"],
            "high_missing": modelling_data.config["feature_contract"]["high_missing_columns"],
            "manual": modelling_data.config["feature_contract"]["manually_excluded_columns"],
        },
        "selected_feature_columns": modelling_data.feature_columns,
    }


def save_summary(summary: dict[str, Any]) -> None:
    """Write the Phase 4 data-contract summary as a small JSON artifact."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved Phase 4 modelling data summary to: {OUTPUT_PATH}")


def print_summary(summary: dict[str, Any]) -> None:
    """Print the most important data-contract results to the terminal."""
    train_summary = summary["train"]
    validation_summary = summary["validation"]
    feature_types = summary["feature_types"]

    print("\nPhase 4 modelling data contract passed.")
    print(f"Train rows: {train_summary['rows']:,}")
    print(f"Validation rows: {validation_summary['rows']:,}")
    print(f"Selected model features: {train_summary['features']:,}")
    print(f"Numeric features: {feature_types['numeric_columns']:,}")
    print(f"Categorical features: {feature_types['categorical_columns']:,}")
    print(f"Train fraud rate: {train_summary['fraud_rate']:.4%}")
    print(f"Validation fraud rate: {validation_summary['fraud_rate']:.4%}")
    print(f"Final test split loaded: {summary['test_split_loaded']}")


def main() -> None:
    """Run the Phase 4 data-contract inspection."""
    summary = build_summary()
    save_summary(summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
