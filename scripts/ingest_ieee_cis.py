"""Ingest and join the IEEE-CIS transaction and identity datasets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
JOIN_KEY = "TransactionID"

REQUIRED_TRANSACTION_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
}

REQUIRED_IDENTITY_COLUMNS = {
    "TransactionID",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest and join IEEE-CIS transaction and identity data."
    )
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=Path("data/raw/ieee_cis"),
        help="Directory containing the immutable raw CSV files.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/interim/joined_transactions"),
        help="Directory for the joined Parquet file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/joined_transactions/ingestion_manifest.json"),
        help="Path for the ingestion manifest.",
    )
    return parser.parse_args()


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    """Validate that a dataframe contains the required columns."""
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{dataframe_name} is missing required columns: {missing_text}")


def load_raw_data(raw_directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the transaction and identity CSV files."""
    transaction_path = raw_directory / TRANSACTION_FILE
    identity_path = raw_directory / IDENTITY_FILE

    if not transaction_path.is_file():
        raise FileNotFoundError(f"Missing transaction file: {transaction_path}")

    if not identity_path.is_file():
        raise FileNotFoundError(f"Missing identity file: {identity_path}")

    transactions = pd.read_csv(transaction_path)
    identities = pd.read_csv(identity_path)

    validate_columns(
        transactions,
        REQUIRED_TRANSACTION_COLUMNS,
        "Transaction dataset",
    )
    validate_columns(
        identities,
        REQUIRED_IDENTITY_COLUMNS,
        "Identity dataset",
    )

    return transactions, identities


def join_datasets(
    transactions: pd.DataFrame,
    identities: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join identity records to transactions using TransactionID."""
    if transactions[JOIN_KEY].duplicated().any():
        raise ValueError("TransactionID is duplicated in the transaction dataset.")

    if identities[JOIN_KEY].duplicated().any():
        raise ValueError("TransactionID is duplicated in the identity dataset.")

    joined = transactions.merge(
        identities,
        on=JOIN_KEY,
        how="left",
        suffixes=("", "_identity"),
        validate="one_to_one",
        indicator=True,
    )

    identity_matches = int((joined["_merge"] == "both").sum())
    transaction_rows = len(transactions)
    identity_rows = len(identities)

    coverage = identity_matches / transaction_rows if transaction_rows else 0.0

    manifest = {
        "transaction_rows": transaction_rows,
        "identity_rows": identity_rows,
        "joined_rows": len(joined),
        "identity_matches": identity_matches,
        "identity_join_coverage": coverage,
        "unmatched_transactions": transaction_rows - identity_matches,
    }

    joined = joined.drop(columns=["_merge"])

    return joined, manifest


def write_outputs(
    joined: pd.DataFrame,
    manifest: dict[str, Any],
    output_directory: Path,
    manifest_path: Path,
    raw_directory: Path,
) -> None:
    """Write the joined Parquet file and ingestion manifest."""
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "train_joined.parquet"

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing intermediate file: {output_path}. "
            "Delete it only after verifying the previous ingestion."
        )

    run_manifest = {
        "dataset_name": "IEEE-CIS Fraud Detection",
        "ingestion_timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_directory": str(raw_directory.resolve()),
        "transaction_file": TRANSACTION_FILE,
        "identity_file": IDENTITY_FILE,
        "join_key": JOIN_KEY,
        "join_type": "left",
        "output_file": str(output_path.resolve()),
        "output_rows": len(joined),
        "output_columns": len(joined.columns),
        **manifest,
    }

    joined.to_parquet(output_path, index=False)
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Joined Parquet written to: {output_path}")
    print(f"Manifest written to: {manifest_path}")
    print(f"Joined rows: {len(joined):,}")
    print(f"Joined columns: {len(joined.columns):,}")
    print(f"Identity join coverage: {manifest['identity_join_coverage']:.2%}")


def main() -> int:
    """Run the IEEE-CIS ingestion pipeline."""
    arguments = parse_arguments()

    try:
        transactions, identities = load_raw_data(arguments.raw_directory)
        joined, join_manifest = join_datasets(transactions, identities)
        write_outputs(
            joined=joined,
            manifest=join_manifest,
            output_directory=arguments.output_directory,
            manifest_path=arguments.manifest,
            raw_directory=arguments.raw_directory,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
