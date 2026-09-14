"""Load IEEE-CIS transaction and identity data into PostgreSQL."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Connection
from sqlalchemy.exc import SQLAlchemyError

DEFAULT_PARQUET_PATH = Path("data/interim/joined_transactions/train_joined.parquet")
DEFAULT_MANIFEST_PATH = Path("data/interim/joined_transactions/ingestion_manifest.json")
DEFAULT_RAW_IDENTITY_PATH = Path("data/raw/ieee_cis/train_identity.csv")

TRANSACTIONS_TABLE = "transactions"
IDENTITIES_TABLE = "identities"
INGESTION_RUNS_TABLE = "ingestion_runs"


def build_database_url() -> URL:
    """Build the PostgreSQL connection URL from environment variables."""
    required_variables = (
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
    )
    missing_variables = [variable for variable in required_variables if not os.getenv(variable)]

    if missing_variables:
        missing_text = ", ".join(missing_variables)
        raise RuntimeError(f"Missing PostgreSQL environment variables: {missing_text}")

    return URL.create(
        drivername="postgresql+psycopg2",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        database=os.environ["POSTGRES_DB"],
    )


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the reproducibility manifest created during ingestion."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing ingestion manifest: {manifest_path}")

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def prepare_transactions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Select and rename transaction columns for PostgreSQL."""
    required_columns = {
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "isFraud",
    }
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing transaction columns: {missing_text}")

    transactions = dataframe[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].copy()
    transactions = transactions.rename(
        columns={
            "TransactionID": "transaction_id",
            "TransactionDT": "transaction_dt",
            "TransactionAmt": "transaction_amount",
            "isFraud": "is_fraud",
        }
    )

    if transactions["transaction_id"].duplicated().any():
        raise ValueError("TransactionID is duplicated in the transaction data.")

    transactions["transaction_id"] = transactions["transaction_id"].astype("int64")
    transactions["transaction_dt"] = transactions["transaction_dt"].astype("int64")
    transactions["transaction_amount"] = transactions["transaction_amount"].astype(float)
    transactions["is_fraud"] = transactions["is_fraud"].astype("int16")

    return transactions


def prepare_identities(identity_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert raw IEEE-CIS identity records into JSON payloads."""
    if "TransactionID" not in identity_dataframe.columns:
        raise ValueError("The raw identity dataset is missing the TransactionID column.")

    if identity_dataframe["TransactionID"].duplicated().any():
        raise ValueError("TransactionID is duplicated in the identity dataset.")

    identity_columns = [
        column for column in identity_dataframe.columns if column != "TransactionID"
    ]
    if not identity_columns:
        raise ValueError("The identity dataset does not contain identity fields.")

    identities = identity_dataframe.rename(columns={"TransactionID": "transaction_id"}).copy()

    def build_payload(row: pd.Series) -> str:
        """Create a JSON payload containing non-null identity values."""
        payload = {column: row[column] for column in identity_columns if pd.notna(row[column])}
        return json.dumps(payload, default=str)

    identities["identity_payload"] = identities.apply(build_payload, axis=1)
    return identities[["transaction_id", "identity_payload"]]


def insert_ingestion_run(
    connection: Connection,
    manifest: dict[str, Any],
    dataframe: pd.DataFrame,
    source_path: Path,
) -> int:
    """Insert ingestion metadata and return the generated run identifier."""
    schema_payload = {
        "columns": list(dataframe.columns),
        "column_count": len(dataframe.columns),
    }
    now_utc = datetime.now(UTC)

    result = connection.execute(
        text(f"""
            INSERT INTO {INGESTION_RUNS_TABLE} (
                dataset_name,
                source_version,
                source_path,
                transaction_rows,
                identity_rows,
                joined_rows,
                identity_join_coverage,
                schema_json,
                started_at_utc,
                completed_at_utc
            )
            VALUES (
                :dataset_name,
                :source_version,
                :source_path,
                :transaction_rows,
                :identity_rows,
                :joined_rows,
                :identity_join_coverage,
                CAST(:schema_json AS JSONB),
                :started_at_utc,
                :completed_at_utc
            )
            RETURNING ingestion_run_id
            """),
        {
            "dataset_name": manifest["dataset_name"],
            "source_version": manifest.get("source_version"),
            "source_path": str(source_path.resolve()),
            "transaction_rows": manifest["transaction_rows"],
            "identity_rows": manifest["identity_rows"],
            "joined_rows": manifest["joined_rows"],
            "identity_join_coverage": manifest["identity_join_coverage"],
            "schema_json": json.dumps(schema_payload),
            "started_at_utc": now_utc,
            "completed_at_utc": now_utc,
        },
    )
    return int(result.scalar_one())


def load_data(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    raw_identity_path: Path = DEFAULT_RAW_IDENTITY_PATH,
) -> None:
    """Load joined transactions and raw identities into PostgreSQL."""
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing Parquet file: {parquet_path}")
    if not raw_identity_path.is_file():
        raise FileNotFoundError(f"Missing raw identity file: {raw_identity_path}")

    load_dotenv()
    manifest = load_manifest(manifest_path)
    engine = create_engine(build_database_url())

    joined_dataframe = pd.read_parquet(parquet_path)
    raw_identity_dataframe = pd.read_csv(raw_identity_path)
    transactions = prepare_transactions(joined_dataframe)
    identities = prepare_identities(raw_identity_dataframe)

    if len(transactions) != manifest["transaction_rows"]:
        raise ValueError("Transaction count does not match the ingestion manifest.")
    if len(identities) != manifest["identity_rows"]:
        raise ValueError("Identity count does not match the ingestion manifest.")

    with engine.begin() as connection:
        ingestion_run_id = insert_ingestion_run(
            connection,
            manifest,
            joined_dataframe,
            parquet_path,
        )
        connection.execute(
            text(f"""
                INSERT INTO {TRANSACTIONS_TABLE} (
                    transaction_id,
                    transaction_dt,
                    transaction_amount,
                    is_fraud,
                    ingestion_run_id
                )
                VALUES (
                    :transaction_id,
                    :transaction_dt,
                    :transaction_amount,
                    :is_fraud,
                    :ingestion_run_id
                )
                ON CONFLICT (transaction_id) DO NOTHING
                """),
            [
                {**row, "ingestion_run_id": ingestion_run_id}
                for row in transactions.to_dict(orient="records")
            ],
        )
        connection.execute(
            text(f"""
                INSERT INTO {IDENTITIES_TABLE} (
                    transaction_id,
                    identity_payload,
                    ingestion_run_id
                )
                VALUES (
                    :transaction_id,
                    CAST(:identity_payload AS JSONB),
                    :ingestion_run_id
                )
                ON CONFLICT (transaction_id) DO NOTHING
                """),
            [
                {**row, "ingestion_run_id": ingestion_run_id}
                for row in identities.to_dict(orient="records")
            ],
        )

    print(f"Loaded transactions: {len(transactions):,}")
    print(f"Loaded identities: {len(identities):,}")
    print(f"Ingestion run ID: {ingestion_run_id}")


def main() -> int:
    """Run the PostgreSQL loading command."""
    try:
        load_data()
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        OSError,
        SQLAlchemyError,
    ) as error:
        print(f"PostgreSQL loading failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
