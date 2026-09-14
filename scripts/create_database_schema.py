"""Create the PostgreSQL schema for the fraud-intelligence platform."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    ingestion_run_id BIGSERIAL PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    source_version TEXT,
    source_path TEXT NOT NULL,
    transaction_rows BIGINT NOT NULL,
    identity_rows BIGINT NOT NULL,
    joined_rows BIGINT NOT NULL,
    identity_join_coverage DOUBLE PRECISION NOT NULL,
    schema_json JSONB NOT NULL,
    started_at_utc TIMESTAMPTZ NOT NULL,
    completed_at_utc TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id BIGINT PRIMARY KEY,
    transaction_dt BIGINT NOT NULL,
    transaction_amount DOUBLE PRECISION NOT NULL,
    is_fraud SMALLINT,
    ingestion_run_id BIGINT REFERENCES ingestion_runs(ingestion_run_id),
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS identities (
    transaction_id BIGINT PRIMARY KEY REFERENCES transactions(transaction_id),
    identity_payload JSONB NOT NULL,
    ingestion_run_id BIGINT REFERENCES ingestion_runs(ingestion_run_id),
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id BIGSERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_value_hash TEXT NOT NULL,
    first_seen_transaction_id BIGINT,
    last_seen_transaction_id BIGINT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (entity_type, entity_value_hash)
);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    feature_snapshot_id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES transactions(transaction_id),
    feature_version TEXT NOT NULL,
    feature_payload JSONB NOT NULL,
    available_at_utc TIMESTAMPTZ NOT NULL,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (transaction_id, feature_version)
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL UNIQUE,
    feature_version TEXT NOT NULL,
    calibration_version TEXT,
    training_dataset_version TEXT NOT NULL,
    parameters JSONB NOT NULL,
    metrics JSONB NOT NULL,
    registered_at_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES transactions(transaction_id),
    model_version TEXT NOT NULL REFERENCES model_versions(model_version),
    fraud_probability DOUBLE PRECISION NOT NULL,
    risk_band TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    expected_cost DOUBLE PRECISION,
    scored_at_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS investigation_cases (
    case_id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL REFERENCES transactions(transaction_id),
    prediction_id BIGINT REFERENCES predictions(prediction_id),
    case_status TEXT NOT NULL DEFAULT 'open',
    priority INTEGER NOT NULL,
    shap_payload JSONB,
    analyst_outcome TEXT,
    analyst_notes TEXT,
    opened_at_utc TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at_utc TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_transactions_transaction_dt
    ON transactions (transaction_dt);

CREATE INDEX IF NOT EXISTS idx_transactions_is_fraud
    ON transactions (is_fraud);

CREATE INDEX IF NOT EXISTS idx_predictions_action
    ON predictions (recommended_action);

CREATE INDEX IF NOT EXISTS idx_cases_status_priority
    ON investigation_cases (case_status, priority);
"""


def build_database_url() -> URL:
    """Build a PostgreSQL connection URL from environment variables."""
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


def create_schema() -> None:
    """Create all PostgreSQL tables and indexes."""
    load_dotenv()
    database_url = build_database_url()
    engine = create_engine(database_url)

    with engine.begin() as connection:
        for statement in SCHEMA_SQL.split(";"):
            cleaned_statement = statement.strip()

            if cleaned_statement:
                connection.execute(text(cleaned_statement))

    print("PostgreSQL schema created successfully.")


def main() -> int:
    """Run the schema creation command."""
    try:
        create_schema()
    except (RuntimeError, SQLAlchemyError, OSError) as error:
        print(f"Schema creation failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
