"""Inspect the Phase 4 XGBoost MLflow run before Phase 5 calibration.

This script is read-only:
- It does not train a model.
- It does not modify MLflow artifacts.
- It does not load transaction feature tables.
- It does not access the locked final test split.

Its only purpose is to identify the exact MLflow artifact path used when the
Phase 4 XGBoost model and preprocessing pipeline were logged.
"""

from __future__ import annotations

import os
from typing import Any

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

PHASE4_XGBOOST_RUN_ID = "d505d735ef344b1e81205c9453ddb1cf"  # Update this with the actual run ID if needed


def print_artifacts(
    client: MlflowClient,
    run_id: str,
    artifact_path: str | None = None,
    indentation: str = "",
) -> None:
    """Recursively print all MLflow artifacts for one experiment run."""
    artifacts = client.list_artifacts(run_id, path=artifact_path)

    if not artifacts:
        location = artifact_path or "/"
        print(f"{indentation}(No artifacts found in: {location})")
        return

    for artifact in artifacts:
        artifact_type = "directory" if artifact.is_dir else "file"
        print(f"{indentation}- [{artifact_type}] {artifact.path}")

        if artifact.is_dir:
            print_artifacts(
                client=client,
                run_id=run_id,
                artifact_path=artifact.path,
                indentation=f"{indentation}  ",
            )


def print_mapping(title: str, values: dict[str, Any]) -> None:
    """Print run metadata in deterministic key order."""
    print(f"\n{title}")

    if not values:
        print("  (None recorded)")
        return

    for key in sorted(values):
        print(f"  {key}: {values[key]}")


def main() -> None:
    """Connect to local MLflow and print Phase 4 XGBoost run details."""
    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. "
            "Add it to your local .env file before running this script."
        )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    run = client.get_run(PHASE4_XGBOOST_RUN_ID)

    print("Phase 4 XGBoost MLflow run found successfully.")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Run status: {run.info.status}")
    print(f"Artifact URI: {run.info.artifact_uri}")

    print_mapping("Run parameters:", dict(run.data.params))
    print_mapping("Run tags:", dict(run.data.tags))
    print_mapping("Run metrics:", dict(run.data.metrics))

    print("\nLogged MLflow artifacts:")
    print_artifacts(
        client=client,
        run_id=PHASE4_XGBOOST_RUN_ID,
    )


if __name__ == "__main__":
    main()