"""Record metadata and SHA-256 hashes for the local IEEE-CIS dataset files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET_SOURCE = "https://www.kaggle.com/competitions/ieee-fraud-detection/data"
COMPETITION_RULES = "https://www.kaggle.com/competitions/ieee-fraud-detection/rules"

EXPECTED_FILES = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
)


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 hash of a file without loading it fully into memory."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def get_file_metadata(file_path: Path) -> dict[str, Any]:
    """Return metadata for one dataset file."""
    file_hash = calculate_sha256(file_path)
    file_size_bytes = file_path.stat().st_size

    return {
        "file_name": file_path.name,
        "relative_path": str(file_path),
        "file_size_bytes": file_size_bytes,
        "sha256": file_hash,
    }


def record_dataset_metadata(raw_directory: Path, output_path: Path) -> None:
    """Record metadata for all expected IEEE-CIS files."""
    missing_files = [
        file_name for file_name in EXPECTED_FILES if not (raw_directory / file_name).is_file()
    ]

    if missing_files:
        missing_text = "\n".join(f"- {file_name}" for file_name in missing_files)
        raise FileNotFoundError(
            "The following IEEE-CIS files were not found:\n"
            f"{missing_text}\n\n"
            f"Expected directory: {raw_directory.resolve()}"
        )

    files_metadata = [get_file_metadata(raw_directory / file_name) for file_name in EXPECTED_FILES]

    metadata = {
        "dataset_name": "IEEE-CIS Fraud Detection",
        "source_url": DATASET_SOURCE,
        "competition_rules_url": COMPETITION_RULES,
        "licence_note": (
            "Kaggle competition data. Access and use are subject to the "
            "IEEE-CIS Fraud Detection competition rules accepted on Kaggle."
        ),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "raw_directory": str(raw_directory.resolve()),
        "files": files_metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Metadata written to: {output_path}")
    print(f"Files recorded: {len(files_metadata)}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Record IEEE-CIS dataset file metadata and SHA-256 hashes."
    )
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=Path("data/raw/ieee_cis"),
        help="Directory containing the raw IEEE-CIS CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/ieee_cis/metadata.json"),
        help="Output JSON path for dataset metadata.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the metadata-recording command."""
    arguments = parse_arguments()

    try:
        record_dataset_metadata(
            raw_directory=arguments.raw_directory,
            output_path=arguments.output,
        )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
