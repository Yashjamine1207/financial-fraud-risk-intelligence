"""Tests for the initial project setup."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist() -> None:
    """Check that the initial project documentation exists."""
    required_files = [
        "README.md",
        "LICENSE",
        ".gitignore",
        ".env.example",
        "pyproject.toml",
        "requirements-dev.txt",
        "Makefile",
        "docs/project_requirements.md",
        "docs/development_roadmap.md",
        "docs/database_schema.md",
        "docs/api_design.md",
        "docs/decision_log.md",
        "docs/cost_assumptions.md",
        "docs/data_dictionary.md",
    ]

    missing_files = [
        file_path for file_path in required_files if not (PROJECT_ROOT / file_path).is_file()
    ]

    assert not missing_files, f"Missing project files: {missing_files}"


def test_sensitive_data_is_outside_tracked_project_files() -> None:
    """Confirm that local datasets and environment secrets are not tracked."""
    sensitive_paths = [
        ".env",
        "ieee-fraud-detection.zip",
        "data/raw/ieee_cis/train_transaction.csv",
        "data/raw/ieee_cis/train_identity.csv",
        "data/raw/ieee_cis/test_transaction.csv",
        "data/raw/ieee_cis/test_identity.csv",
        "data/raw/ieee_cis/sample_submission.csv",
    ]

    for relative_path in sensitive_paths:
        path = PROJECT_ROOT / relative_path
        assert path.exists(), f"Expected local path does not exist: {relative_path}"
