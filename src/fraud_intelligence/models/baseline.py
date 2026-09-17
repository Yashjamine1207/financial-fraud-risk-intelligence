"""Regularised Logistic Regression baseline for Phase 4."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_intelligence.models.preprocessing import build_model_preprocessor


def build_logistic_regression(
    model_config: dict[str, Any],
) -> LogisticRegression:
    """Build an unfitted regularised Logistic Regression classifier.

    Parameters
    ----------
    model_config:
        The ``model`` section from ``configs/logistic_regression.yaml``.

    Returns
    -------
    LogisticRegression
        An unfitted class-weighted Logistic Regression classifier using
        L2 regularisation through ``l1_ratio=0.0``.
    """
    required_parameters = {
        "C",
        "l1_ratio",
        "solver",
        "max_iter",
        "tolerance",
        "class_weight",
        "random_state",
    }
    missing_parameters = required_parameters.difference(model_config)

    if missing_parameters:
        missing = ", ".join(sorted(missing_parameters))
        raise ValueError(
            "Logistic Regression configuration is missing required " f"parameter(s): {missing}"
        )

    return LogisticRegression(
        C=model_config["C"],
        l1_ratio=model_config["l1_ratio"],
        solver=model_config["solver"],
        max_iter=model_config["max_iter"],
        tol=model_config["tolerance"],
        class_weight=model_config["class_weight"],
        random_state=model_config["random_state"],
    )


def build_logistic_regression_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    preprocessing_config: dict[str, Any],
    model_config: dict[str, Any],
) -> Pipeline:
    """Build the complete unfitted Logistic Regression baseline pipeline.

    The returned pipeline contains:

    1. Training-fitted numeric imputation, missingness indicators, and scaling.
    2. Training-fitted categorical imputation and one-hot encoding.
    3. Regularised, class-weighted Logistic Regression.

    Fit this pipeline only with the chronological training split.
    """
    preprocessor = build_model_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        preprocessing_config=preprocessing_config,
    )
    classifier = build_logistic_regression(model_config=model_config)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
