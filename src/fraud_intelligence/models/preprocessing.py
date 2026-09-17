"""Reusable preprocessing pipelines for Phase 4 fraud models.

The preprocessing object is fitted only on training data. It then transforms
validation data using the training-fitted imputation, scaling, and categorical
encoding rules.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    preprocessing_config: dict[str, Any],
) -> ColumnTransformer:
    """Build the common Phase 4 preprocessing pipeline.

    Parameters
    ----------
    numeric_columns:
        Numeric feature columns selected by the modelling data contract.
    categorical_columns:
        Categorical feature columns selected by the modelling data contract.
    preprocessing_config:
        The ``preprocessing`` section from ``configs/modeling.yaml``.

    Returns
    -------
    ColumnTransformer
        An unfitted transformer. Call ``fit`` only with training features.
    """
    if not numeric_columns and not categorical_columns:
        raise ValueError(
            "Cannot build a preprocessing pipeline without numeric or "
            "categorical feature columns."
        )

    numeric_config = preprocessing_config["numeric"]
    categorical_config = preprocessing_config["categorical"]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=numeric_config["imputation_strategy"],
                    add_indicator=numeric_config["add_missing_indicator"],
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=categorical_config["imputation_strategy"],
                ),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown=categorical_config["handle_unknown"],
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
