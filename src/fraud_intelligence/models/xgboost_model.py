"""XGBoost tabular fraud model for Phase 4."""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from fraud_intelligence.models.preprocessing import build_model_preprocessor


def calculate_scale_pos_weight(y_train: Any) -> float:
    """Calculate the XGBoost fraud-class weight from training labels only.

    The value is:

        number of non-fraud transactions / number of fraud transactions

    It is calculated exclusively from the chronological training split and
    never from validation or final test labels.
    """
    labels = np.asarray(y_train, dtype=np.int8)

    fraud_count = int((labels == 1).sum())
    non_fraud_count = int((labels == 0).sum())

    if fraud_count == 0:
        raise ValueError(
            "Cannot calculate scale_pos_weight because training data "
            "contains no fraud transactions."
        )

    if non_fraud_count == 0:
        raise ValueError(
            "Cannot calculate scale_pos_weight because training data "
            "contains no non-fraud transactions."
        )

    return float(non_fraud_count / fraud_count)


def build_xgboost_classifier(
    model_config: dict[str, Any],
    scale_pos_weight: float,
) -> XGBClassifier:
    """Build an unfitted XGBoost classifier from the YAML configuration."""
    required_parameters = {
        "objective",
        "eval_metric",
        "tree_method",
        "device",
        "n_estimators",
        "early_stopping_rounds",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "random_state",
        "n_jobs",
    }
    missing_parameters = required_parameters.difference(model_config)

    if missing_parameters:
        missing = ", ".join(sorted(missing_parameters))
        raise ValueError("XGBoost configuration is missing required parameter(s): " f"{missing}")

    if scale_pos_weight <= 0:
        raise ValueError("scale_pos_weight must be greater than zero.")

    return XGBClassifier(
        objective=model_config["objective"],
        eval_metric=model_config["eval_metric"],
        tree_method=model_config["tree_method"],
        device=model_config["device"],
        n_estimators=model_config["n_estimators"],
        early_stopping_rounds=model_config["early_stopping_rounds"],
        learning_rate=model_config["learning_rate"],
        max_depth=model_config["max_depth"],
        min_child_weight=model_config["min_child_weight"],
        subsample=model_config["subsample"],
        colsample_bytree=model_config["colsample_bytree"],
        reg_alpha=model_config["reg_alpha"],
        reg_lambda=model_config["reg_lambda"],
        gamma=model_config["gamma"],
        scale_pos_weight=scale_pos_weight,
        random_state=model_config["random_state"],
        n_jobs=model_config["n_jobs"],
    )


def build_xgboost_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    preprocessing_config: dict[str, Any],
):
    """Build the shared unfitted Phase 4 preprocessing transformer.

    The same selected feature set and transformation rules used for Logistic
    Regression are used for XGBoost. The transformer must be fitted only on
    chronological training data.
    """
    return build_model_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        preprocessing_config=preprocessing_config,
    )
