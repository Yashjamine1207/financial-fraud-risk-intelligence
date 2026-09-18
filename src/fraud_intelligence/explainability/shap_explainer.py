"""
Reusable SHAP explainability utilities for the selected fraud-risk model.

This module creates global and local SHAP explanations for the frozen Phase 5
XGBoost model. SHAP values describe how input features contributed to a model
prediction. They do not prove fraud or establish that a feature caused fraud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ShapExplanationResult:
    """
    Stores SHAP values and the transformed feature data used to calculate them.

    Attributes:
        shap_values: SHAP contribution values with shape
            (number_of_rows, number_of_transformed_features).
        transformed_features: Model-ready feature matrix after applying the
            frozen training preprocessor.
        feature_names: Names of transformed model features. Categorical fields
            may expand into multiple one-hot-encoded features.
        base_value: Reference model output used as the SHAP baseline.
    """

    shap_values: np.ndarray
    transformed_features: Any
    feature_names: list[str]
    base_value: float


@dataclass(frozen=True)
class LocalExplanation:
    """
    Stores the strongest feature contributions for one transaction.

    Positive SHAP values move the model output toward a higher fraud-risk
    prediction relative to its baseline. Negative values move it lower.
    This is model attribution, not causal evidence.
    """

    transaction_id: int
    predicted_probability: float
    base_value: float
    top_positive_contributors: pd.DataFrame
    top_negative_contributors: pd.DataFrame


def validate_feature_frame(
    feature_frame: pd.DataFrame,
    expected_input_features: list[str],
) -> pd.DataFrame:
    """
    Validate and order raw model-input features before preprocessing.

    The frozen model expects exactly the Phase 4 feature contract. Identifier,
    target, and timestamp columns must already be excluded by the caller.

    Args:
        feature_frame: Raw feature DataFrame to validate.
        expected_input_features: Ordered raw feature list from the preprocessor.

    Returns:
        A copy of the feature frame ordered according to the model contract.

    Raises:
        TypeError: If feature_frame is not a pandas DataFrame.
        ValueError: If expected model features are missing.
    """
    if not isinstance(feature_frame, pd.DataFrame):
        raise TypeError("feature_frame must be a pandas DataFrame.")

    missing_features = [
        feature_name
        for feature_name in expected_input_features
        if feature_name not in feature_frame.columns
    ]

    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise ValueError(
            "Feature frame is missing columns required by the frozen "
            f"preprocessor. Missing examples: {preview}"
        )

    return feature_frame.loc[:, expected_input_features].copy()


def get_raw_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract the raw input feature names expected by a fitted preprocessor.

    ColumnTransformer stores the selected raw input columns for each internal
    transformer after fitting. This function combines those columns without
    duplicates while preserving their first appearance.

    Args:
        preprocessor: Fitted Phase 4 ColumnTransformer.

    Returns:
        Ordered raw feature names required before transformation.

    Raises:
        TypeError: If preprocessor is not a fitted ColumnTransformer.
        ValueError: If no input feature names can be extracted.
    """
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("preprocessor must be a sklearn ColumnTransformer.")

    raw_feature_names: list[str] = []

    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop" or columns is None:
            continue

        if isinstance(columns, slice):
            raise TypeError(
                "The fitted preprocessor uses slice-based feature selection, "
                "which is not supported by this explainability module."
            )

        for column in columns:
            if isinstance(column, str) and column not in raw_feature_names:
                raw_feature_names.append(column)

    if not raw_feature_names:
        raise ValueError(
            "Could not extract raw input feature names from the fitted " "preprocessor."
        )

    return raw_feature_names


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract transformed feature names after imputation, indicators, scaling,
    and one-hot encoding.

    These are the exact columns consumed by the trained XGBoost classifier.

    Args:
        preprocessor: Fitted Phase 4 ColumnTransformer.

    Returns:
        Names of transformed model features.

    Raises:
        TypeError: If preprocessor is not a fitted ColumnTransformer.
        ValueError: If the preprocessor cannot provide output names.
    """
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("preprocessor must be a sklearn ColumnTransformer.")

    try:
        transformed_feature_names = preprocessor.get_feature_names_out()
    except AttributeError as error:
        raise ValueError(
            "The fitted preprocessor does not expose transformed feature names."
        ) from error

    return [str(feature_name) for feature_name in transformed_feature_names]


def transform_features_for_explanation(
    preprocessor: ColumnTransformer,
    raw_feature_frame: pd.DataFrame,
) -> tuple[Any, list[str]]:
    """
    Transform raw model features using the frozen training-fitted preprocessor.

    No fitting occurs here. The function only applies preprocessing already
    learned from chronological training data.

    Args:
        preprocessor: Frozen fitted ColumnTransformer from MLflow.
        raw_feature_frame: Validated raw feature frame in model-input order.

    Returns:
        Tuple containing the transformed feature matrix and transformed names.
    """
    transformed_features = preprocessor.transform(raw_feature_frame)
    transformed_feature_names = get_transformed_feature_names(preprocessor)

    if transformed_features.shape[1] != len(transformed_feature_names):
        raise ValueError(
            "Transformed feature count does not match the number of extracted "
            "transformed feature names."
        )

    return transformed_features, transformed_feature_names


def build_tree_explainer(
    classifier: XGBClassifier,
    background_features: Any,
) -> shap.TreeExplainer:
    """
    Create a TreeExplainer for the frozen XGBoost classifier.

    The background set should be a deterministic sample of transformed training
    data. It provides the reference distribution for SHAP explanations. Sparse
    preprocessor output is converted to a dense matrix because SHAP's
    interventional independent masker requires dense numeric background data.

    Args:
        classifier: Frozen fitted XGBClassifier from MLflow.
        background_features: Transformed training feature sample.

    Returns:
        Configured SHAP TreeExplainer.

    Raises:
        TypeError: If classifier is not an XGBClassifier.
        ValueError: If the background dataset is empty.
    """
    if not isinstance(classifier, XGBClassifier):
        raise TypeError("classifier must be an XGBClassifier.")

    if background_features.shape[0] == 0:
        raise ValueError("background_features must contain at least one row.")

    dense_background_features = (
        background_features.toarray()
        if sparse.issparse(background_features)
        else np.asarray(background_features)
    )
    background_masker = shap.maskers.Independent(
        dense_background_features,
        max_samples=dense_background_features.shape[0],
    )

    return shap.TreeExplainer(
        model=classifier,
        data=background_masker,
        feature_perturbation="interventional",
        model_output="raw",
    )


def calculate_shap_values(
    explainer: shap.TreeExplainer,
    transformed_features: Any,
    feature_names: list[str],
) -> ShapExplanationResult:
    """
    Calculate SHAP values for transformed model inputs.

    The XGBoost classifier is explained on its raw margin because TreeExplainer
    with interventional background data supports a stable additive explanation
    on that scale. Predicted probabilities are calculated separately later.

    Args:
        explainer: Configured SHAP TreeExplainer.
        transformed_features: Preprocessed feature matrix to explain.
        feature_names: Names corresponding to matrix columns.

    Returns:
        A ShapExplanationResult containing SHAP values and explanation metadata.

    Raises:
        ValueError: If the supplied matrix has no rows or incompatible columns.
    """
    if transformed_features.shape[0] == 0:
        raise ValueError("transformed_features must contain at least one row.")

    if transformed_features.shape[1] != len(feature_names):
        raise ValueError(
            "The transformed feature matrix column count does not match " "feature_names."
        )

    features_for_shap = (
        transformed_features.toarray()
        if sparse.issparse(transformed_features)
        else np.asarray(transformed_features)
    )
    shap_values = explainer.shap_values(features_for_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values_array = np.asarray(shap_values)

    if shap_values_array.ndim != 2:
        raise ValueError("Expected two-dimensional SHAP values for binary classification.")

    if shap_values_array.shape != transformed_features.shape:
        raise ValueError("SHAP output shape does not match the transformed feature matrix.")

    expected_value = explainer.expected_value

    if isinstance(expected_value, np.ndarray):
        base_value = float(np.ravel(expected_value)[-1])
    else:
        base_value = float(expected_value)

    return ShapExplanationResult(
        shap_values=shap_values_array,
        transformed_features=transformed_features,
        feature_names=feature_names,
        base_value=base_value,
    )


def calculate_global_feature_importance(
    explanation_result: ShapExplanationResult,
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP importance for every transformed feature.

    A larger mean absolute SHAP value means the feature moved predictions more,
    on average, across the explained transaction sample. It does not mean the
    feature caused fraud or is independently predictive.

    Args:
        explanation_result: Completed SHAP calculation output.

    Returns:
        DataFrame ranked from largest to smallest mean absolute contribution.
    """
    mean_absolute_shap = np.mean(
        np.abs(explanation_result.shap_values),
        axis=0,
    )

    importance_table = pd.DataFrame(
        {
            "transformed_feature": explanation_result.feature_names,
            "mean_absolute_shap_value": mean_absolute_shap,
        }
    )

    return importance_table.sort_values(
        by="mean_absolute_shap_value",
        ascending=False,
        ignore_index=True,
    )


def _extract_feature_values(
    transformed_features: Any,
    row_index: int,
) -> np.ndarray:
    """
    Extract one transformed feature row as a dense one-dimensional array.

    Args:
        transformed_features: Dense or sparse transformed feature matrix.
        row_index: Position of the requested row.

    Returns:
        Dense feature values for one transaction.
    """
    if sparse.issparse(transformed_features):
        return transformed_features.getrow(row_index).toarray().ravel()

    return np.asarray(transformed_features[row_index]).ravel()


def create_local_explanation(
    explanation_result: ShapExplanationResult,
    transaction_id: int,
    row_index: int,
    predicted_probability: float,
    top_feature_count: int,
) -> LocalExplanation:
    """
    Create the strongest positive and negative SHAP contributors for one row.

    Args:
        explanation_result: SHAP result for the complete explained sample.
        transaction_id: TransactionID of the selected transaction.
        row_index: Row position of that transaction in the explained sample.
        predicted_probability: Frozen-model fraud probability for the row.
        top_feature_count: Maximum number of positive and negative contributors.

    Returns:
        Structured local explanation for one transaction.

    Raises:
        ValueError: If row_index or top_feature_count is invalid.
    """
    row_count = explanation_result.shap_values.shape[0]

    if row_index < 0 or row_index >= row_count:
        raise ValueError(f"row_index must be between 0 and {row_count - 1}, got {row_index}.")

    if top_feature_count <= 0:
        raise ValueError("top_feature_count must be greater than zero.")

    if not 0.0 <= predicted_probability <= 1.0:
        raise ValueError("predicted_probability must be a finite value between 0 and 1.")

    row_shap_values = explanation_result.shap_values[row_index]
    row_feature_values = _extract_feature_values(
        explanation_result.transformed_features,
        row_index,
    )

    contributor_table = pd.DataFrame(
        {
            "transformed_feature": explanation_result.feature_names,
            "transformed_feature_value": row_feature_values,
            "shap_value": row_shap_values,
        }
    )

    positive_contributors = (
        contributor_table.loc[contributor_table["shap_value"] > 0]
        .sort_values("shap_value", ascending=False)
        .head(top_feature_count)
        .reset_index(drop=True)
    )

    negative_contributors = (
        contributor_table.loc[contributor_table["shap_value"] < 0]
        .sort_values("shap_value", ascending=True)
        .head(top_feature_count)
        .reset_index(drop=True)
    )

    return LocalExplanation(
        transaction_id=int(transaction_id),
        predicted_probability=float(predicted_probability),
        base_value=explanation_result.base_value,
        top_positive_contributors=positive_contributors,
        top_negative_contributors=negative_contributors,
    )
