"""
Utilities for checking local SHAP explanation stability.

These functions compare the leading absolute SHAP contributors for the same
transactions across multiple deterministic background samples. This evaluates
attribution consistency under changes to the SHAP reference sample. It does
not establish causal stability or model-performance stability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def validate_top_feature_count(
    top_feature_count: int,
    available_feature_count: int,
) -> None:
    """
    Validate the requested number of leading contributors.

    Args:
        top_feature_count: Number of largest absolute SHAP contributors to use.
        available_feature_count: Number of transformed model features.

    Raises:
        ValueError: If the requested top-feature count is invalid.
    """
    if top_feature_count <= 0:
        raise ValueError("top_feature_count must be greater than zero.")

    if top_feature_count > available_feature_count:
        raise ValueError(
            "top_feature_count cannot exceed the available transformed "
            f"feature count of {available_feature_count}."
        )


def get_top_absolute_feature_set(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    top_feature_count: int,
) -> set[str]:
    """
    Return feature names with the largest absolute SHAP contributions.

    Args:
        shap_values: One-dimensional SHAP values for a single transaction.
        feature_names: Transformed model-feature names aligned with SHAP values.
        top_feature_count: Number of leading absolute contributors to return.

    Returns:
        Set of transformed feature names.

    Raises:
        ValueError: If the SHAP vector and feature names are incompatible.
    """
    shap_array = np.asarray(shap_values, dtype=float)

    if shap_array.ndim != 1:
        raise ValueError("shap_values must be a one-dimensional array for one transaction.")

    if len(shap_array) != len(feature_names):
        raise ValueError("shap_values length must match the number of feature names.")

    if not np.isfinite(shap_array).all():
        raise ValueError("shap_values must contain only finite values.")

    validate_top_feature_count(
        top_feature_count=top_feature_count,
        available_feature_count=len(feature_names),
    )

    descending_indices = np.argsort(
        -np.abs(shap_array),
        kind="stable",
    )

    top_indices = descending_indices[:top_feature_count]

    return {str(feature_names[index]) for index in top_indices}


def calculate_jaccard_similarity(
    first_feature_set: set[str],
    second_feature_set: set[str],
) -> float:
    """
    Calculate Jaccard similarity between two feature-name sets.

    Args:
        first_feature_set: First top-contributor feature set.
        second_feature_set: Second top-contributor feature set.

    Returns:
        Intersection size divided by union size.

    Raises:
        ValueError: If either feature set is empty.
    """
    if not first_feature_set or not second_feature_set:
        raise ValueError("Jaccard similarity requires two non-empty feature sets.")

    union = first_feature_set | second_feature_set
    intersection = first_feature_set & second_feature_set

    return len(intersection) / len(union)


def evaluate_shap_stability(
    shap_values_by_seed: Mapping[int, np.ndarray],
    feature_names: Sequence[str],
    case_group_names: Sequence[str],
    top_feature_count: int,
) -> pd.DataFrame:
    """
    Compare top SHAP contributors across background-sample seeds.

    Each value in shap_values_by_seed must contain rows in the same order as
    case_group_names. For every case and every pair of seeds, the function
    compares the top absolute SHAP feature sets and records their Jaccard score.

    Args:
        shap_values_by_seed: Mapping from background seed to a two-dimensional
            SHAP matrix with shape (number_of_cases, number_of_features).
        feature_names: Transformed feature names aligned to each SHAP matrix.
        case_group_names: Ordered representative-case labels.
        top_feature_count: Number of top absolute contributors to compare.

    Returns:
        Pairwise stability table with one row per case and seed pair.

    Raises:
        ValueError: If fewer than two seeds, incompatible shapes, invalid
            feature names, or invalid case labels are supplied.
    """
    if len(shap_values_by_seed) < 2:
        raise ValueError("At least two background seeds are required for stability analysis.")

    if not case_group_names:
        raise ValueError("case_group_names must not be empty.")

    if len(set(case_group_names)) != len(case_group_names):
        raise ValueError("case_group_names must be unique.")

    validate_top_feature_count(
        top_feature_count=top_feature_count,
        available_feature_count=len(feature_names),
    )

    expected_case_count = len(case_group_names)
    expected_feature_count = len(feature_names)

    validated_values_by_seed: dict[int, np.ndarray] = {}

    for seed, shap_values in shap_values_by_seed.items():
        shap_array = np.asarray(shap_values, dtype=float)

        if shap_array.ndim != 2:
            raise ValueError(
                "Each SHAP value entry must be a two-dimensional array with "
                "one row per representative case."
            )

        if shap_array.shape[0] != expected_case_count:
            raise ValueError(
                f"Seed {seed} has {shap_array.shape[0]} SHAP rows, but "
                f"{expected_case_count} case groups were supplied."
            )

        if shap_array.shape[1] != expected_feature_count:
            raise ValueError(
                f"Seed {seed} has {shap_array.shape[1]} SHAP columns, but "
                f"{expected_feature_count} feature names were supplied."
            )

        if not np.isfinite(shap_array).all():
            raise ValueError(f"Seed {seed} contains non-finite SHAP values.")

        validated_values_by_seed[int(seed)] = shap_array

    sorted_seeds = sorted(validated_values_by_seed)
    stability_rows: list[dict[str, object]] = []

    for case_index, case_group in enumerate(case_group_names):
        top_features_by_seed = {
            seed: get_top_absolute_feature_set(
                shap_values=validated_values_by_seed[seed][case_index],
                feature_names=feature_names,
                top_feature_count=top_feature_count,
            )
            for seed in sorted_seeds
        }

        for first_position, first_seed in enumerate(sorted_seeds[:-1]):
            for second_seed in sorted_seeds[first_position + 1 :]:
                first_feature_set = top_features_by_seed[first_seed]
                second_feature_set = top_features_by_seed[second_seed]

                shared_features = sorted(first_feature_set & second_feature_set)
                union_features = first_feature_set | second_feature_set

                stability_rows.append(
                    {
                        "case_group": case_group,
                        "seed_a": first_seed,
                        "seed_b": second_seed,
                        "top_feature_count": top_feature_count,
                        "shared_top_feature_count": len(shared_features),
                        "union_top_feature_count": len(union_features),
                        "jaccard_similarity": calculate_jaccard_similarity(
                            first_feature_set,
                            second_feature_set,
                        ),
                        "shared_top_features": " | ".join(shared_features),
                    }
                )

    return pd.DataFrame(stability_rows).sort_values(
        by=["case_group", "seed_a", "seed_b"],
        ignore_index=True,
    )


def summarise_shap_stability(
    pairwise_stability_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean and minimum pairwise Jaccard stability by case group.

    Args:
        pairwise_stability_table: Output from evaluate_shap_stability().

    Returns:
        One stability-summary row per representative-case group.

    Raises:
        ValueError: If required columns are missing or the table is empty.
    """
    required_columns = {
        "case_group",
        "jaccard_similarity",
        "shared_top_feature_count",
    }

    if pairwise_stability_table.empty:
        raise ValueError("pairwise_stability_table must not be empty.")

    missing_columns = required_columns - set(pairwise_stability_table.columns)

    if missing_columns:
        raise ValueError(
            "pairwise_stability_table is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    summary_table = (
        pairwise_stability_table.groupby("case_group", as_index=False)
        .agg(
            seed_pair_count=("jaccard_similarity", "count"),
            mean_jaccard_similarity=("jaccard_similarity", "mean"),
            min_jaccard_similarity=("jaccard_similarity", "min"),
            max_jaccard_similarity=("jaccard_similarity", "max"),
            mean_shared_top_feature_count=(
                "shared_top_feature_count",
                "mean",
            ),
        )
        .sort_values("case_group", ignore_index=True)
    )

    return summary_table
