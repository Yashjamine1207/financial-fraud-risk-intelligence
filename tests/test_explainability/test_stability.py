import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.explainability.stability import (
    calculate_jaccard_similarity,
    evaluate_shap_stability,
    get_top_absolute_feature_set,
    summarise_shap_stability,
    validate_top_feature_count,
)


def test_validate_top_feature_count_accepts_valid_request() -> None:
    validate_top_feature_count(
        top_feature_count=3,
        available_feature_count=5,
    )


@pytest.mark.parametrize(
    ("top_feature_count", "available_feature_count"),
    [
        (0, 5),
        (-1, 5),
        (6, 5),
    ],
)
def test_validate_top_feature_count_rejects_invalid_request(
    top_feature_count: int,
    available_feature_count: int,
) -> None:
    with pytest.raises(ValueError):
        validate_top_feature_count(
            top_feature_count=top_feature_count,
            available_feature_count=available_feature_count,
        )


def test_get_top_absolute_feature_set_uses_absolute_shap_values() -> None:
    feature_names = ["amount", "velocity", "recency", "device_risk"]
    shap_values = np.array([0.10, -0.90, 0.30, -0.50])

    result = get_top_absolute_feature_set(
        shap_values=shap_values,
        feature_names=feature_names,
        top_feature_count=2,
    )

    assert result == {"velocity", "device_risk"}


def test_get_top_absolute_feature_set_rejects_incompatible_lengths() -> None:
    with pytest.raises(ValueError, match="must match"):
        get_top_absolute_feature_set(
            shap_values=np.array([0.10, -0.20]),
            feature_names=["amount"],
            top_feature_count=1,
        )


def test_calculate_jaccard_similarity_returns_expected_score() -> None:
    result = calculate_jaccard_similarity(
        first_feature_set={"amount", "velocity", "recency"},
        second_feature_set={"velocity", "recency", "device_risk"},
    )

    assert result == pytest.approx(0.5)


def test_calculate_jaccard_similarity_rejects_empty_feature_set() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        calculate_jaccard_similarity(
            first_feature_set=set(),
            second_feature_set={"amount"},
        )


def test_evaluate_shap_stability_returns_pairwise_case_scores() -> None:
    feature_names = ["amount", "velocity", "recency", "device_risk"]
    case_group_names = ["true_positive", "false_positive"]

    shap_values_by_seed = {
        42: np.array(
            [
                [0.80, -0.70, 0.10, 0.05],
                [0.10, 0.20, -0.90, 0.80],
            ]
        ),
        52: np.array(
            [
                [0.75, -0.65, 0.15, 0.10],
                [0.05, 0.30, -0.85, 0.70],
            ]
        ),
        62: np.array(
            [
                [0.70, -0.60, 0.20, 0.10],
                [0.15, 0.25, -0.80, 0.75],
            ]
        ),
    }

    result = evaluate_shap_stability(
        shap_values_by_seed=shap_values_by_seed,
        feature_names=feature_names,
        case_group_names=case_group_names,
        top_feature_count=2,
    )

    assert len(result) == 6
    assert set(result["case_group"]) == {
        "true_positive",
        "false_positive",
    }
    assert set(result["jaccard_similarity"]) == {1.0}
    assert set(result["shared_top_feature_count"]) == {2}
    assert set(result["union_top_feature_count"]) == {2}


def test_evaluate_shap_stability_detects_changed_top_features() -> None:
    result = evaluate_shap_stability(
        shap_values_by_seed={
            42: np.array([[0.90, 0.80, 0.10, 0.05]]),
            52: np.array([[0.10, 0.05, 0.90, 0.80]]),
        },
        feature_names=["amount", "velocity", "recency", "device_risk"],
        case_group_names=["false_negative"],
        top_feature_count=2,
    )

    assert len(result) == 1
    assert result.loc[0, "jaccard_similarity"] == 0.0
    assert result.loc[0, "shared_top_feature_count"] == 0
    assert result.loc[0, "union_top_feature_count"] == 4
    assert result.loc[0, "shared_top_features"] == ""


def test_evaluate_shap_stability_requires_two_background_seeds() -> None:
    with pytest.raises(ValueError, match="At least two"):
        evaluate_shap_stability(
            shap_values_by_seed={
                42: np.array([[0.80, -0.70]]),
            },
            feature_names=["amount", "velocity"],
            case_group_names=["true_positive"],
            top_feature_count=1,
        )


def test_summarise_shap_stability_calculates_case_level_statistics() -> None:
    pairwise_stability_table = pd.DataFrame(
        {
            "case_group": [
                "false_negative",
                "false_negative",
                "true_positive",
            ],
            "jaccard_similarity": [0.50, 1.00, 0.75],
            "shared_top_feature_count": [2, 3, 3],
        }
    )

    result = summarise_shap_stability(pairwise_stability_table)

    false_negative_row = result.loc[result["case_group"] == "false_negative"].iloc[0]

    assert false_negative_row["seed_pair_count"] == 2
    assert false_negative_row["mean_jaccard_similarity"] == pytest.approx(0.75)
    assert false_negative_row["min_jaccard_similarity"] == pytest.approx(0.50)
    assert false_negative_row["max_jaccard_similarity"] == pytest.approx(1.00)
    assert false_negative_row["mean_shared_top_feature_count"] == pytest.approx(2.5)


def test_summarise_shap_stability_rejects_empty_table() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        summarise_shap_stability(pd.DataFrame())
