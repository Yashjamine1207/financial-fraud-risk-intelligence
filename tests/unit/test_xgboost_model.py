"""Tests for the Phase 4 XGBoost fraud model."""

import numpy as np
import pytest

from fraud_intelligence.models.xgboost_model import (
    build_xgboost_classifier,
    calculate_scale_pos_weight,
)


def _build_model_config() -> dict:
    """Return a complete XGBoost configuration for testing."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": "cpu",
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 5.0,
        "gamma": 0.0,
        "scale_pos_weight": "auto",
        "random_state": 42,
        "n_jobs": -1,
    }


def test_calculate_scale_pos_weight_uses_training_class_ratio() -> None:
    """Fraud-class weighting must equal non-fraud rows divided by fraud rows."""
    y_train = np.array([0, 0, 0, 0, 1, 1])

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    assert scale_pos_weight == 2.0


def test_calculate_scale_pos_weight_rejects_single_class_training_data() -> None:
    """A training set with no fraud records must fail clearly."""
    with pytest.raises(ValueError, match="no fraud transactions"):
        calculate_scale_pos_weight(np.array([0, 0, 0]))


def test_build_xgboost_classifier_uses_calculated_class_weight() -> None:
    """The classifier must use the supplied training-only class weight."""
    classifier = build_xgboost_classifier(
        model_config=_build_model_config(),
        scale_pos_weight=27.4,
    )

    parameters = classifier.get_params()

    assert parameters["objective"] == "binary:logistic"
    assert parameters["eval_metric"] == "aucpr"
    assert parameters["tree_method"] == "hist"
    assert parameters["scale_pos_weight"] == 27.4
    assert parameters["random_state"] == 42
