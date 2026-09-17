"""Integrate anomaly score feature into modelling datasets.

This module merges the precomputed anomaly scores into train/validation/test
feature tables, producing augmented feature DataFrames ready for modelling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_anomaly_scores() -> tuple[pd.Series, pd.Series, pd.Series | None]:
    """Load anomaly scores for train, validation, and optional test.

    Returns:
        train_scores, val_scores, test_scores (test_scores may be None)
    """
    base = Path("data/features/anomaly")
    train_scores = pd.read_parquet(base / "train_anomaly_scores.parquet")[
        "anomaly_score_isolation_forest"
    ]
    val_scores = pd.read_parquet(base / "val_anomaly_scores.parquet")[
        "anomaly_score_isolation_forest"
    ]

    test_scores: pd.Series | None = None
    test_path = base / "test_anomaly_scores.parquet"
    if test_path.exists():
        test_scores = pd.read_parquet(test_path)["anomaly_score_isolation_forest"]

    return train_scores, val_scores, test_scores


def augment_with_anomaly_score(
    X: pd.DataFrame,
    scores: pd.Series,
) -> pd.DataFrame:
    """Augment a feature DataFrame with the anomaly score.

    Args:
        X: Original feature DataFrame (without target/ids).
        scores: Anomaly score Series aligned to X.index.

    Returns:
        Augmented DataFrame with the anomaly score column added.
    """
    if len(X) != len(scores):
        raise ValueError("X and scores must have the same length.")
    if not scores.name:
        raise ValueError("scores Series must have a name.")

    X_aug = X.copy()
    X_aug[scores.name] = scores.reset_index(drop=True)
    return X_aug
