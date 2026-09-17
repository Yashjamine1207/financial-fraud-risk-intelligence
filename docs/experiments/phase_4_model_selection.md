# Phase 4 Model Selection

## Objective

Phase 4 compared an interpretable Logistic Regression baseline with an XGBoost
tabular model for fraud-risk ranking.

The objective was not to maximise accuracy or choose a default probability
threshold. The comparison prioritised PR-AUC, Recall@k, Precision@k, fraud
capture at the documented analyst review capacity, and prediction latency.

## Data and leakage controls

| Item | Value |
|---|---|
| Primary dataset | IEEE-CIS Fraud Detection |
| Training rows | 413,378 |
| Validation rows | 88,581 |
| Final test rows | 88,581 |
| Temporal split | Chronological 70% / 15% / 15% |
| Dataset version | ieee-cis-v1 |
| Feature version | point-in-time-v1.0.0 |
| Target-encoding version | time-safe-te-v1.0.0 |
| Shared model-feature count | 461 |
| Numeric features | 432 |
| Categorical features | 29 |
| Final test split loaded during Phase 4 | No |

Both models used the same Phase 3 train and validation feature tables and the
same leakage-safe feature contract.

The final test split was not loaded for preprocessing fitting, model fitting,
hyperparameter selection, early stopping, model comparison, or threshold
selection.

## Preprocessing

The preprocessing pipeline was fitted only on the chronological training data.

Numeric features used:

- Median imputation
- Missingness indicators
- Standard scaling

Categorical features used:

- Most-frequent-value imputation
- One-hot encoding
- Unknown-category handling during validation scoring

The pipeline excluded `TransactionID`, `TransactionDT`, `isFraud`, and the nine
columns identified in Phase 2 with more than 95% missing values.

## Models compared

### Logistic Regression baseline

The baseline used class-weighted, L2-regularised Logistic Regression.

| Parameter | Value |
|---|---:|
| Model version | logistic-regression-v1.0.0 |
| Solver | saga |
| Regularisation strength, C | 0.1 |
| L1 ratio | 0.0 |
| Maximum iterations | 1,000 |
| Convergence tolerance | 0.005 |
| Class weighting | balanced |
| Random seed | 42 |
| MLflow run ID | c04505eb870047ab9a699c71f0ad73c4 |

### XGBoost tabular model

The stronger tabular model used XGBoost with histogram-based CPU tree building
and training-only class-imbalance weighting.

| Parameter | Value |
|---|---:|
| Model version | xgboost-v1.0.0 |
| Objective | binary:logistic |
| Validation metric for early stopping | aucpr |
| Tree method | hist |
| Device | cpu |
| Maximum estimators | 1,000 |
| Early stopping rounds | 50 |
| Learning rate | 0.05 |
| Maximum tree depth | 6 |
| Minimum child weight | 5 |
| Subsample | 0.80 |
| Column subsample by tree | 0.80 |
| L1 regularisation | 0.10 |
| L2 regularisation | 5.0 |
| Class weight | Training-only non-fraud to fraud ratio |
| Best boosting iteration | 499 |
| MLflow run ID | 3976b38b62c441bca56866914ac301d9 |

## Validation results

The review capacity is fixed at 1,000 transactions per day, based on the
illustrative portfolio decision-policy assumptions defined in Phase 0.

| Metric | Logistic Regression | XGBoost | Difference: XGBoost − Logistic Regression |
|---|---:|---:|---:|
| PR-AUC | 0.393668 | 0.539280 | +0.145612 |
| ROC-AUC | 0.842060 | 0.911347 | +0.069287 |
| Precision at 0.50 reference threshold | 0.111211 | 0.254928 | +0.143717 |
| Recall at 0.50 reference threshold | 0.731755 | 0.697239 | -0.034516 |
| F1 at 0.50 reference threshold | 0.193078 | 0.373350 | +0.180272 |
| Precision@1,000 | 0.711000 | 0.864000 | +0.153000 |
| Recall@1,000 | 0.233728 | 0.284024 | +0.050296 |
| Fraud captured at 1,000 reviews | 711 | 864 | +153 |
| Total known fraud in validation | 3,042 | 3,042 | 0 |
| Prediction latency per row | 0.048965 ms | 0.003354 ms | -0.045611 ms |
| Training time | 205.57 seconds | 660.83 seconds | +455.26 seconds |

## Model-selection decision

**Provisional selected model: XGBoost v1.0.0**

XGBoost is selected for Phase 5 because it has the higher validation PR-AUC
and performs better under the documented review-capacity constraint.

At a capacity of 1,000 manual reviews, XGBoost:

- Captured 864 known fraud cases, compared with 711 for Logistic Regression
- Captured 153 additional fraud cases
- Increased Precision@1,000 from 71.1% to 86.4%
- Increased Recall@1,000 from 23.37% to 28.40%
- Produced lower mean validation prediction latency per transaction

The selection is based on the chronological validation period only.

## Important limitations

- The probabilities are not calibrated yet. Raw XGBoost probabilities must not
  be interpreted as final fraud-risk probabilities.
- The 0.50 threshold is a diagnostic reference point only. It is not a final
  approve, review, or block threshold.
- No cost-sensitive decision policy has been selected yet.
- The final chronological test period remains locked and must not be used until
  the calibration method and cost-sensitive policy have been selected.
- These results are benchmark results from an anonymised public dataset, not
  real financial-institution outcomes.

## Phase 5 handover

Phase 5 will:

1. Compare uncalibrated XGBoost probabilities with Platt scaling and isotonic
   calibration using a leakage-safe validation process
2. Measure Brier score, expected calibration error, and calibration curves
3. Apply the documented false-negative cost, review cost, fraud-prevention
   value, and review-capacity assumptions
4. Select approve, review, and block actions using expected financial cost
5. Keep the final test split locked until the calibration and policy choices
   have been fixed