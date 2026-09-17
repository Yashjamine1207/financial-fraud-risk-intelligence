# Phase 4 Model Comparison

## Scope

This report compares the Phase 4 Logistic Regression baseline and XGBoost
tabular model using only the chronological validation split.

The final test split was not loaded by either experiment and remains locked for
later final evaluation.

## Shared evaluation context

| Item | Value |
|---|---:|
| Dataset version | ieee-cis-v1 |
| Feature version | point-in-time-v1.0.0 |
| Target-encoding version | time-safe-te-v1.0.0 |
| Validation review capacity | 1,000 transactions |
| Validation fraud cases | 3,042 |
| Test split loaded | False |

## Validation metrics

| Metric | Logistic Regression | XGBoost | Difference: XGBoost − Logistic Regression |
|---|---:|---:|---:|
| PR-AUC | 0.393668 | 0.539280 | 0.145613 |
| ROC-AUC | 0.842060 | 0.911347 | 0.069287 |
| Precision@1,000 | 0.711000 | 0.864000 | 0.153000 |
| Recall@1,000 | 0.233728 | 0.284024 | 0.050296 |
| Fraud captured at 1,000 reviews | 711 | 864 | +153 |
| Precision at 0.50 reference threshold | 0.111211 | 0.254928 | 0.143717 |
| Recall at 0.50 reference threshold | 0.731755 | 0.697239 | -0.034517 |
| F1 at 0.50 reference threshold | 0.193078 | 0.373350 | 0.180271 |
| Prediction latency per row (ms) | 0.048965 | 0.003354 | -0.045611 |

## Provisional validation champion

**xgboost**

Reason: XGBoost has the higher validation PR-AUC.

At the fixed capacity of 1,000 manual reviews, XGBoost captured
153 more known fraud cases than Logistic Regression.

## Decision boundary

This is a validation-based model-selection decision only.

It does not represent:
- final test-set performance;
- calibrated fraud probabilities;
- an approved production threshold;
- an estimated financial-cost outcome.

Phase 5 will calibrate the selected model and optimise the approve, review, and
block decision policy using documented cost assumptions.
