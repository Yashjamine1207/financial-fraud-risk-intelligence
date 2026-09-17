# Phase 6 — Isolation Forest Anomaly Ablation

## Objective

Compare the Phase 4 XGBoost baseline with the same XGBoost model plus
`anomaly_score_isolation_forest`.

The purpose is to test whether an unsupervised anomaly signal adds
measurable fraud-prioritisation value beyond the existing leakage-safe
point-in-time behavioural features.

## Experimental protocol

- Baseline model: `xgboost-v1.0.0`
- Candidate model: `xgboost-plus-anomaly-v1.0.0`
- Added candidate feature: `anomaly_score_isolation_forest`
- Dataset version: `ieee-cis-v1`
- Base feature version: `point-in-time-v1.0.0`
- Target-encoding version: `time-safe-te-v1.0.0`
- Validation review capacity: 1,000 transactions
- Validation known fraud count: 3,042
- Evaluation split: chronological validation data only
- Final test split loaded: False

## Leakage controls

- Isolation Forest was trained without `isFraud` labels.
- Training anomaly scores were generated using chronological out-of-fold scoring.
- Each training scoring fold used an Isolation Forest fitted only on earlier training folds.
- Validation anomaly scores were produced with an Isolation Forest fitted on the full chronological training period only.
- The earliest training fold has no earlier history and therefore has intentionally missing anomaly scores.
- The existing training-fitted numeric preprocessing pipeline handles those missing values using median imputation and missingness indicators.

## Validation comparison

| Metric | Baseline XGBoost | XGBoost + anomaly | Candidate minus baseline |
|---|---:|---:|---:|
| PR-AUC | 0.539280 | 0.539522 | +0.000242 |
| ROC-AUC | 0.911347 | 0.909860 | -0.001487 |
| Precision at reference threshold | 0.254928 | 0.256432 | +0.001504 |
| Recall at reference threshold | 0.697239 | 0.691321 | -0.005917 |
| F1 at reference threshold | 0.373350 | 0.374099 | +0.000750 |
| Precision@1,000 | 0.864000 | 0.867000 | +0.003000 |
| Recall@1,000 | 0.284024 | 0.285010 | +0.000986 |
| Fraud captured at 1,000 reviews | 864 | 867 | +3 |
| Total known fraud in validation | 3042 | 3042 | +0 |
| Prediction latency per row (ms) | 0.002959 | 0.003094 | +0.000135 |
| Training time (seconds) | 626.86 | 628.11 | +1.26 |
| Best boosting iteration | 499 | 498 | -1 |

## Preliminary interpretation

At the fixed review capacity of 1,000 transactions, the candidate captured 867 known fraud cases compared with 864 for the baseline (+3 cases).

This is a preliminary ranking comparison only. It is not yet the final keep/reject decision because the candidate still needs the same calibration and capacity-constrained policy evaluation used by the Phase 5 model.

## Next evaluation requirement

Fit the candidate calibration mapping using only the earlier Phase 5 validation calibration-fit period. Then compare the baseline and candidate on the later chronological policy-selection period at the same review capacities and cost assumptions. Do not use the final test split during this decision.
