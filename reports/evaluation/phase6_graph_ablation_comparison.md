# Phase 6 - Time-Safe Graph-Feature Ablation

## Objective

Compare the Phase 4 XGBoost baseline with the same XGBoost model plus
ten time-safe graph features derived from historical card-device and
card-recipient-email relationships.

The purpose is to test whether historical relationship structure adds
measurable fraud-prioritisation value beyond the existing leakage-safe
point-in-time behavioural features.

## Experimental protocol

- Baseline model: `xgboost-v1.0.0`
- Candidate model: `xgboost-plus-graph-v1.0.0`
- Graph feature version: `graph-features-v1.0.0`
- Added graph features: 10
- Dataset version: `ieee-cis-v1`
- Base feature version: `point-in-time-v1.0.0`
- Target-encoding version: `time-safe-te-v1.0.0`
- Validation review capacity: 1,000 transactions
- Validation known fraud count: 3,042
- Evaluation split: chronological validation data only
- Final test split loaded: False

## Leakage controls

- Graph features use only earlier transaction timestamps.
- All same-timestamp transactions are scored before graph state updates.
- Validation graph state begins with training history only.
- Validation graph state then updates only from earlier validation timestamps.
- Missing device and recipient-email values do not form shared graph nodes.
- Fraud labels are never used in graph construction or graph features.
- Numeric preprocessing and missingness handling are fitted on training data only.

## Added graph features

| Feature family | Features |
|---|---|
| Historical graph degree | `graph_card1_prior_degree`, `graph_deviceinfo_prior_card_degree`, `graph_remail_prior_card_degree` |
| Historical edge frequency | `graph_card1_device_prior_edge_count`, `graph_card1_remail_prior_edge_count` |
| Historical component size | `graph_card1_prior_component_size`, `graph_deviceinfo_prior_component_size`, `graph_remail_prior_component_size` |
| New relationship indicators | `is_new_card1_device_relationship`, `is_new_card1_remail_relationship` |

## Validation comparison

| Metric | Baseline XGBoost | XGBoost + graph features | Candidate minus baseline |
|---|---:|---:|---:|
| PR-AUC | 0.539280 | 0.547304 | +0.008023 |
| ROC-AUC | 0.911347 | 0.913321 | +0.001974 |
| Precision at reference threshold | 0.254928 | 0.269547 | +0.014619 |
| Recall at reference threshold | 0.697239 | 0.701512 | +0.004274 |
| F1 at reference threshold | 0.373350 | 0.389452 | +0.016102 |
| Precision@1,000 | 0.864000 | 0.868000 | +0.004000 |
| Recall@1,000 | 0.284024 | 0.285339 | +0.001315 |
| Fraud captured at 1,000 reviews | 864 | 868 | +4 |
| Total known fraud in validation | 3042 | 3042 | +0 |
| Prediction latency per row (ms) | 0.003354 | 0.004655 | +0.001301 |
| Training time (seconds) | 660.83 | 676.18 | +15.35 |
| Best boosting iteration | 499 | 499 | +0 |

## Preliminary interpretation

At the fixed review capacity of 1,000 transactions, the candidate captured 868 known fraud cases compared with 864 for the baseline (+4 cases).

This is a preliminary ranking comparison only. It is not the final keep/reject decision because the graph candidate must still undergo the same calibration and capacity-constrained policy evaluation used by the Phase 5 baseline.

## Next evaluation requirement

Fit sigmoid calibration on only the earlier chronological validation calibration-fit period. Compare the baseline and graph candidate on the later policy-selection period using the same capacities and cost assumptions. Do not use the locked final test split during this decision.
