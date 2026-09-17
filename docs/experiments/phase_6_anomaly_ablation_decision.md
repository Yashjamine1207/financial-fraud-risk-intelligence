# Phase 6 — Isolation Forest Anomaly Ablation Decision

## Decision

**REJECT** `anomaly_score_isolation_forest`.

The anomaly feature did not improve the selected operational decision objective at the configured review capacity.

The candidate will not be evaluated on the locked final test split.
The selected system remains the existing Phase 5 baseline XGBoost
with sigmoid calibration and the documented top-k review policy.

## Comparable evaluation protocol

- Baseline model: `xgboost-v1.0.0`
- Candidate model: `xgboost-plus-anomaly-v1.0.0`
- Baseline MLflow run: `a1f99cff34cf44728d3b6c9499ffeee3`
- Candidate MLflow run: `07dde90db2e34d72bf0db108e2038cb1`
- Calibration method: sigmoid / Platt scaling
- Calibration-fit period: earliest chronological 50% of validation
- Policy-selection period: latest chronological 50% of validation
- Timestamp boundary: `11725712`
- Policy-selection transactions: 44,291
- Policy-selection known fraud: 1,449
- Final test split loaded: False

## Operational decision result

The configured operational capacity is **1,000 reviews**.

| Metric | Baseline | XGBoost + anomaly | Candidate minus baseline |
|---|---:|---:|---:|
| Fraud captured | 568 | 567 | -1 |
| Fraud capture rate | 39.20% | 39.13% | -0.07% |
| Net expected value | £279,000 | £278,500 | £-500 |

## Capacity sensitivity

| Review capacity | Baseline captured fraud | Anomaly captured fraud | Difference | Baseline net value | Anomaly net value | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 392 | 392 | +0 | £193,500 | £193,500 | £+0 |
| 1,000 | 568 | 567 | -1 | £279,000 | £278,500 | £-500 |
| 1,500 | 655 | 669 | +14 | £320,000 | £327,000 | £+7,000 |
| 2,000 | 731 | 744 | +13 | £355,500 | £362,000 | £+6,500 |

## Interpretation

The anomaly feature changed the ranking slightly, but it did not improve the project’s selected operational policy at the documented capacity of 1,000 reviews.

Some larger review capacities may show a positive candidate difference. That does not justify retaining the feature for the selected policy because operational capacity is fixed at 1,000 reviews under the versioned portfolio assumptions.

## Cost assumptions

- False-negative cost: £500
- Manual-review cost: £5
- Fraud-prevention value: £500

All financial assumptions are illustrative portfolio-project assumptions and are not real financial-institution outcomes.
