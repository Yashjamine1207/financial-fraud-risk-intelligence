# Phase 6 - Time-Safe Graph-Feature Ablation Decision

## Decision

**REJECT** time-safe graph features.

The graph features did not improve the selected operational decision objective at the configured review capacity.

The graph candidate is rejected and will not be evaluated on the locked final test split. The selected system remains the existing Phase 5 baseline XGBoost while the compact LSTM/GRU sequence ablation is evaluated.

## Comparable evaluation protocol

- Baseline model: `xgboost-v1.0.0`
- Candidate model: `xgboost-plus-graph-v1.0.0`
- Baseline MLflow run: `a1f99cff34cf44728d3b6c9499ffeee3`
- Candidate MLflow run: `83af368ab0ed46d2b6d7029558491e23`
- Graph feature version: `graph-features-v1.0.0`
- Graph feature count: 10
- Calibration method: sigmoid / Platt scaling
- Calibration-fit period: earliest chronological 50% of validation
- Policy-selection period: latest chronological 50% of validation
- Timestamp boundary: `11725712`
- Policy-selection transactions: 44,291
- Policy-selection known fraud: 1,449
- Final test split loaded: False

## Operational decision result

The configured operational capacity is **1,000 reviews**.

| Metric | Baseline | XGBoost + graph features | Candidate minus baseline |
|---|---:|---:|---:|
| Fraud captured | 568 | 564 | -4 |
| Fraud capture rate | 39.20% | 38.92% | -0.28% |
| Net expected value | GBP 279,000 | GBP 277,000 | GBP -2,000 |

## Capacity sensitivity

| Review capacity | Baseline captured fraud | Graph captured fraud | Difference | Baseline net value | Graph net value | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 392 | 389 | -3 | GBP 193,500 | GBP 192,000 | GBP -1,500 |
| 1,000 | 568 | 564 | -4 | GBP 279,000 | GBP 277,000 | GBP -2,000 |
| 1,500 | 655 | 671 | +16 | GBP 320,000 | GBP 328,000 | GBP +8,000 |
| 2,000 | 731 | 743 | +12 | GBP 355,500 | GBP 361,500 | GBP +6,000 |

## Interpretation

Graph features represent historical entity-relationship structure. They contributed to model ranking but do not prove that a card, device, email domain, relationship, or graph component caused fraud.

All financial figures use illustrative portfolio-project assumptions and do not represent real financial-institution outcomes or realised savings.
