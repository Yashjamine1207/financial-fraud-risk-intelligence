# Phase 6 - Compact LSTM and GRU Sequence Ablation Decision

## Decision

**REJECT** both compact GRU and compact LSTM sequence models.

The better recurrent candidate, the compact LSTM, failed to improve fraud capture or illustrative net expected value at the configured operational capacity of 1,000 reviews.

The compact GRU was also rejected. It was inferior to the LSTM in validation PR-AUC and fraud captured at 1,000 reviews, so it was not advanced to a redundant calibrated-policy evaluation.

Neither sequence candidate will be evaluated on the locked final test split. The Phase 5 calibrated XGBoost baseline remains the selected champion.

## Sequence protocol

- Framework: TensorFlow / Keras
- Entity sequence key: `card1`
- Maximum sequence length: 10 transactions
- Per-event inputs: log transaction amount and log prior card time gap
- Same-timestamp rule: score all tied transactions before history updates
- Training history: strictly earlier chronological card1 transactions
- Validation history: training history plus earlier validation transactions
- Fraud labels used as sequence inputs: False
- Final test split loaded: False

## Preliminary architecture comparison

| Metric | Compact GRU | Compact LSTM | LSTM minus GRU |
|---|---:|---:|---:|
| PR-AUC | 0.064503 | 0.069302 | +0.004799 |
| ROC-AUC | 0.589980 | 0.607412 | +0.017433 |
| Precision@1,000 | 0.146000 | 0.153000 | +0.007000 |
| Recall@1,000 | 0.047995 | 0.050296 | +0.002301 |
| Fraud captured at 1,000 reviews | 146 | 153 | +7 |
| Training time (seconds) | 66.73 | 62.61 | -4.12 |
| Epochs completed | 9 | 8 | -1 |

## Comparable calibrated policy result

- Baseline model: `xgboost-v1.0.0`
- LSTM candidate: `compact-sequence-lstm-v1.0.0`
- Calibration: sigmoid / Platt scaling
- Calibration period: earliest chronological 50% of validation
- Policy period: latest chronological 50% of validation
- Timestamp boundary: `11725712`
- Policy-selection transactions: 44,291
- Policy-selection known fraud: 1,449

The operating capacity is **1,000 reviews**.

| Metric | Baseline XGBoost | Compact LSTM | LSTM minus baseline |
|---|---:|---:|---:|
| Fraud captured | 568 | 121 | -447 |
| Fraud capture rate | 39.20% | 8.35% | -30.85% |
| Net expected value | GBP 279,000 | GBP 55,500 | GBP -223,500 |

## Capacity sensitivity

| Capacity | Baseline captured | LSTM captured | Difference | Baseline net value | LSTM net value | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 392 | 68 | -324 | GBP 193,500 | GBP 31,500 | GBP -162,000 |
| 1,000 | 568 | 121 | -447 | GBP 279,000 | GBP 55,500 | GBP -223,500 |
| 1,500 | 655 | 157 | -498 | GBP 320,000 | GBP 71,000 | GBP -249,000 |
| 2,000 | 731 | 193 | -538 | GBP 355,500 | GBP 86,500 | GBP -269,000 |

## Interpretation

The compact recurrent models did not provide sufficient incremental fraud-prioritisation value relative to the existing tabular XGBoost model and its engineered point-in-time behavioural features.

This result does not imply that sequence models are generally ineffective for fraud detection. It applies to this compact, two-input, card1-history experiment under this chronological IEEE-CIS benchmark and documented policy assumptions.

All financial values use illustrative portfolio assumptions and do not represent real financial-institution savings.
