# Phase 6 - Advanced Signal Ablation Summary

## Objective

Phase 6 tested whether advanced fraud-detection signals improved the selected
Phase 5 calibrated XGBoost decision system under the same leakage-safe
chronological validation protocol, sigmoid calibration approach, illustrative
cost assumptions, and constrained investigation-review capacity.

The configured operational capacity was 1,000 transaction reviews.

## Final decision

The Phase 5 calibrated XGBoost model remains the selected champion.

All Phase 6 advanced candidates were rejected because they did not improve the
documented fraud-prioritisation objective at the configured operational review
capacity.

The final test split was not loaded for any rejected Phase 6 candidate.

## Experiment outcomes

| Experiment | Candidate | Outcome | Decision |
|---|---|---|---|
| Anomaly ablation | XGBoost + Isolation Forest anomaly score | Did not improve the selected policy objective at 1,000 reviews | Reject |
| Graph ablation | XGBoost + 10 time-safe graph features | -4 fraud captured and GBP -2,000 net expected value at 1,000 reviews | Reject |
| Sequence ablation | Compact GRU using card1 transaction histories | 146 raw validation fraud captures at 1,000 reviews versus 864 for baseline XGBoost | Reject |
| Sequence ablation | Compact LSTM using card1 transaction histories | Best recurrent candidate, but -447 calibrated-policy fraud captures and GBP -223,500 net expected value at 1,000 reviews | Reject |

## Shared evaluation controls

Every Phase 6 candidate followed these controls:

- Chronological training and validation data only.
- The final test split remained locked during advanced-method selection.
- The same Phase 4 / Phase 5 XGBoost baseline and validation comparison logic.
- The same configured 1,000-review operational capacity.
- The same illustrative cost assumptions.
- The same sigmoid / Platt calibration protocol for policy candidates.
- Calibration fitted on the earliest chronological validation period.
- Policy comparison evaluated on the later chronological validation period.
- No randomly shuffled final evaluation data.

## Anomaly ablation

The Isolation Forest candidate added one anomaly score to the existing XGBoost
feature contract.

Leakage controls:

- Isolation Forest used no `isFraud` labels.
- Training anomaly scores used chronological out-of-fold scoring.
- Validation anomaly scores used a model fitted only on chronological training
  data.
- Early-history missing scores were handled by training-fitted preprocessing.

Result:

The anomaly score did not improve the selected operational objective at the
configured 1,000-review capacity and was rejected.

See:

```text
docs/experiments/phase_6_anomaly_ablation_decision.md
reports/tables/phase6_anomaly_policy_comparison.csv
```

## Graph-feature ablation

The graph candidate added ten features built from historical card-device and
card-recipient-email relationships.

Leakage controls:

- A transaction used graph state from strictly earlier timestamps only.
- Same-timestamp transactions were scored before graph state updated.
- Validation graph state began with training graph history and then updated
  only from earlier validation transactions.
- Missing device and recipient-email values did not create shared graph nodes.
- Fraud labels were not used in graph construction or graph features.

Preliminary ranking result:

```text
Baseline XGBoost fraud captured at 1,000 reviews: 864
XGBoost plus graph features fraud captured at 1,000 reviews: 868
```

Final calibrated-policy result:

```text
Fraud captured difference at 1,000 reviews: -4
Net expected value difference at 1,000 reviews: GBP -2,000
Decision: Reject
```

The graph candidate was not evaluated on the final test split.

See:

```text
docs/experiments/phase_6_graph_ablation_decision.md
reports/tables/phase6_graph_ablation_comparison.csv
reports/tables/phase6_graph_policy_comparison.csv
reports/figures/phase6_graph_validation_policy_evaluation.png
```

## Sequence-model ablation

The sequence experiment used compact TensorFlow/Keras GRU and LSTM models.

Sequence definition:

- Entity history key: `card1`.
- Maximum sequence length: 10 transactions.
- Per-event inputs: log transaction amount and log time since the previous
  available card1 transaction.
- The current transaction was included using fields available at scoring time.
- Earlier sequence positions contained only strictly prior card1 events.
- Same-timestamp transactions could not enter one another's histories.
- Validation sequences used training history plus earlier validation history.
- Fraud labels were not sequence inputs.

### GRU result

```text
PR-AUC: 0.064503
ROC-AUC: 0.589980
Fraud captured at 1,000 raw validation reviews: 146 / 3,042
Decision: Reject
```

### LSTM result

```text
PR-AUC: 0.069302
ROC-AUC: 0.607412
Fraud captured at 1,000 raw validation reviews: 153 / 3,042
```

The LSTM was the better recurrent candidate, so it advanced to the formal
calibrated policy comparison.

```text
Best sequence candidate fraud-capture difference at 1,000 reviews: -447
Best sequence candidate net expected value difference: GBP -223,500
Decision: Reject GRU and LSTM
```

Neither recurrent model was evaluated on the locked final test split.

See:

```text
docs/experiments/phase_6_sequence_ablation_decision.md
reports/tables/phase6_sequence_model_comparison.csv
reports/tables/phase6_sequence_policy_comparison.csv
reports/figures/phase6_lstm_sequence_validation_policy_evaluation.png
```

## Interpretation

The Phase 6 result does not show that anomaly detection, graph methods, GRUs,
or LSTMs are generally ineffective for fraud detection.

It shows that, for this IEEE-CIS benchmark, this feature representation, this
chronological split, and the documented 1,000-review policy, none of the tested
advanced candidates delivered incremental operational value beyond the existing
leakage-safe XGBoost model and its engineered behavioural features.

The project therefore retains the simpler calibrated XGBoost system. This is a
valid and preferable portfolio result because model selection follows measured
decision value rather than model complexity.

All financial values in this project use illustrative benchmark assumptions.
They do not represent real financial-institution outcomes, realised savings,
or production fraud losses.