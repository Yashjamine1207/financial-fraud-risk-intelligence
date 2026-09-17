# Phase 6 — Isolation Forest Anomaly Ablation

## Objective

Test whether an unsupervised Isolation Forest anomaly-score feature improves fraud
prioritisation beyond the selected Phase 5 XGBoost baseline.

The experiment compares:

1. **Baseline:** Phase 5 XGBoost with leakage-safe Phase 3 point-in-time
   behavioural features and sigmoid calibration.

2. **Candidate:** The same XGBoost training configuration and feature contract,
   with one additional feature:

   ```text
   anomaly_score_isolation_forest
   ```

The anomaly feature is retained only if it improves the documented operational
decision objective at the configured investigation capacity without reducing
reliability or decision value.

---

## Baseline system

| Item | Value |
|---|---|
| Base model | XGBoost |
| Base model version | `xgboost-v1.0.0` |
| Phase 4 frozen MLflow model run | `a1f99cff34cf44728d3b6c9499ffeee3` |
| Dataset version | `ieee-cis-v1` |
| Behavioural feature version | `point-in-time-v1.0.0` |
| Target-encoding version | `time-safe-te-v1.0.0` |
| Selected calibration method | Sigmoid / Platt scaling |
| Calibration version | `calibration-v1.0.0` |
| Threshold-policy version | `threshold-policy-v1.0.0` |
| Cost-policy version | `policy-v0.1.0` |
| Operational manual-review capacity | 1,000 transactions |

---

## Candidate anomaly feature

| Item | Value |
|---|---|
| Feature name | `anomaly_score_isolation_forest` |
| Feature version | `isolation-forest-v1.0.0` |
| Model | Isolation Forest |
| Number of estimators | 200 |
| Contamination | 0.035 |
| Maximum samples per estimator | 256 |
| Random seed | 42 |
| Parallel jobs | All available CPU cores |
| Candidate XGBoost version | `xgboost-plus-anomaly-v1.0.0` |
| Candidate MLflow training run | `07dde90db2e34d72bf0db108e2038cb1` |
| Candidate policy MLflow run | `8bdd5028d13042ffb2c6a99e62363426` |

Isolation Forest is unsupervised. It does not use `isFraud` labels during fitting
or scoring.

A larger anomaly score represents a transaction that Isolation Forest considers
more unusual relative to the historical training data. This score is a model
input only. It is not evidence that a transaction is fraudulent.

---

## Leakage-safe design

### Training anomaly scores

Training anomaly scores were generated through chronological out-of-fold scoring.

The chronological training period was divided into five timestamp-safe folds.
Transactions sharing the same `TransactionDT` were always assigned to the same
fold.

For each usable scoring fold:

- Isolation Forest was fitted only on earlier chronological training folds.
- The fitted model then scored the current fold.
- No later rows were used when fitting the anomaly model.
- No same-timestamp rows were split between the fit and scoring datasets.
- `isFraud` was not supplied to Isolation Forest.

The first chronological fold had no earlier transactions available for fitting.
Its anomaly scores were intentionally recorded as missing.

| Training anomaly-score item | Result |
|---|---:|
| Total training rows | 413,378 |
| Earliest-fold rows with no prior anomaly-model history | 83,061 |
| Chronological anomaly folds | 5 |
| Fraud labels used by Isolation Forest | No |

The existing Phase 4 numeric preprocessing pipeline handles these intentionally
missing early-history values using training-fitted median imputation and a
missingness indicator.

### Validation anomaly scores

Validation anomaly scores were generated using an Isolation Forest fitted on the
complete chronological training period only.

Validation labels were not used by Isolation Forest.

| Validation anomaly-score item | Result |
|---|---:|
| Validation rows scored | 88,581 |
| Isolation Forest fitting data | Training period only |
| Validation fraud labels used in anomaly scoring | No |

### Final test protection

A final-test anomaly score file was generated only after the Phase 5 model,
calibration method, policy assumptions, and operational capacity were already
locked.

However, because the anomaly candidate was rejected during validation-based
policy evaluation, its final-test score file was not used for model selection,
calibration selection, policy selection, or holdout reporting.

| Data-protection rule | Status |
|---|---|
| Final test used to fit Isolation Forest | No |
| Final test labels used by Isolation Forest | No |
| Final test used for anomaly-model selection | No |
| Final test used for anomaly policy evaluation | No |
| Final test used after anomaly rejection | No |

---

## Training comparison

Both models used:

- The same chronological Phase 3 training and validation feature tables.
- The same Phase 4 XGBoost configuration.
- The same training-only class-weight calculation.
- The same training-fitted preprocessing approach.
- The same validation review capacity of 1,000 transactions.
- The same fixed reference threshold of 0.50 for diagnostic metrics only.
- No final test data.

| Metric | Phase 4 XGBoost baseline | XGBoost + anomaly | Candidate minus baseline |
|---|---:|---:|---:|
| PR-AUC | 0.539280 | 0.539522 | +0.000242 |
| ROC-AUC | 0.911347 | 0.909860 | -0.001487 |
| Precision at reference threshold | 0.254928 | 0.256432 | +0.001504 |
| Recall at reference threshold | 0.697239 | 0.691321 | -0.005918 |
| F1 at reference threshold | 0.373350 | 0.374099 | +0.000749 |
| Precision@1,000 | 0.864000 | 0.867000 | +0.003000 |
| Recall@1,000 | 0.284024 | 0.285010 | +0.000986 |
| Fraud captured at 1,000 reviews | 864 | 867 | +3 |
| Prediction latency per row | 0.003354 ms | 0.003094 ms | -0.000260 ms |
| Training time | 660.83 s | 628.11 s | -32.72 s |
| Best boosting iteration | 499 | 498 | -1 |

The full-validation ranking comparison showed only a very small improvement in
PR-AUC and fraud captured at 1,000 reviews. This was insufficient for retention
without the identical calibrated policy evaluation used by the Phase 5 baseline.

---

## Calibration and policy protocol

The candidate model was assessed under the same chronological validation protocol
used in Phase 5.

| Validation period | Rows | Purpose |
|---|---:|---|
| Earlier validation period | 44,290 | Fit sigmoid / Platt calibration |
| Later validation period | 44,291 | Evaluate calibrated policy |
| Timestamp boundary | 11,725,712 | Keeps same-second transactions in one period |

The candidate model used sigmoid calibration because it was the calibration method
already selected on the Phase 5 baseline validation process.

The policy used the existing illustrative portfolio assumptions:

| Assumption | Value |
|---|---:|
| False-negative cost | £500 |
| Manual-review cost | £5 |
| False-positive escalation cost | £10 |
| Fraud-prevention value | £500 |
| Configured operational capacity | 1,000 reviews |

All values are illustrative benchmark assumptions. They are not real bank costs,
fraud losses, savings, or financial-institution outcomes.

---

## Policy comparison

The anomaly candidate and Phase 5 baseline were evaluated on the same later
chronological policy-selection period containing 1,449 known fraud transactions.

| Review capacity | Baseline fraud captured | Candidate fraud captured | Difference | Baseline net expected value | Candidate net expected value | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 392 | 392 | 0 | £193,500 | £193,500 | £0 |
| 1,000 | 568 | 567 | -1 | £279,000 | £278,500 | -£500 |
| 1,500 | 655 | 669 | +14 | £320,000 | £327,000 | +£7,000 |
| 2,000 | 731 | 744 | +13 | £355,500 | £362,000 | +£6,500 |

### Operational-capacity result

The documented operational capacity is 1,000 manual reviews.

| Metric at 1,000 reviews | Baseline | XGBoost + anomaly | Candidate minus baseline |
|---|---:|---:|---:|
| Fraud captured | 568 | 567 | -1 |
| Fraud capture rate | 39.20% | 39.13% | -0.07 percentage points |
| Net expected value | £279,000 | £278,500 | -£500 |

---

## Decision

**Decision: Reject `anomaly_score_isolation_forest` from the selected system.**

The anomaly candidate did not improve fraud capture or net expected value at the
configured operational capacity of 1,000 reviews. It captured one fewer known
fraud transaction and produced £500 less net expected value than the selected
Phase 5 baseline under the same calibration, capacity, and cost assumptions.

The candidate improved results at larger hypothetical capacities of 1,500 and
2,000 reviews. Those improvements do not justify retaining the feature because
the documented operational decision policy is constrained to 1,000 daily reviews.

The anomaly feature is therefore excluded from:

- The selected XGBoost feature contract.
- The selected sigmoid calibration workflow.
- The final policy configuration.
- Final holdout reporting.
- The final portfolio model.

The selected system remains:

```text
Phase 5 XGBoost v1.0.0
+ Phase 3 point-in-time behavioural features
+ time-safe target encoding
+ sigmoid / Platt calibration
+ top-k review policy at 1,000 reviews
```

---

## Generated artifacts

| Artifact | Purpose |
|---|---|
| `data/features/anomaly/train_anomaly_scores.parquet` | Local chronological out-of-fold training anomaly scores |
| `data/features/anomaly/val_anomaly_scores.parquet` | Local train-fitted validation anomaly scores |
| `data/features/anomaly/test_anomaly_scores.parquet` | Local train-fitted test anomaly scores; not used after candidate rejection |
| `data/features/anomaly/anomaly_feature_metadata.json` | Local anomaly-feature metadata and leakage controls |
| `reports/tables/xgboost_anomaly_validation_metrics.json` | Candidate full-validation ranking metrics |
| `reports/tables/xgboost_anomaly_feature_contract.json` | Candidate feature-contract record |
| `reports/tables/phase6_anomaly_ablation_comparison.csv` | Full-validation baseline-versus-candidate ranking comparison |
| `models/metrics/phase6_anomaly_validation_policy_evaluation.json` | Candidate validation policy evaluation |
| `reports/figures/phase6_anomaly_validation_policy_evaluation.png` | Candidate capacity-sensitivity figure |
| `reports/tables/phase6_anomaly_policy_comparison.csv` | Baseline-versus-candidate policy comparison |
| `docs/experiments/phase_6_anomaly_ablation_decision.md` | Formal keep/reject decision |

Large generated feature tables, local MLflow databases, MLflow artifacts, model
objects, and raw data remain excluded from GitHub.

---

## Conclusion

Isolation Forest was tested as a controlled advanced-feature ablation. It produced
a small full-validation ranking change, but it did not improve the selected
cost-sensitive policy at the configured review capacity.

The feature is rejected based on validation evidence. This is a valid Phase 6
result and will be reported clearly in the final README and ablation summary.