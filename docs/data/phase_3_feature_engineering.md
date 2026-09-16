# Phase 3: Behavioural Features and Statistical Evidence

## Purpose

Phase 3 created reusable, point-in-time-safe behavioural features for the IEEE-CIS Fraud Detection dataset.

The objective was to create transaction-scoring features that use only information available before the transaction being scored. The pipeline does not use future transactions, post-investigation outcomes, or final-test data during feature creation.

Feature pipeline version: `v1.0.0`

## Input datasets

| Dataset | Rows | Purpose |
|---|---:|---|
| Training split | 413,378 | Feature generation, chronological out-of-fold target encoding, and statistical evidence analysis |
| Validation split | 88,581 | Feature generation using training history and earlier validation history |
| Final test split | 88,581 | Locked and not used during Phase 3 |

The final test split was not loaded during validation feature generation, target encoding, or statistical evidence analysis.

## Point-in-time safety rules

Every behavioural feature follows these rules:

1. Transactions are processed in chronological order using `TransactionDT`.
2. Historical features use only transactions with a strictly earlier timestamp.
3. Transactions that share the same `TransactionDT` do not use one another as history.
4. The transaction currently being scored is excluded from its own velocity, amount, recency, and entity-history calculations.
5. Missing entity identifiers are not treated as one shared entity.
6. The `isFraud` target is not used in velocity, amount, recency, historical-count, or new-entity features.
7. Target encoding uses chronological out-of-fold training logic.
8. Validation target encodings use mappings fitted from training labels only.
9. Validation labels are not used to generate validation target encodings.
10. The final test period remains locked until final model, calibration, and policy selection.

## Behavioural features

### Velocity features

| Feature | Definition | Lookback window | Availability |
|---|---|---|---|
| `velocity_global_5min` | Number of earlier transactions across the full transaction stream | 5 minutes | Available at scoring time |
| `velocity_global_60min` | Number of earlier transactions across the full transaction stream | 1 hour | Available at scoring time |
| `velocity_global_1440min` | Number of earlier transactions across the full transaction stream | 24 hours | Available at scoring time |
| `velocity_card1_5min` | Number of earlier transactions for the same `card1` proxy | 5 minutes | Available at scoring time if `card1` is available |
| `velocity_card1_60min` | Number of earlier transactions for the same `card1` proxy | 1 hour | Available at scoring time if `card1` is available |
| `velocity_card1_1440min` | Number of earlier transactions for the same `card1` proxy | 24 hours | Available at scoring time if `card1` is available |

### Historical amount features

For each configured window, the pipeline creates:

- Historical mean transaction amount
- Historical standard deviation of transaction amount
- Historical transaction amount z-score

The z-score represents how unusual the current amount is relative to prior observed amounts. It is a model input, not proof that a transaction is fraudulent.

| Entity | Windows |
|---|---|
| Global transaction stream | 1 hour, 24 hours, 168 hours |
| `card1` proxy | 1 hour, 24 hours, 168 hours |

Created amount feature pattern:

```text
amount_<entity>_<window>_mean
amount_<entity>_<window>_std
amount_<entity>_<window>_zscore
```

Examples:

```text
amount_global_1h_mean
amount_global_24h_zscore
amount_card1_1h_mean
amount_card1_168h_std
```

Where insufficient earlier history exists:

- Historical mean is missing when no prior transaction exists.
- Historical standard deviation and z-score are missing when fewer than two prior transactions exist.
- These missing values will be handled later by preprocessing fitted on training data only.

### Recency features

| Feature | Definition | Availability |
|---|---|---|
| `recency_global_seconds` | Seconds since the most recent transaction at a strictly earlier timestamp | Available at scoring time |
| `recency_card1_seconds` | Seconds since the previous transaction for the same `card1` proxy at a strictly earlier timestamp | Available when `card1` is available |

### Historical entity counts and new-entity indicators

| Entity | Historical count feature | New-entity indicator | Missing entity handling |
|---|---|---|---|
| Card proxy | `history_card1_transaction_count` | `is_new_card1` | `card1` is available for all training rows |
| Address proxy | `history_addr1_transaction_count` | `is_new_addr1` | Missing values remain missing |
| Device | `history_DeviceInfo_transaction_count` | `is_new_DeviceInfo` | Missing device information is not treated as a shared device |
| Recipient email domain | `history_R_emaildomain_transaction_count` | `is_new_R_emaildomain` | Missing domains are not treated as a shared domain |

Interpretation:

- `history_<entity>_transaction_count = 0` means no earlier transaction exists for that known entity.
- `is_new_<entity> = 1` means the entity has not appeared at an earlier timestamp.
- Missing feature values mean that the underlying entity identifier was unavailable.

## Time-safe target encoding

The pipeline creates smoothed target-encoded versions of these categorical fields:

```text
ProductCD_target_encoded
card4_target_encoded
card6_target_encoded
P_emaildomain_target_encoded
R_emaildomain_target_encoded
```

Configuration:

| Setting | Value |
|---|---:|
| Smoothing | 10 |
| Minimum category samples | 20 |
| Chronological out-of-fold splits | 5 |
| Training global fraud rate | 0.035169 |

Training rows use chronological out-of-fold encoding:

- Each time fold is encoded using labels from strictly earlier timestamp folds.
- The earliest chronological fold has no earlier labelled history, so its target-encoded values are missing.
- The later preprocessing pipeline will impute those values using training-only logic.

Validation rows use mappings fitted on the complete training period only:

- Validation labels are never used.
- Unseen validation categories use the global fraud rate from training data.
- Validation target-encoded columns contain no missing values.

## Generated feature tables

| Output file | Rows | Columns | Description |
|---|---:|---:|---|
| `data/features/point_in_time/train_behavioural_features.parquet` | 413,378 | 473 | Training behavioural features and target encodings |
| `data/features/point_in_time/validation_behavioural_features.parquet` | 88,581 | 473 | Validation features generated using valid historical training context |
| `data/features/point_in_time/train_feature_metadata.json` | N/A | N/A | Behavioural feature metadata |
| `data/features/point_in_time/validation_feature_metadata.json` | N/A | N/A | Validation feature metadata |
| `data/features/point_in_time/target_encoding_metadata.json` | N/A | N/A | Target-encoding configuration and leakage rules |

The 473 columns comprise the original modelling columns plus 39 Phase 3 engineered features:

- 6 velocity features
- 18 historical amount features
- 2 recency features
- 4 historical entity-count features
- 4 new-entity indicators
- 5 target-encoded categorical features

## Statistical evidence

The analysis used the training feature table only. It did not load validation or final-test data.

| Comparison | Group A fraud rate | Group B fraud rate | Difference | Relative risk | p-value |
|---|---:|---:|---:|---:|---:|
| New `card1` versus previously seen `card1` | 2.30% | 3.55% | -1.25 percentage points | 0.65 | < 0.00000001 |
| New device versus previously seen device | 5.63% | 6.81% | -1.18 percentage points | 0.83 | 0.06745201 |
| High 60-minute `card1` velocity versus lower velocity | 3.44% | 3.52% | -0.08 percentage points | 0.98 | 0.51451119 |

High card velocity was defined as:

```text
velocity_card1_60min >= 6
```

This threshold is the 95th percentile of the training distribution.

### Interpretation limits

- The findings describe association within the IEEE-CIS training period.
- They do not prove that a new card, new device, or high velocity caused fraud.
- A feature that has a weak individual association may still improve a multivariable model through interactions with other features.
- No feature will be retained or removed solely from these univariate comparisons.
- Model selection in Phase 4 will use chronological validation, PR-AUC, precision and recall at review capacity, calibration, latency, and expected cost rather than p-values alone.
- Public-dataset results are benchmark results, not financial-institution production outcomes.

## Automated tests

The following automated tests passed:

```text
tests/unit/test_point_in_time_features.py
7 passed
```

The tests verify:

- Current transactions are excluded from historical calculations.
- Same-timestamp transactions do not use one another as history.
- Historical velocity, amount, recency, entity-count, and new-entity features use prior timestamps only.
- Missing entity identifiers are not grouped together.
- Behavioural features do not depend on `isFraud`.

```text
tests/unit/test_target_encoding.py
4 passed
```

The tests verify:

- Same-timestamp transactions cannot use one another's fraud labels.
- A training row cannot use its own `isFraud` label for target encoding.
- Target encodings use chronological out-of-fold history.
- Validation mappings are fitted from training labels only.
- Unseen validation categories fall back to the training-period global fraud rate.

## Phase 3 completion decision

Phase 3 is complete when the final Phase 3 test suite passes and the implementation is committed to Git.

The final test set remains untouched. It will be used only after model selection, calibration selection, and decision-threshold policy selection.