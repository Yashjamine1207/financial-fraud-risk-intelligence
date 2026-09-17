# Leakage Audit — IEEE-CIS Fraud Detection

**Purpose**: Document all potential data leakage risks, feature availability constraints, and exclusion decisions to ensure production-safe modelling.

**Last updated**: 2026-09-14  
**Dataset version**: IEEE-CIS train_transaction + train_identity (joined)  
**Temporal split**: 70% train (DT: 86,400–10,437,996), 15% val (DT: 10,438,003–13,151,840), 15% test (DT: 13,151,880–15,811,131)

---

## 1. Target Leakage Risks

### High-risk columns (excluded from features)

| Column | Reason for exclusion | Production availability |
|--------|---------------------|------------------------|
| `isFraud` | Target variable — never use as feature | Post-investigation (days/weeks later) |
| Any column with >95% nulls | Too sparse for reliable signal | Unreliable |
| Identity columns with 31%+ differential missingness | Missingness itself predicts fraud (leakage risk) | Requires careful handling |

### Columns with differential missingness (p<0.05)

326 columns show significantly different null rates between fraud/non-fraud classes. Top concerns:

- `R_emaildomain`: 32.2% missingness difference
- `id_02`, `id_15`, `id_35`–`id_38`, `id_11`, `id_28`, `id_29`: ~31.5% difference

**Decision**: These columns can be used ONLY if:
1. Missingness is explicitly modelled (e.g., add `is_null` indicator)
2. Imputation uses training data statistics only
3. No target-based imputation

---

## 2. Temporal Leakage Prevention

### Rules enforced

- ✅ **Chronological splits only**: No random shuffling of final evaluation data
- ✅ **Test set untouched**: Reserved for final evaluation only (no model selection, no threshold tuning)
- ✅ **Feature lookback windows**: All behavioural features must use only past transactions (TransactionDT < current transaction)

### Validation checks

- [ ] All feature engineering scripts validated for time-safety
- [ ] Target encodings computed within CV folds only
- [ ] No global statistics computed on full dataset

---

## 3. Feature Availability at Scoring Time

### Safe to use (available at transaction time)

| Feature category | Examples | Notes |
|-----------------|----------|-------|
| Transaction metadata | `TransactionDT`, `TransactionAmt`, `ProductCD` | Always available |
| Card proxies | `card1`–`card6` | Available, but some have high null rates |
| Address | `addr1`, `addr2` | Available |
| Device/browser | `DeviceType`, `DeviceInfo` | Available |
| Email domain | `R_emaildomain`, `P_emaildomain` | Available, but high missingness |

### Requires historical lookback

| Feature | Requires | Production implementation |
|---------|----------|--------------------------|
| Transaction velocity (5min, 1hr, 24hr) | Past transactions for same card/device | Rolling window aggregation |
| Amount deviation from historical mean | Past transactions | Store entity-level statistics |
| New device indicator | Historical device usage | Entity history table |
| Relationship counts (shared email/device) | Graph of historical transactions | Entity-resolution system |

### NOT available at scoring time (exclude from features)

- Any post-investigation labels
- Aggregations that include future transactions
- Global statistics computed on full dataset (must use training-only)

---

## 4. Identity Table Join

**Join method**: Left join on `TransactionID` (transaction → identity)  
**Identity coverage**: 24.4% (144,233 / 590,540)

**Leakage risk**: Identity fields are only populated for 24% of transactions. If identity presence correlates with fraud (it does — 31% missingness difference), using identity fields without accounting for missingness creates leakage.

**Mitigation**:
- Always include `has_identity` flag as a feature
- Model identity fields separately or use missingness indicators
- Never impute identity fields using fraud-label information

---

## 5. Data Quality Validation

**Schema validation**: Passed (Pandera)  
**Duplicate TransactionIDs**: 0  
**Timestamp monotonicity**: Passed  
**Invalid value ranges**: 0

**High-null columns (>95%)**: 9 columns identified  
- `id_24`, `id_25`, `id_07`, `id_08`, `id_21`, `id_26`, `id_22`, `id_23`, `id_27`

**Decision**: Exclude from initial modelling; revisit after baseline established.

---

## 6. Decisions Log

| Date | Decision | Rationale | Owner |
|------|----------|-----------|-------|
| 2026-09-14 | Exclude columns with >95% nulls from baseline | Too sparse, unreliable signal | Data team |
| 2026-09-14 | Use chronological 70/15/15 split | Prevent temporal leakage, mimic production | Data team |
| 2026-09-14 | Test set locked for final evaluation only | No model selection on test data | Data team |
| 2026-09-14 | Add missingness indicators for high-differential columns | Capture signal without leakage | Data team |

---

## 7. Production Checklist

Before deploying any model to production scoring:

- [ ] All features computable using only data available at transaction time
- [ ] No target leakage in feature engineering (reviewed by second engineer)
- [ ] Temporal validation shows stable performance across time periods
- [ ] Calibration performed on validation set only
- [ ] Thresholds optimized on validation set, evaluated once on test set
- [ ] Model card documents all feature availability assumptions

---

## References

- IEEE-CIS Fraud Detection competition rules: https://www.kaggle.com/competitions/ieee-fraud-detection
- Data dictionary: `docs/data/data_dictionary.md`
- Feature catalog: `docs/data/feature_catalog.md`