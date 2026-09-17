# Feature Catalog — IEEE-CIS Fraud Detection

**Purpose**: Document all available features, their types, availability at scoring time, and production implementation requirements.

**Dataset version**: IEEE-CIS train_transaction + train_identity (joined)  
**Last updated**: 2026-09-14

---

## 1. Feature Categories

| Category | Feature count | Availability | Notes |
|----------|---------------|--------------|-------|
| Transaction metadata | 4 | Real-time | Always available |
| Product & card | 9 | Real-time | High-cardinality proxies |
| Address & email | 4 | Real-time | High missingness |
| Device & browser | 2 + identity fields | Real-time | Identity table 24% coverage |
| Identity signals | 38 | Real-time (partial) | Only for 24% of transactions |
| Obfuscated M features | 9 | Real-time | Categorical, anonymized |
| Obfuscated C features | 14 | Real-time | Count-based, anonymized |
| Obfuscated D features | 15 | Real-time | Time-since-previous, anonymized |
| **Total** | **~433** | — | Excluding TransactionID, isFraud |

---

## 2. Transaction Metadata Features

### Always available, high reliability

| Feature | Type | Description | Production notes |
|---------|------|-------------|------------------|
| TransactionDT | int64 | Seconds since reference point | Use for temporal ordering, feature lookback windows |
| TransactionAmt | float64 | Transaction amount in USD | Log-transform recommended (right-skewed) |
| ProductCD | object | Card product type: C, S, H, R, W | Strong signal — 'C' has 11.2% fraud rate, 'W' has 2.1% |
| TransactionID | int64 | Unique transaction identifier | Key for joins, not a predictive feature |

**Leakage risk**: None — all available at transaction time.

---

## 3. Card Features

### Available at scoring time, moderate missingness

| Feature | Type | Unique values | Null rate | Notes |
|---------|------|---------------|-----------|-------|
| card1 | float64 | ~3,000 | 0% | Card ID proxy — high cardinality |
| card2 | float64 | ~500 | 2.2% | Card ID proxy — medium cardinality |
| card3 | float64 | ~100 | 3.2% | Card ID proxy — low cardinality |
| card4 | object | 4 | 0% | Card type: debit, credit, charge card |
| card5 | float64 | ~200 | 3.2% | Card issuer proxy |
| card6 | object | 2–4 | 18% | Card category: debit/credit (some missing) |

**Production implementation**:
- Encode card1–card3, card5 as categorical (target encoding or frequency encoding)
- card4, card6: one-hot or ordinal encoding
- Add `card2_null`, `card6_null` indicators (missingness is informative)

**Leakage risk**: Low — available at transaction time.

---

## 4. Address & Email Features

### Available at scoring time, high missingness

| Feature | Type | Unique values | Null rate | Fraud-rate difference |
|---------|------|---------------|-----------|----------------------|
| addr1 | float64 | ~300 | 15% | Moderate |
| addr2 | float64 | ~100 | 15% | Moderate |
| P_emaildomain | object | ~100 | 13% | 32% difference (high leakage risk) |
| R_emaildomain | object | ~100 | 54% | 32% difference (high leakage risk) |

**Production implementation**:
- Encode as categorical (frequency or target encoding with smoothing)
- **Must include** `P_emaildomain_null`, `R_emaildomain_null` flags
- Consider grouping rare domains into "other" category

**Leakage risk**: **HIGH** — missingness strongly correlates with fraud. Do not impute without including null indicators.

---

## 5. Device Features

### Available at scoring time

| Feature | Type | Unique values | Null rate | Notes |
|---------|------|---------------|-----------|-------|
| DeviceType | object | 2 | 0% | Desktop or Mobile |
| DeviceInfo | object | ~3,000 | 4% | Detailed device info (browser + OS + version) |

**Production implementation**:
- DeviceType: binary encoding
- DeviceInfo: extract manufacturer/model, encode as categorical
- Add `DeviceInfo_null` flag

**Leakage risk**: Low — available at transaction time.

---

## 6. Identity Features (id_01–id_38)

### Partially available (24% coverage), high leakage risk

| Feature | Type | Null rate | Fraud-rate difference | Notes |
|---------|------|-----------|----------------------|-------|
| id_01–id_03 | float64 | 10–40% | 31% (id_02) | Identity verification scores |
| id_04 | object | 40% | — | T/F flag |
| id_05–id_06 | float64 / object | 40–60% | 31% (id_06) | Identity signals |
| id_07–id_27 | mixed | 93–99% | — | **Exclude from baseline** — too sparse |
| id_28–id_29 | object | 40–60% | 31% | T/F flags |
| id_30–id_33 | object | 40–60% | — | OS, browser info |
| id_34–id_38 | object / float64 | 40–60% | 31% | T/F flags, ratios |

**Production implementation**:
- **Only use identity features if identity table is present** (24% of transactions)
- Always include `has_identity` flag (1 if identity record exists, 0 otherwise)
- For identity fields: include null indicators for all columns with >5% nulls
- Consider separate model for identity-rich vs. identity-poor transactions

**Leakage risk**: **VERY HIGH** — 326 columns have differential missingness (p<0.05). Missingness itself predicts fraud.

**Recommendation**: Start with identity features excluded or heavily regularized. Add incrementally with ablation experiments.

---

## 7. Obfuscated Features

### M Features (M1–M9)

| Feature | Type | Unique values | Null rate | Notes |
|---------|------|---------------|-----------|-------|
| M1–M9 | object | 2 (T/F) | 0–50% | Categorical flags, anonymized |

**Production implementation**:
- Binary encoding (T=1, F=0)
- Add null indicators for M4, M6 (higher missingness)

### C Features (C1–C14)

| Feature | Type | Description | Null rate |
|---------|------|-------------|-----------|
| C1–C14 | float64 | Count-based features (e.g., number of transactions in time window) | 0–30% |

**Production implementation**:
- Log-transform (count data, right-skewed)
- Some may represent velocity features — validate against behavioural feature pipeline

### D Features (D1–D15)

| Feature | Type | Description | Null rate |
|---------|------|-------------|-----------|
| D1–D15 | float64 | Time since previous event (e.g., days since last transaction) | 0–85% |

**Production implementation**:
- These are **time-since-previous** features — similar to behavioural features we'll engineer
- Clip extreme values (right-skewed)
- Add null indicators for high-missingness D features

**Leakage risk**: Moderate — these are pre-computed behavioural features. Ensure our engineered features don't duplicate them (avoid multicollinearity).

---

## 8. Behavioural Features to Engineer (Phase 3)

### Point-in-time-safe features (available at scoring time)

| Feature | Lookback window | Requires | Production implementation |
|---------|----------------|-----------|--------------------------|
| Transaction velocity (5min, 1hr, 24hr) | Rolling window | Historical transactions by card/device | Streaming aggregation or batch pre-computation |
| Amount z-score vs. historical mean | 7–30 days | Entity-level statistics | Store rolling mean/std per card/device |
| Recency (time since last transaction) | N/A | Last transaction timestamp | Track last_seen per entity |
| New device indicator | N/A | Historical device usage | Flag if DeviceInfo not seen before for entity |
| Relationship counts (shared email/device) | N/A | Entity graph | Pre-compute entity-degree, shared-attribute counts |
| Historical fraud rate (entity-level) | 30–90 days | Past labels | **Leakage risk** — only use if labels available in production |

**Leakage prevention**:
- All features must use only transactions with `TransactionDT < current_transaction`
- No global statistics — only entity-level aggregations
- Target encodings must be computed within CV folds (no full-dataset leakage)

---

## 9. Feature Exclusion List (Baseline Model)

### Exclude from initial modelling

| Feature | Reason for exclusion |
|---------|---------------------|
| id_07, id_08, id_21, id_22, id_23, id_24, id_25, id_26, id_27 | >95% nulls — too sparse |
| Any feature with >90% nulls and no clear production availability | Unreliable signal |
| Global aggregates computed on full dataset | Temporal leakage |

### Revisit after baseline

| Feature | Condition for inclusion |
|---------|------------------------|
| Identity fields (id_01–id_38) | After ablation shows positive lift |
| D1–D15 (time-since-previous) | After validating against engineered behavioural features |
| High-cardinality fields (card1, DeviceInfo) | After implementing proper encoding strategy |

---

## 10. Production Feature Pipeline Requirements

### Real-time scoring (API)

- **Input**: Transaction fields (amount, product, card, device, email)
- **Lookup**: Entity-level historical features from database (velocity, recency, amount stats)
- **Computation**: Feature engineering must complete in <100ms for API latency SLA
- **Versioning**: Feature version must be logged with every prediction

### Batch scoring

- **Input**: Historical transaction stream
- **Pre-computation**: Entity-level aggregates updated daily/hourly
- **Validation**: Schema checks, null rate monitoring, drift detection

---

## References

- Data card: `docs/data/data_card.md`
- Leakage audit: `docs/data/leakage_audit.md`
- Data dictionary: `docs/data/data_dictionary.md`
- Feature engineering module: `src/fraud_intelligence/features/`