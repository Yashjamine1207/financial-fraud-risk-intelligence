# Data Card — IEEE-CIS Fraud Detection Dataset

**Dataset name**: IEEE-CIS Fraud Detection  
**Source**: Kaggle competition (Vesta Corporation)  
**License**: Competition terms apply — see [Kaggle competition page](https://www.kaggle.com/competitions/ieee-fraud-detection)  
**Access date**: 2026-09-14  
**Version**: Raw CSV files preserved in `data/raw/ieee_cis/`

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total transactions | 590,540 |
| Fraud rate | 3.5% |
| Identity table coverage | 24.4% |
| Time span (TransactionDT) | 86,400 to 15,811,131 seconds |
| Number of features | 433 (excluding TransactionID, isFraud) |

**Class imbalance**: Severe (3.5% fraud, 96.5% non-fraud)  
**Data type**: E-commerce transactions with identity/linkage information

---

## 2. Table Schema

### Transaction Table (train_transaction.csv)

**Primary key**: TransactionID (unique, no duplicates)  
**Foreign keys**: None (identity table references this)

| Column group | Columns | Description |
|-------------|---------|-------------|
| **Key fields** | TransactionID, isFraud | Unique ID, binary target (0=legit, 1=fraud) |
| **Timestamp** | TransactionDT | Seconds since reference point (monotonic increasing) |
| **Amount** | TransactionAmt | Transaction amount in USD |
| **Product** | ProductCD | C, S, H, R, W (card product type) |
| **Card features** | card1–card6 | Card ID proxies, category, type, issuer |
| **Address** | addr1, addr2 | Billing address proxies |
| **Email** | P_emaildomain, R_emaildomain | Purchaser and recipient email domains |
| **Device** | DeviceType, DeviceInfo | Device category and detailed info |
| **Identity features** | id_01–id_38 | Identity verification signals (mixed types) |
| **Obfuscated features** | M1–M9, C1–C14, D1–D15 | Anonymized categorical, count, and time-since-previous features |

### Identity Table (train_identity.csv)

**Primary key**: TransactionID (subset of transaction table)  
**Foreign key**: References TransactionID in transaction table

| Column group | Columns | Description |
|-------------|---------|-------------|
| **Key** | TransactionID | Links to transaction table |
| **Identity signals** | id_01–id_38 | Device, browser, location, and behavioural signals |

**Join coverage**: 144,233 / 590,540 (24.4%)  
**Join method**: Left join (transaction → identity) on TransactionID

---

## 3. Data Quality

### Validation results (2026-09-14)

- ✅ Schema validation: PASSED (Pandera)
- ✅ Duplicate TransactionIDs: 0
- ✅ Timestamp monotonicity: PASSED
- ✅ Invalid value ranges: 0

### Missingness summary

| Category | Count | Notes |
|----------|-------|-------|
| Columns with any nulls | 414 / 434 | 95.6% of columns |
| Columns with >95% nulls | 9 | id_24, id_25, id_07, id_08, id_21, id_26, id_22, id_23, id_27 |
| Columns with differential missingness (p<0.05) | 326 | Missingness correlates with fraud label |

**See**: `docs/data/leakage_audit.md` for detailed leakage risks

---

## 4. Temporal Structure

**TransactionDT**: Seconds since an undisclosed reference point  
**Approximate span**: ~183 days (assuming 1-second granularity)

### Temporal splits (70/15/15)

| Split | Rows | Percentage | TransactionDT range | Fraud rate |
|-------|------|------------|---------------------|------------|
| Train | 413,378 | 70.0% | 86,400 – 10,437,996 | 3.52% |
| Validation | 88,581 | 15.0% | 10,438,003 – 13,151,840 | 3.43% |
| Test | 88,581 | 15.0% | 13,151,880 – 15,811,131 | 3.48% |

**Important**: Test set is held out for final evaluation only — no model selection or threshold tuning.

---

## 5. Known Limitations

1. **Obfuscated features**: M, C, D features are anonymized — semantic meaning unknown
2. **Identity coverage**: Only 24% of transactions have identity data
3. **Synthetic elements**: Some features may be synthetic or modified for privacy
4. **Label delay**: Fraud labels may be assigned days/weeks after transaction (not reflected in data)
5. **Class imbalance**: 3.5% fraud rate requires specialized evaluation metrics (PR-AUC, not accuracy)
6. **Temporal drift**: Fraud patterns may change over time (requires monitoring in production)

---

## 6. Intended Use

**This dataset is for**:
- Fraud detection model development and benchmarking
- Research on imbalanced classification, temporal validation, and cost-sensitive decisioning
- Educational purposes (portfolio project)

**NOT for**:
- Production deployment without additional validation
- Inferring real customer behaviour (data is anonymized/obfuscated)
- Claims about actual financial institution performance

---

## 7. References

- Kaggle competition: https://www.kaggle.com/competitions/ieee-fraud-detection
- Competition rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
- Data dictionary: `docs/data/data_dictionary.md`
- Leakage audit: `docs/data/leakage_audit.md`
- Feature catalog: `docs/data/feature_catalog.md`