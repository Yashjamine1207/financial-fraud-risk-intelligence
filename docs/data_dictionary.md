# Data Dictionary

## Dataset

IEEE-CIS Fraud Detection dataset from Kaggle.

Source:

<https://www.kaggle.com/competitions/ieee-fraud-detection/data>

## Files

| File | Purpose |
|---|---|
| `train_transaction.csv` | Training transaction records and fraud labels |
| `train_identity.csv` | Identity information associated with training transactions |
| `test_transaction.csv` | Unlabelled test transaction records |
| `test_identity.csv` | Identity information associated with test transactions |
| `sample_submission.csv` | Kaggle submission format |

## Join Key

The transaction and identity tables will be joined using:

```text
TransactionID
```

## Important Fields

| Field | Source | Description |
|---|---|---|
| `TransactionID` | Transaction | Unique transaction identifier |
| `isFraud` | Transaction | Binary target label available in training data |
| `TransactionDT` | Transaction | Relative transaction-time field |
| `TransactionAmt` | Transaction | Transaction amount |
| `ProductCD` | Transaction | Product code |
| `card1` to `card6` | Transaction | Obfuscated card-related fields |
| `addr1`, `addr2` | Transaction | Obfuscated address-related fields |
| `dist1`, `dist2` | Transaction | Distance-related fields |
| `P_emaildomain` | Transaction | Purchaser email-domain proxy |
| `R_emaildomain` | Transaction | Recipient email-domain proxy |
| `DeviceType` | Identity | Device-type field |
| `DeviceInfo` | Identity | Device-information field |
| `id_01` to `id_38` | Identity | Obfuscated identity fields |

## Data Handling Rules

- Preserve raw files without modification.
- Do not upload full CSV files to GitHub.
- Do not treat obfuscated fields as directly interpretable real-world identifiers.
- Investigate missingness before deciding on imputation.
- Use only scoring-time-available information in features.
- Exclude target-derived information from production scoring features.
- Record feature definitions and availability assumptions.

## Known Dataset Limitations

- The dataset is anonymised and contains obfuscated fields.
- Field meanings are partly unknown.
- The labels represent the available dataset labels and may not reflect all real-world fraud.
- Public-dataset results are benchmark results, not financial-institution outcomes.