# Financial Fraud Risk & Investigation Intelligence Platform

An end-to-end financial fraud decision-intelligence platform that ranks suspicious transactions, estimates calibrated fraud risk, explains model predictions, and recommends approve, review, or block actions under a configurable investigation capacity.

## Project Status

Phase 0 — Planning and project setup.

The repository structure and Python environment are being prepared before data modelling begins.

## Project Objective

The platform is designed to answer four operational questions:

1. Which transactions should be flagged?
2. How confident is the system?
3. Which model features contributed to the prediction?
4. What action should be recommended under financial cost and review-capacity constraints?

The project is designed as a decision system rather than a simple fraud-classification notebook.

## Primary Dataset

IEEE-CIS Fraud Detection dataset from Kaggle:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

The primary modelling dataset is created by joining the transaction and identity tables using `TransactionID`.

Dataset source:

<https://www.kaggle.com/competitions/ieee-fraud-detection/data>

The full dataset is not included in this repository.

## Optional Dataset

PaySim may be used later for pipeline-scale or load testing.

PaySim is simulated mobile-money transaction data and will not be presented as the primary dataset or as evidence of real financial-institution performance.

Dataset source:

<https://www.kaggle.com/datasets/ealaxi/paysim1>

## Planned Technology Stack

- Python
- Pandas and Polars
- NumPy, SciPy, and statsmodels
- PostgreSQL and SQLAlchemy
- Apache Parquet and PyArrow
- Pandera or Great Expectations
- scikit-learn
- Logistic Regression
- XGBoost or LightGBM
- SHAP
- MLflow
- FastAPI
- Streamlit
- Docker and Docker Compose
- pytest
- GitHub Actions

Advanced methods such as anomaly detection, sequence modelling, and graph features will be added only after controlled ablation experiments.

## Planned System Pipeline

```text
IEEE-CIS data
    ↓
Data ingestion
    ↓
Data validation and quality checks
    ↓
Temporal splitting and leakage audit
    ↓
Point-in-time behavioural features
    ↓
Logistic Regression baseline
    ↓
XGBoost or LightGBM comparison
    ↓
Probability calibration
    ↓
Cost-sensitive decision policy
    ↓
SHAP explanations
    ↓
Investigation queue
    ↓
FastAPI scoring service
    ↓
Monitoring and evaluation
```

## Evaluation Principles

The project will use chronological train, validation, and test splits.

The final evaluation data will remain untouched until model selection, calibration, and policy selection are complete.

Primary evaluation measures will include:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- Precision@k
- Recall@k
- Brier score
- Calibration error
- Expected financial cost
- Fraud captured at investigation capacity
- API latency and throughput

Accuracy will not be used as the main success metric.

## Decision Policy

The system will use documented and versioned assumptions for:

- False-negative loss
- False-positive and manual-review cost
- Fraud-prevention value
- Investigation capacity
- Escalation actions
- Approve, review, and block recommendations

Thresholds and review priorities will be selected using the documented cost model rather than arbitrary probability cut-offs.

## Leakage and Explainability Rules

All features must use only information available at transaction-scoring time.

Historical and behavioural features must be calculated using prior transactions only.

SHAP explanations will describe which features contributed to a prediction. They will not be presented as causal explanations or proof that a transaction is fraudulent.

## Repository Safety

The repository will not contain:

- Full datasets
- Raw customer-like records
- API keys or credentials
- `.env` files
- Database volumes
- Large model artifacts
- MLflow run artifacts

Only small, anonymised sample files may be stored in `data/sample/`.

## Project Limitations

Results will be reported as public-dataset benchmark results and simulated policy outcomes.

They will not be presented as results from a real financial institution.

The dataset is anonymised and contains obfuscated fields. Cost assumptions, investigation capacity, and operational outcomes are illustrative. Model predictions require human review.

## Development Approach

The project will be built in phases.

Each completed milestone will be tested and committed to GitHub before the next phase begins.

The MVP will be completed before adding LSTM or GRU models, graph analysis, LLM features, Kafka, streaming, or cloud deployment.

<!-- FINAL_RESULTS:START -->

## Final Model and Results

### Selected approach

The final benchmark model is **XGBoost** (`xgboost-v1.0.0`),
with sigmoid / Platt calibration and a constrained top-
`1,000` transaction review policy.

The model family, feature set, calibration method, and policy were selected using
chronological training and validation periods. The final chronological holdout was
reserved for locked evaluation.

### Final holdout performance

| Metric | Result |
| --- | ---: |
| Review capacity | 1,000 |
| Fraud-labelled transactions | 3,083 |
| Captured fraud | 870 |
| Fraud capture rate | 28.22% |
| Legitimate transactions reviewed | 130 |
| Review precision | 87.00% |
| Illustrative net expected value | GBP 430,000.00 |

The net expected value is a benchmark calculation under documented illustrative
cost assumptions. It is not realised savings or a real financial-institution
forecast.

### Why this model

- XGBoost was selected after chronological comparison against Logistic Regression.
- Sigmoid calibration reduced validation Brier score from
  `0.0685` to
  `0.0229` and ECE from
  `0.1612` to
  `0.0037`.
- The final policy respects a fixed investigation capacity instead of using an
  arbitrary probability threshold.
- Phase 6 anomaly, graph, and sequence experiments were treated as controlled
  ablations; the simpler calibrated XGBoost model remained the selected champion.
- Phase 7 includes global and local SHAP attribution, SHAP stability checks,
  missed-fraud and unnecessary-review analysis, and capacity/value trade-offs.

### Main limitations

- This is an IEEE-CIS public-data benchmark, not a production fraud system.
- SHAP contributors describe model behaviour and do not prove fraud or causation.
- Review capacity and financial values are documented illustrative assumptions.
- Results should not be presented as actual fraud savings, institutional
  performance, or a deployed financial-service product.

See the final summary in
[`reports/evaluation/final_model_selection_report.md`](reports/evaluation/final_model_selection_report.md).

<!-- FINAL_RESULTS:END -->
