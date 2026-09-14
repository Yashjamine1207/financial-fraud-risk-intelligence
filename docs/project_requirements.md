# Project Requirements

## Project Name

Financial Fraud Risk & Investigation Intelligence Platform

## Repository

`financial-fraud-risk-intelligence`

## Objective

Build an end-to-end fraud-risk decision system that:

- Scores financial transactions.
- Estimates calibrated fraud probability.
- Ranks transactions for investigation.
- Explains model predictions.
- Recommends approve, review, or block actions.
- Considers financial cost and limited investigation capacity.
- Records model, feature, data, calibration, threshold, and policy versions.

## Primary Dataset

IEEE-CIS Fraud Detection from Kaggle.

The transaction and identity tables will be joined using `TransactionID`.

## Functional Requirements

1. Ingest raw IEEE-CIS transaction and identity files.
2. Preserve raw data separately from processed data.
3. Convert processed data to Parquet.
4. Support analytical storage in PostgreSQL.
5. Validate schemas, missingness, duplicates, timestamps, and invalid values.
6. Create chronological train, validation, and test splits.
7. Perform and document a leakage audit.
8. Build point-in-time-safe behavioural features.
9. Train a Logistic Regression baseline.
10. Train and compare XGBoost or LightGBM.
11. Calibrate fraud probabilities.
12. Implement cost-sensitive approve, review, and block decisions.
13. Enforce an investigation-capacity constraint.
14. Generate SHAP explanations.
15. Provide a Streamlit investigation queue.
16. Provide FastAPI scoring endpoints.
17. Add batch scoring and historical replay.
18. Add monitoring for data drift and delayed-label performance.
19. Add automated unit, integration, and API tests.
20. Track experiments using MLflow.

## Non-Functional Requirements

- Use chronological validation.
- Prevent target leakage.
- Make experiments reproducible.
- Keep secrets outside Git.
- Do not upload the full dataset.
- Use modular Python code inside `src/`.
- Use configuration files for business assumptions.
- Record relevant versions with every prediction.
- Provide clear error messages and structured logs.
- Keep public-dataset limitations explicit.

## Success Criteria

The system must show:

- A reproducible ingestion path.
- A written leakage audit.
- Behavioural features computed using prior information only.
- A Logistic Regression and boosted-tree comparison.
- Calibrated probability results.
- Policy results at investigation capacity.
- SHAP explanations for reviewed cases.
- A working API prediction path.
- Passing automated tests.
- A documented business-value evaluation.

## Out of Scope for the MVP

The following will not be added before the MVP is complete:

- LSTM or GRU sequence models.
- Graph modelling.
- LLM features.
- Kafka or real-time streaming.
- Public cloud deployment.
- Complex authentication.