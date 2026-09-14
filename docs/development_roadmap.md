# Development Roadmap

## Phase 0 — Planning and Project Setup

- Create repository and project structure.
- Add README and Apache License 2.0.
- Add `.gitignore`, `.env.example`, and Python environment.
- Define requirements, data dictionary, database schema, API design, cost assumptions, and decision log.
- Create architecture and data-lineage diagrams.
- Create GitHub Issues and Project board.

## Phase 1 — Data Acquisition and Storage

- Document the IEEE-CIS dataset source and access information.
- Preserve immutable raw CSV files locally.
- Join transaction and identity tables using `TransactionID`.
- Convert processed data to Parquet.
- Create PostgreSQL analytical tables.
- Record ingestion row counts, schema, and join coverage.

## Phase 2 — Data Quality, Leakage Controls, and EDA

- Validate schema, duplicates, missingness, timestamps, and invalid values.
- Analyse missingness by time and fraud label.
- Create chronological train, validation, and test splits.
- Perform a written leakage audit.
- Produce fraud-pattern and temporal EDA reports.

## Phase 3 — Behavioural Features

- Create point-in-time-safe velocity features.
- Create recency and amount-deviation features.
- Create entity relationship and new-entity indicators.
- Create historical activity and rolling aggregate features.
- Catalogue feature availability and lookback windows.

## Phase 4 — Model Comparison

- Train Logistic Regression as the interpretable baseline.
- Train XGBoost or LightGBM using the same evaluation design.
- Compare PR-AUC, recall@k, precision@k, calibration, cost, and latency.
- Track experiments and model versions using MLflow.

## Phase 5 — Calibration and Decisioning

- Compare uncalibrated, Platt-scaled, and isotonic probabilities.
- Measure Brier score and calibration error.
- Implement cost-sensitive decisions.
- Optimise thresholds or top-k review policy using validation data.
- Evaluate the locked policy once on the final holdout.

## Phase 6 — Controlled Advanced Experiments

- Test anomaly scores.
- Test graph-derived features.
- Test sequence models only if justified.
- Use ablation experiments for every advanced component.
- Keep only components that improve the decision objective or analyst usefulness.

## Phase 7 — Explainability and Investigation Workflow

- Generate SHAP global and local explanations.
- Store versioned investigation cases.
- Build a Streamlit investigation queue.
- Display risk, action, explanation, and related historical information.

## Phase 8 — API and Replay

- Implement FastAPI endpoints.
- Validate scoring requests.
- Add batch scoring.
- Replay historical transactions in chronological order.
- Measure latency and throughput.

## Phase 9 — Testing and Monitoring

- Add unit, integration, and API tests.
- Monitor drift, missingness, volume, and risk-score distributions.
- Add delayed-label performance monitoring.
- Run checks through GitHub Actions.

## Phase 10 — Security and Presentation

- Redact sensitive fields from logs and dashboards.
- Prepare a reliable Docker demonstration or deployment.
- Complete the README, diagrams, screenshots, and evaluation reports.
- Document limitations and reproducibility instructions.

## Required Milestones

Each milestone must be tested and committed to Git before the next milestone begins.