# Decision Log

## Decision 001 — Primary Dataset

- Date: 2026-09-14
- Decision: Use IEEE-CIS Fraud Detection as the primary dataset.
- Reason: It contains transaction and identity tables, severe imbalance, missingness, high-cardinality fields, obfuscated variables, and temporal structure.
- Status: Accepted.

## Decision 002 — Optional Dataset

- Date: 2026-09-14
- Decision: Use PaySim only for optional pipeline-scale or load testing.
- Reason: PaySim is simulated mobile-money data and is not suitable as the main real-world benchmark for this project.
- Status: Accepted.

## Decision 003 — Licence

- Date: 2026-09-14
- Decision: Use Apache License 2.0 for the repository.
- Status: Accepted.

## Decision 004 — Validation Design

- Date: 2026-09-14
- Decision: Use chronological train, validation, and test splits.
- Reason: The fraud problem contains temporal structure, and random final evaluation splits may produce unrealistic estimates.
- Status: Accepted.

## Decision 005 — Model Order

- Date: 2026-09-14
- Decision: Train Logistic Regression before XGBoost or LightGBM.
- Reason: The baseline is required to measure incremental value from more complex models.
- Status: Accepted.

## Decision 006 — Advanced Methods

- Date: 2026-09-14
- Decision: Add anomaly, sequence, or graph features only after ablation evidence.
- Reason: Technology will not be added without measurable predictive, calibration, cost, reliability, or analyst value.
- Status: Accepted.

## Decision 007 — Repository Data Policy

- Date: 2026-09-14
- Decision: Do not upload full datasets, database volumes, secrets, or large model artifacts to GitHub.
- Status: Accepted.