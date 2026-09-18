# Financial Fraud Risk & Investigation Intelligence

A portfolio data-science project that evaluates **fraud-risk prioritisation** on the IEEE-CIS Fraud Detection benchmark. The project ranks transactions for limited investigation capacity, estimates calibrated fraud probabilities, explains model predictions with SHAP, and evaluates decision trade-offs under explicit illustrative cost assumptions.

> **Scope:** This is a reproducible public-data benchmark project—not a production fraud system. It does not include an API, deployment, monitoring, streaming, cloud infrastructure, or live financial-institution data.

## Project outcome

The selected benchmark approach is an **XGBoost classifier** (`xgboost-v1.0.0`) with sigmoid / Platt calibration and a constrained top-1,000 transaction review policy.

| Final chronological holdout metric | Result |
| --- | ---: |
| Review capacity | 1,000 transactions |
| Fraud-labelled transactions | 3,083 |
| Captured fraud | 870 |
| Fraud capture rate | 28.22% |
| Legitimate transactions reviewed | 130 |
| Review precision | 87.00% |
| Illustrative review cost | GBP 5,000.00 |
| Illustrative prevention value | GBP 435,000.00 |
| Illustrative net expected value | GBP 430,000.00 |

The financial values above are **illustrative benchmark calculations** based on documented assumptions. They are not realised savings, actual fraud losses, or a financial-institution forecast.

## Decision question

Rather than treating fraud detection as a simple classification problem, this project addresses four decision questions:

1. Which transactions should be prioritised when review capacity is limited?
2. How well calibrated are the model's fraud-risk probabilities?
3. Which model features contributed to a prioritised transaction's score?
4. What are the trade-offs between captured fraud, missed fraud, review volume, and illustrative decision value?

## Dataset

**Primary dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

The dataset includes transaction and identity tables joined only on `TransactionID`. It has severe class imbalance, missing values, high-cardinality categorical features, obfuscated variables, and temporal structure—all relevant challenges for fraud-risk modelling.

The repository does not contain full raw, interim, processed, feature, or MLflow artifact data. Raw source data must be downloaded separately from Kaggle and stored outside version control.

## Methodology

### Leakage-safe evaluation

- Chronological training, validation, and final holdout splits.
- The final holdout remained isolated until model family, feature set, calibration method, and policy were selected.
- Point-in-time behavioural features use historical transactions only.
- Same-timestamp records cannot enter one another's history.
- Target encoding, where used, is fitted only within appropriate chronological training folds.

### Modelling and calibration

- Logistic Regression baseline and XGBoost comparison using identical temporal splits.
- Evaluation prioritises PR-AUC, precision@k, recall@k, calibration quality, fraud capture, and expected decision value at capacity—not accuracy.
- Uncalibrated, sigmoid / Platt, and isotonic probability outputs were compared on a chronological validation calibration-selection period.
- Sigmoid / Platt scaling was selected:

| Validation calibration metric | Uncalibrated | Sigmoid / Platt | Isotonic |
| --- | ---: | ---: | ---: |
| Brier score | 0.068533 | **0.022892** | 0.022902 |
| Expected calibration error | 0.161188 | **0.003719** | 0.004235 |
| PR-AUC | 0.458580 | 0.458580 | 0.441714 |
| ROC-AUC | 0.897459 | 0.897459 | 0.896277 |

### Capacity-constrained decision policy

The final policy ranks transactions by fraud risk and selects the highest-ranked cases up to the documented capacity of 1,000 reviews. This avoids presenting an arbitrary standalone probability cutoff as an operational recommendation.

Validation capacity sensitivity demonstrates the trade-off between capture and review burden:

| Validation review capacity | Fraud captured | Fraud capture rate | Legitimate reviews | Illustrative net expected value |
| --- | ---: | ---: | ---: | ---: |
| 500 | 392 | 27.05% | 108 | GBP 193,500.00 |
| 1,000 | 568 | 39.20% | 432 | GBP 279,000.00 |
| 1,500 | 655 | 45.20% | 845 | GBP 320,000.00 |
| 2,000 | 731 | 50.45% | 1,269 | GBP 355,500.00 |

### Controlled advanced ablations

Anomaly-score, time-safe graph-feature, and compact sequence-model experiments were evaluated as controlled ablations under the same temporal and capacity-aware framework. The calibrated base XGBoost model remained the selected champion because the advanced alternatives did not provide a clear enough improvement in the documented decision-value comparison.

## Explainability and error analysis

### SHAP explainability

- Global SHAP importance was calculated on a deterministic 2,000-row final-holdout sample.
- The SHAP background used a deterministic 500-row sample from chronological training data only.
- The largest global mean-absolute SHAP contributor was `amount_global_24h_zscore`.
- Local explanations were created for a correctly ranked fraud case, correctly ranked legitimate case, missed fraud case, and unnecessary-review case.
- Top-10 local contributor stability was tested across deterministic SHAP background samples. Mean pairwise Jaccard similarity ranged from 0.878788 to 1.000000.

SHAP values describe how features contributed to the fitted model's predictions. They do **not** prove fraud, establish causality, or justify automated adverse action.

### Error analysis

At the locked final-holdout review capacity of 1,000:

- 2,213 fraud-labelled transactions were outside the review cohort.
- 130 legitimate transactions were selected for review.
- Fraud capture varied across chronological holdout periods from 14.46% to 41.58%.

These findings demonstrate why a single aggregate metric is insufficient: fraud prioritisation requires explicit consideration of review capacity, missed fraud, unnecessary investigations, and temporal variation.

## Repository structure

```text
financial-fraud-risk-intelligence/
├── configs/                         # Versioned data, model, calibration, and policy settings
├── data/
│   ├── external/                    # Dataset documentation only
│   └── sample/                      # Small recruiter-friendly sample only
├── docs/                            # Data card, leakage audit, feature catalogue, decisions
├── models/
│   ├── metrics/                     # Compact evaluation metadata and metrics
│   └── registry/                    # Model-registry documentation
├── notebooks/
│   ├── 10_shap_explainability.ipynb
│   ├── 11_error_analysis.ipynb
│   ├── 12_calibration_and_policy.ipynb
│   ├── 13_business_value.ipynb
│   └── 14_final_portfolio_summary.ipynb
├── reports/
│   ├── evaluation/                  # Final analytical reports
│   ├── figures/                     # Generated figures (local / ignored where configured)
│   └── tables/                      # Compact reproducible result tables
├── scripts/                         # Reproducible ingestion, feature, training, and evaluation scripts
├── src/fraud_intelligence/          # Reusable project code
└── tests/                           # Unit and integration-style tests
```

## Key reports

- [Final model selection and evaluation](reports/evaluation/final_model_selection_report.md)
- [Explainability report](reports/evaluation/explainability_report.md)
- [Error analysis report](reports/evaluation/error_analysis_report.md)
- [Calibration and policy report](reports/evaluation/calibration_and_policy_report.md)
- [Business value report](reports/evaluation/business_value_report.md)

## Reproducibility

### Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
```

Create a local `.env` file from `.env.example`. Do not commit `.env`, raw data, model artifacts, MLflow artifacts, or other sensitive/local files.

### Tests

```powershell
pytest -q
ruff check src tests
black --check src tests
```

The final development test suite passed 113 tests after the Phase 7A stability work was added.

## Limitations

- IEEE-CIS is a public benchmark with anonymised and obfuscated variables; it is not a live institutional fraud environment.
- The project does not measure human-investigator accuracy, customer impact, intervention effectiveness, or actual financial loss recovery.
- Cost assumptions, review capacity, and prevention value are illustrative and should be replaced by governed business inputs in a real setting.
- Model performance and calibration can change as fraud patterns and transaction populations change.
- SHAP is model attribution, not a causal explanation.
- The project intentionally ends at analytical portfolio deliverables; it does not implement production deployment or operational monitoring.

## Portfolio claim

> This project evaluates fraud prioritisation using leakage-safe chronological validation, point-in-time behavioural features, calibrated probabilities, explicit cost assumptions and constrained investigation capacity. It compares simple and advanced methods through controlled ablations, explains prioritised cases, analyses errors and reports expected decision value under documented benchmark assumptions.

## Licence

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).