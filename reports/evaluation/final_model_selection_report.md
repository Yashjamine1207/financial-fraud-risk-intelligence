# Final Model Selection and Evaluation Report

## Executive Summary

This portfolio project evaluates fraud prioritisation using leakage-safe
chronological validation, point-in-time behavioural features, calibrated
probabilities, explicit cost assumptions, and constrained investigation capacity.

The selected champion is the Phase 5 XGBoost model, version
`xgboost-v1.0.0`, with sigmoid / Platt probability calibration
and a fixed top-`1,000` review-capacity policy.
The final chronological holdout evaluation captured
`870` of
`3,083` fraud-labelled transactions
(`0.282193` capture rate).

This is a public IEEE-CIS benchmark evaluation. The results do not represent
real financial-institution performance, realised fraud savings, or a live
production decision system.

## Decision Objective

The system ranks transactions by fraud risk and supports constrained
approve/review/escalation reasoning under explicit benchmark assumptions.

The central question is not simply whether a transaction is predicted as fraud.
It is:

> Which transactions should receive limited investigation capacity, how confident
> is the model, what model features contributed to prioritisation, and what is the
> expected decision trade-off under documented assumptions?

## Selected Champion

| Item | Selected value |
| --- | --- |
| Model family | XGBoost classifier |
| Model version | `xgboost-v1.0.0` |
| MLflow run ID | `a1f99cff34cf44728d3b6c9499ffeee3` |
| Calibration method | Sigmoid / Platt scaling |
| Policy type | Capacity-constrained top-k review prioritisation |
| Review capacity | `1,000` transactions |
| Policy version | `threshold-policy-v1.0.0-holdout` |
| Final evaluation period | Untouched chronological holdout |

The XGBoost model was selected after comparison with Logistic Regression using
identical chronological splits and ranking-focused metrics. Calibration was
selected on chronological validation periods before the final holdout was
evaluated.

## Calibration Evidence

| Method | Brier score | Expected calibration error | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Uncalibrated | 0.068533 | 0.161188 | 0.458580 | 0.897459 |
| Sigmoid / Platt | 0.022892 | 0.003719 | 0.458580 | 0.897459 |
| Isotonic | 0.022902 | 0.004235 | 0.441714 | 0.896277 |

Sigmoid / Platt scaling was selected because it produced the lowest validation
Brier score and expected calibration error. It improved Brier score from
`0.068533` to
`0.022892` and ECE from
`0.161188` to
`0.003719`.

## Final Holdout Results

| Metric | Final holdout result |
| --- | ---: |
| Review capacity | 1,000 |
| Selected review count | 1,000 |
| Total fraud-labelled transactions | 3,083 |
| Captured fraud | 870 |
| Missed fraud | 2,213 |
| Fraud capture rate | 0.282193 |
| Legitimate transactions reviewed | 130 |
| Review precision | 0.870000 |
| Illustrative review cost | GBP 5,000.00 |
| Illustrative prevention value | GBP 435,000.00 |
| Illustrative net expected value | GBP 430,000.00 |

The `GBP 1,106,500.00`
residual missed-fraud figure is reported separately as illustrative exposure under
the documented false-negative cost assumption. It is not automatically subtracted
from the saved Phase 5 net-expected-value definition.

## Phase 6 Ablations

Advanced methods were evaluated as controlled ablations using the same temporal
evaluation principles and review-capacity framing. They were not retained as the
champion unless they showed clear measurable decision value.

| experiment | review_capacity | baseline_captured_fraud | candidate_captured_fraud | captured_fraud_difference | baseline_net_expected_value | candidate_net_expected_value | net_expected_value_difference | selection_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Isolation Forest anomaly-score feature | 1000 | 568 | 567 | -1 | 279,000.000000 | 278,500.000000 | -500.000000 | Rejected as champion replacement |
| Time-safe graph features | 1000 | 568 | 564 | -4 | 279,000.000000 | 277,000.000000 | -2,000.000000 | Rejected as champion replacement |
| Compact LSTM sequence model | 1000 | 568 | 121 | -447 | 279,000.000000 | 55,500.000000 | -223,500.000000 | Rejected as champion replacement |

## Explainability and Error Analysis

Global SHAP analysis found `amount_global_24h_zscore`
as the largest mean-absolute contributor within the deterministic final-holdout
explanation sample. SHAP values describe how the fitted model features contributed
to a prediction; they do not prove fraud or establish causal drivers.

The top-10 local contributor sets were stable across the tested deterministic SHAP
background samples, with mean pairwise Jaccard similarity ranging from
`0.878788` to `1.000000`.

At the fixed review capacity, Phase 7B identified
`2,213` fraud-labelled transactions
outside the review cohort and
`130` legitimate
transactions inside it. Chronological fraud-capture variation confirms that a
single final result should be interpreted with temporal caution.

## Limitations

- The IEEE-CIS dataset is public benchmark data with obfuscated variables,
  missingness, and limited business semantics.
- The analysis uses chronological splits, but benchmark performance cannot
  guarantee performance in other institutions, periods, or fraud environments.
- Calibration, costs, prevention value, and review capacity are documented
  benchmark assumptions rather than real operational policy inputs.
- SHAP provides model attribution only; it does not establish causality, confirm
  fraud, or justify automated adverse action.
- The review policy is capacity constrained. Different capacity, investigation
  quality, customer-friction costs, or intervention effectiveness would change
  the resulting trade-offs.
- No API, streaming, monitoring, cloud deployment, or production service is
  included. This repository is a reproducible analytical portfolio project.

## Portfolio Claim

> This project evaluates fraud prioritisation using leakage-safe chronological
> validation, point-in-time behavioural features, calibrated probabilities,
> explicit cost assumptions and constrained investigation capacity. It compares
> simple and advanced methods through controlled ablations, explains prioritised
> cases, analyses errors and reports expected decision value under documented
> benchmark assumptions.

## Supporting Reports

- `reports/evaluation/explainability_report.md`
- `reports/evaluation/error_analysis_report.md`
- `reports/evaluation/calibration_and_policy_report.md`
- `reports/evaluation/business_value_report.md`
- `reports/tables/phase7e_advanced_ablation_summary.csv`
