# Phase 7C — Calibration and Policy Report

## Purpose

This report compares uncalibrated and calibrated fraud probabilities, documents
the validation-based calibration selection process, evaluates the constrained
review-capacity policy, and summarises policy sensitivity at different review
capacities.

All results are from the public IEEE-CIS Fraud Detection benchmark under
documented illustrative cost assumptions. Expected-value figures are simulated
policy results and must not be described as real financial-institution savings.

## Frozen Scope

| Item | Value |
| --- | --- |
| Model name | `xgboost` |
| Model version | `xgboost-v1.0.0` |
| MLflow training run | `a1f99cff34cf44728d3b6c9499ffeee3` |
| Selected calibration method | `sigmoid` |
| Primary calibration-selection metric | `brier_score` |
| Secondary calibration-selection metric | `expected_calibration_error` |
| Calibration-fit rows | `44,290` |
| Policy-selection rows | `44,291` |
| Locked operational review capacity | `1,000` |
| Currency for illustrative assumptions | `GBP` |
| Final holdout evaluation complete | `True` |

Calibration selection used chronological validation data only. The earliest
validation period fit the calibrators, and the later validation period selected
the calibration method and review policy. The final chronological holdout was
kept separate until the final evaluation.

## Calibration Comparison

The primary selection criterion was Brier score, with expected calibration error
(ECE) used as the secondary criterion. Lower values are better for both metrics.

![Calibration metric comparison](../figures/phase7c_calibration_metric_comparison.png)

| method | brier_score | expected_calibration_error | pr_auc | roc_auc | reliability_bin_count |
| --- | --- | --- | --- | --- | --- |
| sigmoid | 0.022892 | 0.003719 | 0.458580 | 0.897459 | 10 |
| isotonic | 0.022902 | 0.004235 | 0.441714 | 0.896277 | 9 |
| uncalibrated | 0.068533 | 0.161188 | 0.458580 | 0.897459 | 10 |

Sigmoid / Platt scaling was selected because it achieved the lowest Brier score
of `0.022892` and the lowest ECE of
`0.003719`. Compared with the
uncalibrated model, Brier score improved from
`0.068533` to
`0.022892`, while ECE improved from
`0.161188` to
`0.003719`.

Isotonic calibration was close on calibration metrics, with Brier score
`0.022902` and ECE
`0.004235`, but did not outperform
sigmoid calibration under the documented selection rule.

The PR-AUC for uncalibrated and sigmoid probabilities remained
`0.458580` because monotonic sigmoid calibration
preserves ranking. Isotonic calibration produced PR-AUC
`0.441714` in the saved validation comparison.

## Reliability Evidence

![Reliability curves](../figures/phase7c_calibration_reliability_curves.png)

The uncalibrated model was substantially overconfident in the highest recorded
reliability bin: mean predicted probability was
`0.683921`, while observed
fraud rate was `0.218785`.

For the sigmoid-calibrated output, the highest recorded bin had mean predicted
probability `0.231800` and
observed fraud rate `0.218785`.
This closer alignment supports the selection of sigmoid calibration for
probability-based decision interpretation.

## Capacity-Constrained Policy

The final policy is capacity constrained: transactions are prioritised by model
risk ranking, and the highest-ranked cases are selected until the configured
review capacity is reached. This is preferable to presenting a standalone fixed
probability threshold as an operational recommendation, because a threshold alone
does not guarantee that investigation demand stays within available capacity.

At the selected validation capacity of
`1,000`, the policy reviewed
`1,000` transactions, captured
`568` of
`1,449` fraud cases, and produced an
illustrative net expected value of GBP
`279,000.00`.

## Capacity Sensitivity

![Capacity capture curve](../figures/phase7c_capacity_capture_curve.png)

![Capacity expected value curve](../figures/phase7c_capacity_expected_value_curve.png)

| review_capacity | selected_review_count | captured_fraud_count | total_fraud_count | fraud_capture_rate | false_positive_review_count | total_expected_review_cost | total_expected_prevention_value | net_expected_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 500 | 392 | 1449 | 0.270531 | 108 | 2500.000000 | 196000.000000 | 193500.000000 |
| 1000 | 1000 | 568 | 1449 | 0.391994 | 432 | 5000.000000 | 284000.000000 | 279000.000000 |
| 1500 | 1500 | 655 | 1449 | 0.452036 | 845 | 7500.000000 | 327500.000000 | 320000.000000 |
| 2000 | 2000 | 731 | 1449 | 0.504486 | 1269 | 10000.000000 | 365500.000000 | 355500.000000 |

Increasing validation capacity from
`500` to
`2,000` increased fraud capture rate from
`0.270531` to
`0.504486`. It also increased
legitimate review burden from
`108` to
`1,269`.

The configured capacity of 1,000 is therefore a documented operational
assumption, not an arbitrary score cutoff. Phase 7D will frame these capacity
trade-offs in the final business-value summary.

## Validation and Holdout Results

| evaluation_period | review_capacity | selected_review_count | captured_fraud_count | total_fraud_count | fraud_capture_rate | false_positive_review_count | total_expected_review_cost | total_expected_prevention_value | net_expected_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chronological_validation_policy_selection | 1000 | 1000 | 568 | 1449 | 0.391994 | 432 | 5000.000000 | 284000.000000 | 279000.000000 |
| final_chronological_holdout | 1000 | 1000 | 870 | 3083 | 0.282193 | 130 | 5000.000000 | 435000.000000 | 430000.000000 |

On the validation policy-selection period, the 1,000-case review cohort captured
`568` of
`1,449` fraud cases
(`0.391994` capture rate).

On the untouched final chronological holdout, the same 1,000-case capacity
captured `870` of
`3,083` fraud cases
(`0.282193` capture rate).

The absolute illustrative expected-value figures are not directly comparable
between validation and holdout because the evaluation periods have different
numbers of transactions and fraud cases. Capture rate, review precision, review
volume, and cost assumptions provide the more meaningful capacity-normalised
comparison.

## Limitations

- Calibration quality was evaluated on chronological validation periods, but no
  calibration approach guarantees identical probability calibration in every
  future time period.
- Sigmoid calibration preserves ranking, which is useful for top-k review
  prioritisation, but calibration can still affect probability-based thresholds
  and cost estimates.
- The review capacity of 1,000 is a documented benchmark assumption rather than
  a real operational staffing limit.
- A capacity-constrained top-k cohort is the primary operational comparison.
  A threshold without a capacity control can create unpredictable review volume.
- Expected value uses illustrative GBP cost assumptions and should not be
  interpreted as realised savings, prevented losses, or an institutional forecast.
- Final-holdout results are one chronological benchmark evaluation and should not
  be used for further model, threshold, or calibrator selection.

## Output Inventory

```text
reports/figures/phase7c_calibration_metric_comparison.png
reports/figures/phase7c_calibration_reliability_curves.png
reports/figures/phase7c_capacity_capture_curve.png
reports/figures/phase7c_capacity_expected_value_curve.png
reports/tables/phase7c_calibration_comparison.csv
reports/tables/phase7c_calibration_policy_metadata.csv
reports/tables/phase7c_capacity_sensitivity.csv
reports/tables/phase7c_reliability_curve_data.csv
reports/tables/phase7c_validation_holdout_policy_comparison.csv
```
