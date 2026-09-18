# Phase 7D — Business Value Report

## Purpose

This report translates the frozen fraud-prioritisation model's final-holdout
performance into review volume, fraud capture, missed fraud, illustrative review
cost, illustrative prevention value, and constrained-capacity trade-offs.

All monetary figures are benchmark calculations under documented illustrative
assumptions. They are not realised savings, actual fraud losses, institutional
forecasts, or financial advice.

## Frozen Evaluation Scope

| Item | Value |
| --- | --- |
| Model name | `xgboost` |
| Model version | `xgboost-v1.0.0` |
| MLflow training run | `a1f99cff34cf44728d3b6c9499ffeee3` |
| Evaluation period | Final chronological holdout |
| Review capacity | `1,000` |
| Currency | `GBP` |
| Manual review cost per reviewed transaction | `GBP 5.00` |
| Fraud prevention value per captured fraud case | `GBP 500.00` |
| False-negative cost per missed fraud case | `GBP 500.00` |

The model, calibration method, and capacity policy were selected using earlier
chronological validation evidence. The final holdout is reported here as the
locked benchmark evaluation, not used for additional model or policy selection.

## Final Holdout Outcome

At the locked review capacity of `1,000`, the policy reviewed
`1,000` transactions and captured
`870` of `3,083` fraud-labelled transactions.
This corresponds to a fraud capture rate of `0.282193` and review
precision of `0.870000`.

- Captured fraud: `870`
- Missed fraud: `2,213`
- Legitimate transactions reviewed: `130`
- Total review volume: `1,000`

| review_capacity | reviewed_transaction_count | total_fraud_count | captured_fraud_count | missed_fraud_count | unnecessary_review_count | fraud_capture_rate | review_precision | total_expected_review_cost | total_expected_prevention_value | net_expected_value | illustrative_residual_missed_fraud_exposure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 1000 | 3083 | 870 | 2213 | 130 | 0.282193 | 0.870000 | 5,000.000000 | 435,000.000000 | 430,000.000000 | 1,106,500.000000 |

## Illustrative Value Components

![Final holdout value components](../figures/phase7d_holdout_value_components.png)

Under the documented assumptions, the `1,000` reviews
have an illustrative manual review cost of `GBP 5,000.00`.
The `870` captured fraud cases correspond to illustrative
prevention value of `GBP 435,000.00`, producing
illustrative net expected value of `GBP 430,000.00`.

The `2,213` missed fraud-labelled cases correspond to an
illustrative residual missed-fraud exposure of
`GBP 1,106,500.00` if every missed
case incurs the documented false-negative cost. This figure is shown separately:
it is an exposure indicator, not an amount subtracted from the saved Phase 5 net
expected value calculation.

## Capacity Trade-offs

![Capacity value trade-off](../figures/phase7d_capacity_value_tradeoff.png)

| review_capacity | selected_review_count | captured_fraud_count | total_fraud_count | fraud_capture_rate | false_positive_review_count | total_expected_review_cost | total_expected_prevention_value | net_expected_value | illustrative_residual_missed_fraud_exposure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 500 | 392 | 1449 | 0.270531 | 108 | 2,500.000000 | 196,000.000000 | 193,500.000000 | 528,500.000000 |
| 1000 | 1000 | 568 | 1449 | 0.391994 | 432 | 5,000.000000 | 284,000.000000 | 279,000.000000 | 440,500.000000 |
| 1500 | 1500 | 655 | 1449 | 0.452036 | 845 | 7,500.000000 | 327,500.000000 | 320,000.000000 | 397,000.000000 |
| 2000 | 2000 | 731 | 1449 | 0.504486 | 1269 | 10,000.000000 | 365,500.000000 | 355,500.000000 | 359,000.000000 |

Increasing validation review capacity from `500`
to `2,000` increased fraud capture rate from
`0.270531` to
`0.504486`. It also increased review
volume, manual review costs, and legitimate-review burden.

The configured 1,000-case capacity is a documented benchmark assumption. It
illustrates the operational trade-off between review capacity, fraud capture, and
review burden; it is not a recommendation for any real financial institution.

## Validation Versus Holdout

| evaluation_period | review_capacity | selected_review_count | captured_fraud_count | total_fraud_count | fraud_capture_rate | review_precision | false_positive_review_count | net_expected_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chronological_validation_policy_selection | 1000 | 1000 | 568 | 1449 | 0.391994 | 0.568000 | 432 | 279,000.000000 |
| final_chronological_holdout | 1000 | 1000 | 870 | 3083 | 0.282193 | 0.870000 | 130 | 430,000.000000 |

The validation and holdout periods contain different transaction and fraud volumes.
Therefore, absolute illustrative net expected value should not be compared as
though the periods were identical. More comparable indicators are capacity,
review volume, fraud capture rate, review precision, and the documented cost
assumptions.

## Limitations

- The IEEE-CIS dataset is a public benchmark with obfuscated variables and does
  not represent a live financial-institution operating environment.
- Review cost, prevention value, and false-negative cost are illustrative
  assumptions documented for portfolio decision modelling.
- Captured fraud labels are benchmark outcomes; they do not represent prevented
  loss or confirmed operational intervention.
- The residual missed-fraud exposure estimate assumes the same false-negative cost
  for every missed fraud case and should not be interpreted as actual loss.
- The selected policy reflects a fixed capacity constraint. A different review
  capacity, investigation quality, customer-friction cost, or intervention
  effectiveness would change the decision value.
- These figures are not causal estimates and must not be presented as realised
  savings or institutional performance.

## Output Inventory

```text
reports/figures/phase7d_holdout_value_components.png
reports/figures/phase7d_capacity_value_tradeoff.png
reports/tables/phase7d_business_value_summary.csv
reports/tables/phase7d_capacity_business_value.csv
reports/tables/phase7d_policy_value_components.csv
reports/tables/phase7d_validation_holdout_value_comparison.csv
reports/evaluation/business_value_report.md
```
