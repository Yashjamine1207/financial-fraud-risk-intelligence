# Phase 7B — Error Analysis Report

## Purpose

This report analyses final-holdout ranking and review errors for the frozen Phase 5
XGBoost champion model. It focuses on fraud cases missed outside the constrained
review cohort, legitimate transactions unnecessarily selected for review, score
overlap between fraud and legitimate transactions, and variation in results across
chronological holdout periods.

This is a public IEEE-CIS benchmark analysis under documented policy assumptions.
It must not be represented as a real financial-institution outcome, realised
savings estimate, or causal assessment of fraud behaviour.

## Locked Evaluation Scope

| Item | Value |
| --- | --- |
| Model name | `xgboost` |
| Model version | `xgboost-v1.0.0` |
| MLflow training run | `a1f99cff34cf44728d3b6c9499ffeee3` |
| Final-holdout transaction count | `88,581` |
| Locked review capacity | `1,000` |
| Selected review count | `1,000` |
| Review-cohort selection basis | Top-k model-score ranking at fixed capacity |
| Calibration selected in Phase 5 | Sigmoid / Platt scaling |

The analysis loads the frozen final model and the untouched chronological final
holdout. No model fitting, hyperparameter tuning, calibration fitting, threshold
selection, or policy optimisation occurred in this Phase 7B notebook.

The review cohort is reconstructed as the highest-ranked
`1,000` holdout transactions. Sigmoid calibration is
monotonic, so it preserves the classifier ranking used to construct this top-k
review cohort.

## Review Outcome Summary

At the locked review capacity, the model selected
`1,000` transactions for review. It captured
`870` of
`3,083` fraud-labelled transactions, producing a fraud
capture rate of `0.282193` and review precision of
`0.870000`.

- Captured fraud reviewed: `870`
- Missed fraud not reviewed: `2,213`
- Unnecessary legitimate reviews: `130`
- Correctly low-risk legitimate transactions:
  `85,368`

![Error category counts](../figures/phase7b_error_category_counts.png)

| error_category | transaction_count | fraud_count | transaction_share | mean_raw_score | median_raw_score | mean_transaction_amount | median_transaction_amount |
| --- | --- | --- | --- | --- | --- | --- | --- |
| captured_fraud_reviewed | 870 | 870 | 0.009822 | 0.989410 | 0.993778 | 87.24820114942528 | 40.7 |
| missed_fraud_not_reviewed | 2213 | 2213 | 0.024983 | 0.544224 | 0.560300 | 177.90446723904202 | 77.0 |
| unnecessary_review_legitimate | 130 | 0 | 0.001468 | 0.982973 | 0.983874 | 61.27389230769231 | 30.494 |
| correctly_low_risk_legitimate | 85368 | 0 | 0.963728 | 0.182044 | 0.119020 | 136.7161004006185 | 68.95 |

## Missed Fraud

The final holdout contains
`2,213` missed fraud cases outside the
top-`1,000` review cohort. These cases have a mean raw
classifier score of `0.544224` and a median
raw classifier score of `0.560300`.

This result does not mean that the missed transactions were safe or non-fraudulent.
It shows that, under the fixed review-capacity constraint, their model scores did
not rank highly enough to enter the limited review cohort. Phase 7C will compare
capacity and decision-policy alternatives; this report only documents the error
population under the locked evaluation policy.

## Unnecessary Reviews

The model selected
`130` legitimate transactions for
review. These cases are false-positive review burden at the fixed capacity, not
evidence that the transactions were fraudulent.

Their mean raw classifier score was
`0.982973`, compared with
`0.989410` for captured fraud. This score
overlap is the practical ranking trade-off: some legitimate transactions have
patterns that the frozen model associates with higher fraud risk, while some fraud
transactions receive comparatively low scores.

## Score Distributions

![Score distributions by true label](../figures/phase7b_score_distribution_by_label.png)

| label_name | transaction_count | mean_raw_score | median_raw_score | p05_raw_score | p25_raw_score | p75_raw_score | p95_raw_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fraud | 3083 | 0.669852 | 0.740134 | 0.121673 | 0.402810 | 0.973497 | 0.998096 |
| legitimate | 85498 | 0.183262 | 0.119285 | 0.016218 | 0.055839 | 0.239904 | 0.598763 |

The score distributions should be interpreted as ranking evidence. Separation
between fraud and legitimate score distributions supports prioritisation, while
their overlap explains why both missed fraud and unnecessary reviews remain under
limited investigation capacity.

## Score-Decile Analysis

| score_decile | transaction_count | fraud_count | fraud_rate | reviewed_count | captured_fraud_count | mean_raw_score |
| --- | --- | --- | --- | --- | --- | --- |
| score_decile_1 | 8859 | 19 | 0.002145 | 0 | 0 | 0.016057 |
| score_decile_2 | 8858 | 29 | 0.003274 | 0 | 0 | 0.037361 |
| score_decile_3 | 8858 | 31 | 0.003500 | 0 | 0 | 0.057616 |
| score_decile_4 | 8858 | 27 | 0.003048 | 0 | 0 | 0.080632 |
| score_decile_5 | 8858 | 58 | 0.006548 | 0 | 0 | 0.108596 |
| score_decile_6 | 8858 | 65 | 0.007338 | 0 | 0 | 0.143376 |
| score_decile_7 | 8857 | 109 | 0.012307 | 0 | 0 | 0.189732 |
| score_decile_8 | 8859 | 214 | 0.024156 | 0 | 0 | 0.260856 |
| score_decile_9 | 8858 | 443 | 0.050011 | 0 | 0 | 0.395016 |
| score_decile_10 | 8858 | 2088 | 0.235719 | 1000 | 870 | 0.712741 |

The highest score deciles contain the constrained review cohort. Lower score
deciles still contain fraud-labelled transactions, illustrating the residual-risk
trade-off created by limited review capacity rather than a claim that lower-scored
transactions are definitively legitimate.

## Chronological Variation

![Chronological error variation](../figures/phase7b_chronological_error_variation.png)

| chronological_decile | transaction_count | fraud_count | fraud_rate | reviewed_count | captured_fraud_count | missed_fraud_count | unnecessary_review_count | fraud_capture_rate | mean_raw_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_period_1 | 8859 | 271 | 0.030590 | 98 | 94 | 177 | 4 | 0.346863 | 0.181405 |
| time_period_2 | 8858 | 258 | 0.029126 | 76 | 71 | 187 | 5 | 0.275194 | 0.203182 |
| time_period_3 | 8858 | 269 | 0.030368 | 75 | 65 | 204 | 10 | 0.241636 | 0.196388 |
| time_period_4 | 8858 | 303 | 0.034206 | 163 | 126 | 177 | 37 | 0.415842 | 0.221612 |
| time_period_5 | 8858 | 249 | 0.028110 | 46 | 36 | 213 | 10 | 0.144578 | 0.190969 |
| time_period_6 | 8858 | 347 | 0.039174 | 81 | 77 | 270 | 4 | 0.221902 | 0.205024 |
| time_period_7 | 8857 | 285 | 0.032178 | 85 | 73 | 212 | 12 | 0.256140 | 0.178980 |
| time_period_8 | 8859 | 354 | 0.039959 | 133 | 118 | 236 | 15 | 0.333333 | 0.207885 |
| time_period_9 | 8858 | 390 | 0.044028 | 135 | 125 | 265 | 10 | 0.320513 | 0.210364 |
| time_period_10 | 8858 | 357 | 0.040303 | 108 | 85 | 272 | 23 | 0.238095 | 0.206161 |

Across the ten chronological final-holdout periods, fraud capture rate ranged from
`0.144578` in
`time_period_5` to
`0.415842` in
`time_period_4`.

This temporal variation is descriptive benchmark evidence. It may reflect changing
transaction composition, fraud prevalence, feature distributions, or model-ranking
difficulty. It does not establish a causal explanation.

## Representative Error Cases

The following records are limited examples selected from the two principal error
populations. They support qualitative inspection but are not population-level
estimates.

| error_category | holdout_row_position | transaction_timestamp | true_label | raw_classifier_probability | reviewed_at_capacity | transaction_amount |
| --- | --- | --- | --- | --- | --- | --- |
| missed_fraud_not_reviewed | 74474 | 15335752 | 1 | 0.960998 | False | 35.000000 |
| missed_fraud_not_reviewed | 46893 | 14495759 | 1 | 0.960842 | False | 15.862000 |
| missed_fraud_not_reviewed | 85673 | 15722009 | 1 | 0.960777 | False | 8.115000 |
| missed_fraud_not_reviewed | 32258 | 14070660 | 1 | 0.960725 | False | 39.962000 |
| missed_fraud_not_reviewed | 64779 | 15023573 | 1 | 0.960570 | False | 9.283000 |
| missed_fraud_not_reviewed | 50102 | 14586942 | 1 | 0.960542 | False | 80.416000 |
| missed_fraud_not_reviewed | 8181 | 13364530 | 1 | 0.960515 | False | 69.595000 |
| missed_fraud_not_reviewed | 56507 | 14775011 | 1 | 0.960274 | False | 250.000000 |
| missed_fraud_not_reviewed | 70895 | 15208003 | 1 | 0.960014 | False | 10.943000 |
| missed_fraud_not_reviewed | 64943 | 15026944 | 1 | 0.959974 | False | 8.115000 |
| missed_fraud_not_reviewed | 73042 | 15279772 | 1 | 0.959762 | False | 33.199000 |
| missed_fraud_not_reviewed | 56914 | 14788521 | 1 | 0.959413 | False | 50.000000 |
| missed_fraud_not_reviewed | 21223 | 13736757 | 1 | 0.959265 | False | 83.490000 |
| missed_fraud_not_reviewed | 74826 | 15348039 | 1 | 0.959143 | False | 18.198000 |
| missed_fraud_not_reviewed | 64760 | 15023082 | 1 | 0.959124 | False | 8.915000 |
| missed_fraud_not_reviewed | 62013 | 14939574 | 1 | 0.958860 | False | 55.639000 |
| missed_fraud_not_reviewed | 42121 | 14344888 | 1 | 0.958853 | False | 8.361000 |
| missed_fraud_not_reviewed | 49367 | 14574441 | 1 | 0.958836 | False | 200.000000 |
| missed_fraud_not_reviewed | 80557 | 15554437 | 1 | 0.958746 | False | 101.626000 |
| missed_fraud_not_reviewed | 86307 | 15738742 | 1 | 0.958483 | False | 200.000000 |
| unnecessary_review_legitimate | 80686 | 15557752 | 0 | 0.999426 | True | 130.645000 |
| unnecessary_review_legitimate | 80383 | 15550017 | 0 | 0.999341 | True | 78.387000 |
| unnecessary_review_legitimate | 80740 | 15559262 | 0 | 0.998842 | True | 130.645000 |
| unnecessary_review_legitimate | 80723 | 15558632 | 0 | 0.998690 | True | 130.645000 |
| unnecessary_review_legitimate | 76276 | 15380826 | 0 | 0.998160 | True | 86.687000 |
| unnecessary_review_legitimate | 63083 | 14962372 | 0 | 0.998154 | True | 81.092000 |
| unnecessary_review_legitimate | 27669 | 13947981 | 0 | 0.997991 | True | 15.247000 |
| unnecessary_review_legitimate | 79859 | 15537444 | 0 | 0.997954 | True | 22.625000 |
| unnecessary_review_legitimate | 34906 | 14150187 | 0 | 0.997935 | True | 20.903000 |
| unnecessary_review_legitimate | 27668 | 13947916 | 0 | 0.997734 | True | 15.247000 |
| unnecessary_review_legitimate | 80790 | 15561030 | 0 | 0.997517 | True | 128.001000 |
| unnecessary_review_legitimate | 34917 | 14150359 | 0 | 0.997451 | True | 20.903000 |
| unnecessary_review_legitimate | 63082 | 14962276 | 0 | 0.997429 | True | 81.092000 |
| unnecessary_review_legitimate | 34902 | 14150067 | 0 | 0.996778 | True | 20.903000 |
| unnecessary_review_legitimate | 27663 | 13947625 | 0 | 0.996604 | True | 15.247000 |
| unnecessary_review_legitimate | 27661 | 13947392 | 0 | 0.996548 | True | 15.247000 |
| unnecessary_review_legitimate | 79583 | 15529915 | 0 | 0.996397 | True | 22.625000 |
| unnecessary_review_legitimate | 34911 | 14150271 | 0 | 0.996263 | True | 20.903000 |
| unnecessary_review_legitimate | 85879 | 15725795 | 0 | 0.996213 | True | 15.493000 |
| unnecessary_review_legitimate | 76274 | 15380783 | 0 | 0.996035 | True | 86.687000 |

## Limitations

- This analysis uses the public IEEE-CIS Fraud Detection benchmark and documented
  illustrative decision assumptions; it is not real banking or card-network data.
- A fixed top-k review capacity constrains fraud capture by design. Missed fraud
  cases are not necessarily low risk; they were simply outside the selected review
  cohort at the locked capacity.
- The analysis uses model score ranking to reconstruct the fixed-size review cohort.
  It does not re-optimise the threshold, capacity, calibration method, or action
  policy on the holdout.
- Raw classifier probabilities are shown for ranking/error-analysis context. The
  approved confidence-calibration method remains Phase 5 sigmoid / Platt scaling.
- Score distributions and chronological summaries identify associations and
  operational trade-offs, not causal drivers of fraud.
- The transaction-level error-case export is limited to representative records and
  does not replace the aggregate summaries used for portfolio conclusions.

## Output Inventory

```text
reports/figures/phase7b_error_category_counts.png
reports/figures/phase7b_score_distribution_by_label.png
reports/figures/phase7b_chronological_error_variation.png
reports/tables/phase7b_error_category_summary.csv
reports/tables/phase7b_policy_error_summary.csv
reports/tables/phase7b_score_distribution_summary.csv
reports/tables/phase7b_score_decile_summary.csv
reports/tables/phase7b_chronological_error_summary.csv
reports/tables/phase7b_representative_error_cases.csv
```
