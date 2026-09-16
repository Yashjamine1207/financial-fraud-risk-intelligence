# Phase 3 Statistical Evidence Report

## Scope

This report evaluates selected associations between point-in-time-safe
behavioural features and the `isFraud` label in the IEEE-CIS training
period only. Validation and final-test data were not used.

- Training transactions analysed: 413,378
- Overall training fraud rate: 3.52%
- Confidence interval method: Wilson 95% confidence interval
- Group comparison: two-sided two-proportion z-test

## Velocity definition

High card velocity is defined using the 95th percentile of the
`velocity_card1_60min` distribution in the training period.

- High-velocity threshold: `velocity_card1_60min >= 6`

## Results

| Feature comparison | Group A fraud rate (95% CI) | Group B fraud rate (95% CI) | Difference | Relative risk | p-value |
|---|---:|---:|---:|---:|---:|
| New card1 versus previously seen card1 | 2.30% (2.05%–2.58%) | 3.55% (3.50%–3.61%) | -1.25% | 0.65 | 0.000000 |
| New device versus previously seen device | 5.63% (4.58%–6.89%) | 6.81% (6.64%–6.97%) | -1.18% | 0.83 | 0.067452 |
| High card1 60-minute velocity versus lower card1 velocity | 3.44% (3.21%–3.68%) | 3.52% (3.46%–3.58%) | -0.08% | 0.98 | 0.514511 |

## Interpretation limits

- These results show statistical association in an anonymised public fraud benchmark dataset.
- A higher observed fraud rate does not prove that a new card, device, or high velocity caused fraud.
- The features may correlate with unobserved factors, collection processes, product mix, or other variables.
- Results from this analysis must not be treated as financial-institution production outcomes.
- The final test period remains locked and has not been used in this Phase 3 analysis.
