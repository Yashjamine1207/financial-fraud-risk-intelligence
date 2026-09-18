# Phase 7A — Explainability Report

## Purpose

This report documents global and local SHAP explanations for the locked fraud-risk
champion model. The analysis evaluates which transformed input features contributed
to the model's predictions, illustrates representative correct and incorrect
ranking outcomes, and checks local attribution stability when the SHAP background
reference sample changes.

SHAP values describe how features contributed to this model's prediction. They do
not prove fraud, establish causation, or demonstrate that a transaction attribute
independently caused risk.

## Frozen Analysis Scope

| Item | Value |
| --- | --- |
| Model name | `xgboost` |
| Model version | `xgboost-v1.0.0` |
| MLflow training run | `a1f99cff34cf44728d3b6c9499ffeee3` |
| Preprocessor artifact | `runs:/a1f99cff34cf44728d3b6c9499ffeee3/xgboost_preprocessor` |
| Classifier artifact | `runs:/a1f99cff34cf44728d3b6c9499ffeee3/xgboost_classifier` |
| Probability calibration selected in Phase 5 | `sigmoid` |
| SHAP output scale | `XGBoost raw margin` |
| SHAP background data | `chronological training split only` |
| Background sample size | `500` |
| Global explanation sample | `deterministic random final-holdout sample` |
| Global explanation sample size | `2,000` |
| Local representative cases | `4` |
| Stability background seeds | `42 | 52 | 62` |
| Stability top-feature count | `10` |

The XGBoost classifier and its fitted preprocessing pipeline were loaded from the
frozen Phase 5 artifacts. No model fitting, feature fitting, threshold selection,
calibration fitting, or policy optimisation occurred in this notebook.

The SHAP background sample was selected deterministically from chronological
training data only. The global explanation sample and representative explanations
use the final holdout after the model family, features, calibration method, and
decision policy had already been locked.

## Global Feature Contributions

Global importance is calculated as the mean absolute SHAP value across a
deterministic sample of `2,000`
final-holdout transactions. A higher value means that the transformed feature
moved the frozen model's raw prediction more on average within this explained
sample. It does not imply that the feature caused fraud.

![Global SHAP importance](../figures/phase7a_global_shap_importance.png)

### Top 10 Global Contributors

| rank | transformed_feature | mean_absolute_shap_value |
| --- | --- | --- |
| 1 | amount_global_24h_zscore | 0.169216 |
| 2 | C1 | 0.169206 |
| 3 | amount_global_168h_zscore | 0.158346 |
| 4 | C13 | 0.155574 |
| 5 | card1 | 0.148949 |
| 6 | C5 | 0.130318 |
| 7 | V70 | 0.130127 |
| 8 | D1 | 0.129612 |
| 9 | card6_target_encoded | 0.127404 |
| 10 | C14 | 0.126739 |

The full ranking is saved in:

```text
reports/tables/phase7a_global_shap_importance.csv
```

## Representative Local Explanations

The four cases below are selected to show both correct ranking behaviour and
important error patterns. The displayed risk values are raw frozen-XGBoost
probabilities for explanation context. Phase 5's sigmoid calibration remains the
approved probability-calibration approach for operational policy evaluation.

| case_group | holdout_row_position | true_label | raw_classifier_probability | transaction_timestamp |
| --- | --- | --- | --- | --- |
| correctly_ranked_fraud | 33672 | 1 | 0.999690 | 14096183 |
| correctly_ranked_legitimate | 37502 | 0 | 0.000211 | 14229044 |
| missed_fraud_low_score | 3563 | 1 | 0.004672 | 13227323 |
| unnecessary_review_high_score | 80686 | 0 | 0.999426 | 15557752 |

### Correctly Ranked Fraud

A fraud-labelled transaction with the highest raw XGBoost risk score among final-holdout fraud cases. This is a representative correct high-risk ranking, not proof that any contributor caused fraud.

- Holdout row position: `33672`
- True label: `1`
- Raw frozen-XGBoost probability: `0.999690`
- Transaction timestamp value: `14096183`

Top absolute SHAP contributors:

| transformed_feature | shap_value | absolute_shap_value |
| --- | --- | --- |
| C1 | +1.519251 | 1.519251 |
| V258 | +0.830042 | 0.830042 |
| V45 | -0.528838 | 0.528838 |
| C2 | +0.456197 | 0.456197 |
| V87 | +0.339012 | 0.339012 |
### Correctly Ranked Legitimate

A legitimate transaction with the lowest raw XGBoost risk score among final-holdout legitimate cases. This is a representative correct low-risk ranking.

- Holdout row position: `37502`
- True label: `0`
- Raw frozen-XGBoost probability: `0.000211`
- Transaction timestamp value: `14229044`

Top absolute SHAP contributors:

| transformed_feature | shap_value | absolute_shap_value |
| --- | --- | --- |
| V258 | -0.526139 | 0.526139 |
| D13 | +0.484724 | 0.484724 |
| history_DeviceInfo_transaction_count | +0.364697 | 0.364697 |
| history_addr1_transaction_count | -0.355139 | 0.355139 |
| V285 | -0.346966 | 0.346966 |
### Missed Fraud Low Score

A fraud-labelled transaction with the lowest raw XGBoost risk score among final-holdout fraud cases. It illustrates an important missed-fraud / low-priority error pattern for later Phase 7B analysis.

- Holdout row position: `3563`
- True label: `1`
- Raw frozen-XGBoost probability: `0.004672`
- Transaction timestamp value: `13227323`

Top absolute SHAP contributors:

| transformed_feature | shap_value | absolute_shap_value |
| --- | --- | --- |
| C14 | -0.630320 | 0.630320 |
| C1 | +0.593431 | 0.593431 |
| C13 | -0.468226 | 0.468226 |
| amount_global_24h_zscore | -0.389685 | 0.389685 |
| V70 | -0.290656 | 0.290656 |
### Unnecessary Review High Score

A legitimate transaction with the highest raw XGBoost risk score among final-holdout legitimate cases. It illustrates an unnecessary-review / false-positive error pattern for later Phase 7B analysis.

- Holdout row position: `80686`
- True label: `0`
- Raw frozen-XGBoost probability: `0.999426`
- Transaction timestamp value: `15557752`

Top absolute SHAP contributors:

| transformed_feature | shap_value | absolute_shap_value |
| --- | --- | --- |
| V258 | +0.789582 | 0.789582 |
| C14 | +0.530572 | 0.530572 |
| C13 | +0.481307 | 0.481307 |
| C1 | +0.468417 | 0.468417 |
| history_R_emaildomain_transaction_count | +0.382567 | 0.382567 |


Local explanation figures are stored in:

```text
reports/figures/phase7a_local_shap_correctly_ranked_fraud.png
reports/figures/phase7a_local_shap_correctly_ranked_legitimate.png
reports/figures/phase7a_local_shap_missed_fraud_low_score.png
reports/figures/phase7a_local_shap_unnecessary_review_high_score.png
```

## Attribution Stability

Stability was evaluated by recalculating local SHAP values for the same four
representative cases using deterministic training-background samples with seeds
`42 | 52 | 62`. For each pair of background samples,
the top `10` features ranked by
absolute SHAP contribution were compared using Jaccard similarity.

A Jaccard similarity of `1.0` means the two runs selected the same top-feature
set. Lower values indicate that the leading transformed contributors varied when
the background reference sample changed.

| case_group | seed_pair_count | mean_jaccard_similarity | min_jaccard_similarity | max_jaccard_similarity | mean_shared_top_feature_count |
| --- | --- | --- | --- | --- | --- |
| correctly_ranked_fraud | 3 | 1.000000 | 1.000000 | 1.000000 | 10.000000 |
| correctly_ranked_legitimate | 3 | 1.000000 | 1.000000 | 1.000000 | 10.000000 |
| missed_fraud_low_score | 3 | 0.878788 | 0.818182 | 1.000000 | 9.333333 |
| unnecessary_review_high_score | 3 | 0.878788 | 0.818182 | 1.000000 | 9.333333 |

The complete pairwise results are stored in:

```text
reports/tables/phase7a_shap_stability_pairwise.csv
reports/tables/phase7a_shap_stability_summary.csv
```

## Limitations

- SHAP explains the fitted model's behaviour, not causal fraud mechanisms.
- The global ranking uses a deterministic final-holdout sample rather than every
  holdout transaction to keep the analysis computationally practical.
- Feature names are transformed preprocessor outputs; categorical variables may
  appear as one-hot-encoded levels rather than as a single raw business feature.
- SHAP values were calculated on the XGBoost raw-margin scale because this is the
  additive scale supported by the interventional TreeExplainer configuration.
- Representative cases illustrate model ranking behaviour and error patterns;
  they are not estimates of the prevalence of each error type.
- Attribution stability tests sensitivity to the selected SHAP background
  reference sample only. It does not demonstrate causal stability, fairness,
  temporal robustness, or generalisation beyond the public IEEE-CIS benchmark.
- All conclusions are benchmark findings under the project’s documented
  assumptions and must not be represented as results from a real financial
  institution.

## Output Inventory

```text
reports/figures/phase7a_global_shap_importance.png
reports/figures/phase7a_local_shap_correctly_ranked_fraud.png
reports/figures/phase7a_local_shap_correctly_ranked_legitimate.png
reports/figures/phase7a_local_shap_missed_fraud_low_score.png
reports/figures/phase7a_local_shap_unnecessary_review_high_score.png
reports/tables/phase7a_explainability_metadata.csv
reports/tables/phase7a_global_shap_importance.csv
reports/tables/phase7a_local_shap_contributors.csv
reports/tables/phase7a_representative_cases.csv
reports/tables/phase7a_shap_stability_pairwise.csv
reports/tables/phase7a_shap_stability_summary.csv
```
