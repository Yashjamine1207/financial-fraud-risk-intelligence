# Phase 6 — Time-Safe Graph-Feature Ablation Plan

## Objective

Test whether historical graph relationships between card proxies, device
identifiers, and recipient email domains improve fraud prioritisation beyond the
selected Phase 5 XGBoost baseline.

The graph experiment is a controlled ablation. It will be retained only if it
improves the documented decision objective at the configured operational review
capacity of 1,000 transactions.

## Candidate graph

The graph is an undirected entity-relationship graph built from earlier
transactions only.

Each available transaction can add either or both of these relationships:

```text
card1 ↔ DeviceInfo
card1 ↔ R_emaildomain
```

Node prefixes distinguish entity families:

```text
card1::<value>
device::<value>
remail::<value>
```

For example:

```text
card1::1234 ↔ device::Windows
card1::1234 ↔ remail::gmail.com
```

The same raw value in two different entity families is never treated as the
same graph node.

## Point-in-time leakage controls

Graph features must follow these rules:

- A transaction may use relationships from `TransactionDT` values strictly
  earlier than its own timestamp only.
- All rows sharing one timestamp are scored before any row from that timestamp
  updates the graph.
- Same-second transactions cannot create graph connections, relationship counts,
  degree values, or component-size information for one another.
- Missing `DeviceInfo` and missing `R_emaildomain` values are not treated as
  shared entities and never create graph edges.
- Fraud labels are not used to construct graph nodes, graph edges, degree
  features, component-size features, or relationship counts.
- Validation starts with graph history from training data and updates only with
  earlier validation transactions.
- The final test period remains locked until all validation-based graph-feature
  decisions are complete.

## Candidate features

| Feature | Definition at transaction-scoring time |
|---|---|
| `graph_card1_prior_degree` | Number of distinct historical device/email nodes linked to the card proxy |
| `graph_deviceinfo_prior_card_degree` | Number of distinct historical card proxies linked to the device |
| `graph_remail_prior_card_degree` | Number of distinct historical card proxies linked to the recipient email domain |
| `graph_card1_device_prior_edge_count` | Earlier transaction count for the exact card–device relationship |
| `graph_card1_remail_prior_edge_count` | Earlier transaction count for the exact card–email relationship |
| `graph_card1_prior_component_size` | Historical connected-component size containing the card proxy |
| `graph_deviceinfo_prior_component_size` | Historical connected-component size containing the device |
| `graph_remail_prior_component_size` | Historical connected-component size containing the email domain |
| `is_new_card1_device_relationship` | 1 when the available card–device edge has not occurred earlier |
| `is_new_card1_remail_relationship` | 1 when the available card–email edge has not occurred earlier |

These features represent relationship structure observed in prior transactions.
They are model inputs and do not prove that an entity, relationship, connected
component, device, card, or email domain caused fraud.

## Evaluation protocol

### Baseline

```text
XGBoost v1.0.0
+ Phase 3 point-in-time behavioural features
+ time-safe target encoding
+ sigmoid / Platt calibration
+ top-k policy at 1,000 reviews
```

### Candidate

```text
The same XGBoost configuration
+ all baseline features
+ time-safe graph features
```

### Validation design

- Train XGBoost on the chronological training period.
- Evaluate initial ranking metrics on the full chronological validation period.
- Fit sigmoid calibration using the earlier 50% of validation time.
- Evaluate the calibrated top-k policy using the later 50% of validation time.
- Compare capacities of 500, 1,000, 1,500, and 2,000 reviews.
- Keep the final test split untouched until the graph candidate is accepted.

### Decision rule

Keep graph features only if they improve fraud capture or net expected value at
the configured operational capacity of 1,000 reviews, without weakening
leakage safety, calibration quality, or analyst usefulness.

If the candidate does not improve the selected policy, document the rejection
and do not use the final test split for graph-feature evaluation.

## Expected artifacts

```text
src/fraud_intelligence/features/graph_features.py
scripts/generate_graph_features.py
tests/unit/test_graph_features.py
scripts/train_xgboost_with_graph_features.py
scripts/compare_phase6_graph_ablation.py
scripts/evaluate_phase6_graph_validation_policy.py
scripts/decide_phase6_graph_ablation.py
reports/tables/phase6_graph_ablation_comparison.csv
reports/tables/phase6_graph_policy_comparison.csv
models/metrics/phase6_graph_validation_policy_evaluation.json
docs/experiments/phase_6_graph_ablation_decision.md
```