# Database Schema

## Purpose

PostgreSQL will store analytical data, feature snapshots, predictions, model metadata, and investigation cases.

## Planned Tables

### transactions

Stores joined transaction-level records used for analysis and scoring.

Key fields:

- `transaction_id`
- `transaction_timestamp`
- `transaction_amount`
- `fraud_label`
- Source transaction features
- Source identity features
- `data_version`

### entities

Stores entity-level references and historical relationship information.

Possible entity types:

- Card proxy
- Device proxy
- Email proxy
- Address proxy
- Account proxy

### feature_snapshots

Stores point-in-time feature values used during scoring.

Key fields:

- `transaction_id`
- `feature_version`
- `feature_timestamp`
- Feature values
- `availability_timestamp`

### predictions

Stores every model prediction and decision.

Key fields:

- `prediction_id`
- `transaction_id`
- `risk_score`
- `risk_band`
- `recommended_action`
- `expected_cost`
- `model_version`
- `feature_version`
- `calibration_version`
- `threshold_version`
- `policy_version`
- `scored_at`

### model_versions

Stores model and experiment metadata.

Key fields:

- `model_version`
- `model_type`
- `training_dataset_version`
- `feature_version`
- `calibration_version`
- Training timestamp
- Validation metrics
- Artifact reference

### investigation_cases

Stores cases created for analyst review.

Key fields:

- `case_id`
- `transaction_id`
- `prediction_id`
- `priority_rank`
- `case_status`
- `assigned_to`
- `analyst_outcome`
- `analyst_notes`
- `created_at`
- `updated_at`

## Relationships

```text
transactions 1 ──── many feature_snapshots
transactions 1 ──── many predictions
predictions 1 ──── zero_or_one investigation_cases
model_versions 1 ──── many predictions
```

## Data Protection

- Do not store secrets in the database schema.
- Do not expose raw identity-like fields in public outputs.
- Redact sensitive values in logs.
- Use environment variables for connection settings.