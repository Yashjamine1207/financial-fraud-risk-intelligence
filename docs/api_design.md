# API Design

## Base URL

Local development:

```text
http://localhost:8000
```

## Endpoints

### GET `/health`

Returns service health and loaded model information.

Example response:

```json
{
  "status": "healthy",
  "service": "fraud-risk-scoring-api",
  "model_version": "not_loaded"
}
```

### POST `/predict`

Scores one transaction.

Request fields will include:

- Transaction identifier
- Transaction timestamp
- Transaction amount
- Available entity and transaction features

Response fields:

- Transaction identifier
- Fraud-risk score
- Risk band
- Recommended action
- Expected cost
- Model version
- Feature version
- Calibration version
- Threshold version
- Policy version
- SHAP explanation reference

### POST `/batch-score`

Scores multiple transactions.

The response will include:

- Number of records received
- Number successfully scored
- Number rejected
- Scoring results
- Validation errors

### GET `/cases/{case_id}`

Returns a stored investigation case.

The response may include:

- Case information
- Transaction summary
- Risk score
- Recommended action
- Feature contributions
- Model and policy versions
- Case status
- Analyst outcome

## Validation Rules

The API must reject:

- Missing required fields.
- Invalid timestamps.
- Negative transaction amounts.
- Invalid data types.
- Oversized batch requests.
- Malformed identifiers.

## Error Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "transaction_amount must be greater than or equal to zero",
    "details": {}
  }
}
```

## Security

- Do not return secrets.
- Do not expose raw customer-like data unnecessarily.
- Add authentication and rate limiting if the service is deployed publicly.