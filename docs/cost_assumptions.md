# Cost Assumptions

## Purpose

The decision policy will compare approve, review, and block actions using explicit financial assumptions.

These values are illustrative policy assumptions for a public-dataset portfolio project. They are not real financial-institution costs.

## Initial Version

- Policy version: `policy-v0.1.0`
- Effective date: 2026-09-14
- Currency: GBP

| Parameter | Initial value | Description |
|---|---:|---|
| False-negative cost | £500 | Estimated loss when fraudulent activity is not stopped |
| Manual-review cost | £5 | Estimated operational cost of reviewing one transaction |
| False-positive escalation cost | £10 | Estimated additional cost when a legitimate transaction is escalated |
| Fraud-prevention value | £500 | Estimated value of preventing a fraudulent transaction |
| Daily review capacity | 1,000 | Maximum number of transactions available for manual review per day |
| Block threshold | To be learned | Selected using validation data and the cost policy |
| Review policy | Top-k under capacity | Rank transactions by decision value and review the available capacity |

## Candidate Actions

### Approve

The transaction is not sent for manual review.

Potential cost:

- Fraud missed: false-negative cost.
- Legitimate transaction: no review cost.

### Review

The transaction is sent to an analyst.

Potential cost:

- Manual-review cost.
- Fraud may still be missed after review.
- Legitimate transactions may create escalation or customer-friction costs.

### Block or Escalate

The transaction is blocked or escalated according to the policy.

Potential cost:

- Legitimate transaction blocked: false-positive or customer-friction cost.
- Fraud prevented: prevention value.

## Policy Rules

1. Values must remain versioned in configuration and documentation.
2. Thresholds must not be selected arbitrarily.
3. Policy selection must use validation data.
4. The final holdout must be evaluated only after the policy is locked.
5. Review capacity must be applied explicitly.
6. Results must report expected cost and fraud captured at capacity.
7. Assumptions must be described as illustrative.

## Sensitivity Analysis

The final report should test how policy performance changes when:

- False-negative cost changes.
- Review cost changes.
- Fraud-prevention value changes.
- Investigation capacity changes.