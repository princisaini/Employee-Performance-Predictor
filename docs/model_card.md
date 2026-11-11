# Model Card

## Intended use
Educational demonstration of a multiclass ML workflow using synthetic HR-style data.

## Out-of-scope use
Do not use to rank real employees or make decisions about hiring, promotion, compensation, performance management, termination, or access to opportunities.

## Target
Synthetic future performance category: Low, Medium, High.

## Inputs
Prior-period simulated work indicators, department, role, tenure, workload, attendance, training, skills, feedback, engagement, and overtime.

## Training and evaluation
A stratified 80/20 train/test split. Preprocessing is fitted inside an sklearn Pipeline to avoid fitting transformations on test data. Logistic Regression and Random Forest are compared; selection uses macro-F1.

## Limitations
- Synthetic labels are generated from a known formula, so results can be artificially optimistic.
- No real-world construct validity, temporal validity, or external validation is established.
- Features may encode structural or measurement biases even if protected attributes are excluded.
- Probability outputs may be miscalibrated.
- The dataset is not longitudinal employee history from an actual organization.

## Responsible-use safeguards
Use only for learning and aggregate exploratory analysis. Require human review, validated job-related criteria, employee transparency, privacy controls, documented governance, and independent fairness audits before any real-world research or deployment.
