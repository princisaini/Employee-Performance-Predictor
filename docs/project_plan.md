# Phase-wise Project Plan and Proof Checklist

| Phase | Work | Expected proof |
|---|---|---|
| 1. Setup | Create venv, install packages, verify imports | Terminal screenshot, requirements.txt |
| 2. Data | Generate 2,500 synthetic rows | CSV preview, schema |
| 3. Cleaning | Inspect missing values; pipeline imputes inside training folds | Missing-value summary |
| 4. EDA | Distribution, correlations, training-hours chart | Three EDA PNGs |
| 5. Features | Exclude identifier and future score; encode categories | Source code and data dictionary |
| 6. Model | Fit Logistic Regression and Random Forest | Terminal training output |
| 7. Evaluation | Accuracy, macro precision/recall/F1, confusion matrix | Metrics JSON and confusion matrix |
| 8. Insights | Interpret aggregate patterns cautiously | Short findings document |
| 9. Dashboard | Filters, metrics, charts, hypothetical prediction | Dashboard screenshots |
| 10. GitHub | README, docs, clean commits | Repository URL and commit history |

## Suggested five-day execution

- Day 1: Setup and repository skeleton — `chore: initialize employee performance analytics project`
- Day 2: Synthetic data and data dictionary — `feat: add reproducible synthetic HR data generator`
- Day 3: EDA and preprocessing — `feat: add EDA and leakage-aware preprocessing pipeline`
- Day 4: Model training and evaluation — `feat: compare multiclass performance models`
- Day 5: Dashboard, documentation, and screenshots — `feat: add Streamlit analytics dashboard`

Do not create artificial daily commits just to look active. Commit meaningful, tested milestones.
