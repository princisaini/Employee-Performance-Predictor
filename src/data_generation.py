"""Generate synthetic HR-style data. No real employee data is used."""
import numpy as np
import pandas as pd
from src.config import DATA_PATH

RANDOM_STATE = 42

def generate_employee_data(n_rows: int = 2500, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    departments = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations"]
    roles_by_dept = {
        "Engineering": ["Software Engineer", "QA Engineer", "Data Analyst"],
        "Sales": ["Sales Associate", "Account Executive"],
        "Marketing": ["Marketing Analyst", "Content Specialist"],
        "Finance": ["Financial Analyst", "Accountant"],
        "HR": ["HR Associate", "Recruiter"],
        "Operations": ["Operations Analyst", "Coordinator"],
    }

    department = rng.choice(departments, n_rows)
    role = [rng.choice(roles_by_dept[d]) for d in department]

    tenure_years = np.round(rng.uniform(0.2, 15, n_rows), 1)
    prior_projects_completed = rng.poisson(5.5, n_rows).clip(0, 18)
    workload_hours_week = np.round(rng.normal(42, 6, n_rows).clip(25, 65), 1)
    attendance_rate = np.round(rng.normal(0.94, 0.045, n_rows).clip(0.70, 1.0), 3)
    training_hours = np.round(rng.gamma(2.2, 9, n_rows).clip(0, 90), 1)
    skill_assessment = np.round(rng.normal(72, 12, n_rows).clip(30, 100), 1)
    manager_feedback_score = np.round(rng.normal(3.4, 0.75, n_rows).clip(1, 5), 2)
    engagement_score = np.round(rng.normal(3.3, 0.8, n_rows).clip(1, 5), 2)
    overtime_hours = np.round((workload_hours_week - 40).clip(0) + rng.normal(2, 2, n_rows), 1).clip(0, 25)

    # A synthetic future score is generated from prior-period work indicators.
    # It is the label source only and is deliberately excluded from model inputs.
    workload_penalty = np.maximum(workload_hours_week - 48, 0) * 0.75
    latent_score = (
        0.25 * skill_assessment
        + 0.17 * (prior_projects_completed / 12 * 100)
        + 0.14 * (attendance_rate * 100)
        + 0.10 * (training_hours / 60 * 100).clip(0, 100)
        + 5.0 * (manager_feedback_score - 1)
        + 4.0 * (engagement_score - 1)
        + 0.35 * tenure_years
        - workload_penalty
        - 0.25 * overtime_hours
        + rng.normal(0, 8, n_rows)
    )
    future_performance_score = np.round(latent_score.clip(0, 100), 1)
    performance_rating = pd.cut(
        future_performance_score,
        bins=[-0.1, 55, 75, 100.1],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:05d}" for i in range(1, n_rows + 1)],
        "department": department,
        "role": role,
        "tenure_years": tenure_years,
        "prior_projects_completed": prior_projects_completed,
        "workload_hours_week": workload_hours_week,
        "attendance_rate": attendance_rate,
        "training_hours": training_hours,
        "skill_assessment": skill_assessment,
        "manager_feedback_score": manager_feedback_score,
        "engagement_score": engagement_score,
        "overtime_hours": overtime_hours,
        "future_performance_score": future_performance_score,
        "performance_rating": performance_rating,
    })

    # Introduce a small amount of missingness to demonstrate preprocessing.
    for col in ["training_hours", "manager_feedback_score", "engagement_score"]:
        mask = rng.random(n_rows) < 0.015
        df.loc[mask, col] = np.nan

    return df

def save_dataset(n_rows: int = 2500) -> pd.DataFrame:
    df = generate_employee_data(n_rows=n_rows)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} synthetic employee records to: {DATA_PATH}")
    return df

if __name__ == "__main__":
    save_dataset()
