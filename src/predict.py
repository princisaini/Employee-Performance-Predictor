"""Example prediction using the saved model pipeline."""
import joblib
import pandas as pd
from src.config import MODEL_PATH

def predict_employee(employee: dict) -> dict:
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([employee])
    predicted = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    confidence = {
        str(label): round(float(prob), 4)
        for label, prob in zip(model.classes_, probabilities)
    }
    return {
        "predicted_category": str(predicted),
        "class_probabilities": confidence,
        "notice": "Synthetic demonstration; not for employment decisions."
    }

if __name__ == "__main__":
    example = {
        "department": "Engineering",
        "role": "Software Engineer",
        "tenure_years": 2.5,
        "prior_projects_completed": 7,
        "workload_hours_week": 42,
        "attendance_rate": 0.96,
        "training_hours": 30,
        "skill_assessment": 78,
        "manager_feedback_score": 4.0,
        "engagement_score": 3.8,
        "overtime_hours": 3,
    }
    print(predict_employee(example))
