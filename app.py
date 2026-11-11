"""Streamlit dashboard for synthetic HR analytics and model demonstration."""
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_PATH, MODEL_PATH, METRICS_PATH

st.set_page_config(page_title="Employee Performance Analytics", page_icon="📊", layout="wide")
st.title("Employee Performance Predictor")
st.caption("Synthetic-data decision-support demonstration • Not for employment decisions")

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.warning("Project data/model not found. Run `python main.py` in the terminal first.")
    st.stop()

df = pd.read_csv(DATA_PATH)
with open(METRICS_PATH, encoding="utf-8") as f:
    metrics = json.load(f)
model = joblib.load(MODEL_PATH)

st.sidebar.header("Dashboard filters")
departments = ["All"] + sorted(df["department"].unique().tolist())
selected_department = st.sidebar.selectbox("Department", departments)
filtered = df if selected_department == "All" else df[df["department"] == selected_department]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Synthetic records", f"{len(filtered):,}")
c2.metric("Average training hours", f"{filtered['training_hours'].mean():.1f}")
c3.metric("Average attendance", f"{filtered['attendance_rate'].mean():.1%}")
c4.metric("Selected model", metrics["selected_model"])

left, right = st.columns(2)
with left:
    rating_counts = filtered["performance_rating"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                 labels={"x": "Synthetic future rating", "y": "Records"},
                 title="Performance category distribution")
    st.plotly_chart(fig, use_container_width=True)
with right:
    dept_summary = filtered.groupby("department", as_index=False).agg(
        average_training_hours=("training_hours", "mean"),
        average_skill_score=("skill_assessment", "mean")
    )
    fig2 = px.bar(dept_summary, x="department", y="average_training_hours",
                  title="Average training hours by department")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Model evaluation")
model_rows = []
for name, result in metrics["models"].items():
    model_rows.append({
        "Model": name,
        "Accuracy": result["accuracy"],
        "Macro precision": result["macro_precision"],
        "Macro recall": result["macro_recall"],
        "Macro F1": result["macro_f1"],
    })
st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
st.caption("Metrics are from a held-out test split of synthetic data. They do not establish real-world validity.")

st.subheader("Model input relevance")
importance_path = "outputs/permutation_importance.csv"
try:
    importance_df = pd.read_csv(importance_path).head(10).sort_values("importance_mean")
    fig_imp = px.bar(importance_df, x="importance_mean", y="feature", orientation="h",
                     error_x="importance_std",
                     title="Top input permutation importance (macro-F1 decrease)")
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Permutation importance is an association-based diagnostic, not a causal explanation or an employee-level reason.")
except FileNotFoundError:
    st.info("Run python main.py to generate feature relevance results.")

st.subheader("Interactive prediction demonstration")
st.write("Enter hypothetical work-related indicators. Avoid entering real employee personal information.")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        department = st.selectbox("Department", sorted(df.department.unique()))
        role_options = sorted(df.loc[df.department == department, "role"].unique())
        role = st.selectbox("Role", role_options)
        tenure = st.slider("Tenure (years)", 0.2, 15.0, 3.0, 0.1)
        projects = st.slider("Prior projects completed", 0, 18, 6)
    with col2:
        workload = st.slider("Workload (hours/week)", 25, 65, 42)
        attendance = st.slider("Attendance rate", 0.70, 1.00, 0.95, 0.01)
        training = st.slider("Training hours", 0.0, 90.0, 30.0)
        skill = st.slider("Skill assessment", 30.0, 100.0, 75.0)
    with col3:
        feedback = st.slider("Manager feedback score", 1.0, 5.0, 3.5, 0.1)
        engagement = st.slider("Engagement score", 1.0, 5.0, 3.5, 0.1)
        overtime = st.slider("Overtime hours/week", 0.0, 25.0, 3.0)
    submitted = st.form_submit_button("Generate demonstration prediction")

if submitted:
    sample = pd.DataFrame([{
        "department": department, "role": role, "tenure_years": tenure,
        "prior_projects_completed": projects, "workload_hours_week": workload,
        "attendance_rate": attendance, "training_hours": training,
        "skill_assessment": skill, "manager_feedback_score": feedback,
        "engagement_score": engagement, "overtime_hours": overtime,
    }])
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    st.info(f"Model output: **{prediction}**")
    probability_df = pd.DataFrame({"Category": model.classes_, "Model probability": probabilities})
    st.plotly_chart(px.bar(probability_df, x="Category", y="Model probability",
                           title="Model class probabilities", range_y=[0, 1]),
                    use_container_width=True)
    st.warning("This is a synthetic scenario, not an assessment of a real person. A probability is not a measure of employee worth or a recommendation for an employment action.")

st.subheader("Sample synthetic records")
st.dataframe(filtered.drop(columns=["employee_id"]).head(15), use_container_width=True)
