import pandas as pd
from src.data_generation import generate_employee_data
from src.model import DROP_COLUMNS, TARGET

def test_synthetic_dataset_schema_and_target():
    df = generate_employee_data(n_rows=150, random_state=7)
    assert len(df) == 150
    assert TARGET in df.columns
    assert set(df[TARGET].unique()).issubset({"Low", "Medium", "High"})
    assert not set(DROP_COLUMNS).isdisjoint(df.columns)

def test_model_inputs_exclude_label_source_and_identifier():
    df = generate_employee_data(n_rows=100)
    X = df.drop(columns=DROP_COLUMNS)
    assert "employee_id" not in X.columns
    assert "future_performance_score" not in X.columns
    assert TARGET not in X.columns
