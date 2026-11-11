"""Run the complete local training workflow."""
from src.data_generation import save_dataset
from src.eda import run_eda
from src.model import train_and_evaluate

def main():
    print("=" * 60)
    print("EMPLOYEE PERFORMANCE PREDICTOR — SYNTHETIC DEMO")
    print("=" * 60)
    df = save_dataset(n_rows=2500)
    run_eda(df)
    train_and_evaluate(df)
    print("\nPipeline completed. Next run: streamlit run app.py")

if __name__ == "__main__":
    main()
