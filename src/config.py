from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
IMAGE_DIR = ROOT_DIR / "images"

for folder in (DATA_DIR, MODEL_DIR, OUTPUT_DIR, IMAGE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "synthetic_employee_performance.csv"
MODEL_PATH = MODEL_DIR / "performance_pipeline.joblib"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
