"""Exploratory data analysis and chart generation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import DATA_PATH, IMAGE_DIR

sns.set_theme()

def run_eda(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nRating distribution:\n", df["performance_rating"].value_counts(normalize=True).round(3))
    print("\nNumeric summary:\n", df.select_dtypes(include="number").describe().round(2))

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="performance_rating", order=["Low", "Medium", "High"], color="steelblue")
    plt.title("Synthetic Future Performance Rating Distribution")
    plt.xlabel("Performance rating")
    plt.ylabel("Employee records")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "rating_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    corr = df.select_dtypes(include="number").drop(columns=["future_performance_score"], errors="ignore").corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_correlation.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    sns.boxplot(data=df, x="performance_rating", y="training_hours",
                order=["Low", "Medium", "High"], color="steelblue")
    plt.title("Training Hours by Synthetic Rating")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "training_by_rating.png", dpi=160)
    plt.close()

    print(f"EDA charts saved in {IMAGE_DIR}")
