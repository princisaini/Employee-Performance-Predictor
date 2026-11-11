"""Train and evaluate leakage-aware multiclass classification pipelines."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PATH, MODEL_PATH, METRICS_PATH, IMAGE_DIR, OUTPUT_DIR

TARGET = "performance_rating"
DROP_COLUMNS = ["employee_id", "future_performance_score", TARGET]

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X.select_dtypes(include="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numerical),
        ("categorical", categorical_pipe, categorical),
    ])

def train_and_evaluate(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_csv(DATA_PATH)

    # Do not use employee identifiers or the future score that created the label.
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=3, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    results = {}
    fitted = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_precision": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
            "classification_report": classification_report(
                y_test, pred, output_dict=True, zero_division=0
            ),
        }
        fitted[name] = pipeline
        print(f"\n{name}")
        print(classification_report(y_test, pred, zero_division=0))

    # Select by macro-F1 rather than accuracy alone.
    best_name = max(results, key=lambda name: results[name]["macro_f1"])
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)

    joblib.dump(best_model, MODEL_PATH)
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, labels=["Low", "Medium", "High"],
        display_labels=["Low", "Medium", "High"], cmap="Blues"
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Model-agnostic feature relevance: change in macro-F1 after shuffling one input.
    # This is descriptive, not causal, and may be unstable with correlated features.
    perm = permutation_importance(
        best_model, X_test, y_test, scoring="f1_macro",
        n_repeats=5, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)

    plt.figure(figsize=(9, 5))
    plot_df = importance_df.head(10).sort_values("importance_mean")
    plt.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_std"])
    plt.title("Top Input Permutation Importance (Macro-F1 decrease)")
    plt.xlabel("Decrease in macro-F1 when feature is shuffled")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "permutation_importance.png", dpi=160)
    plt.close()

    metrics = {
        "selected_model": best_name,
        "selection_metric": "macro_f1",
        "test_rows": int(len(y_test)),
        "class_distribution": y.value_counts(normalize=True).round(4).to_dict(),
        "top_permutation_features": importance_df.head(10).to_dict(orient="records"),
        "models": results,
        "important_note": "Synthetic-data demonstration only; metrics are not evidence of real-world HR validity."
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
