"""Train the Student Result prediction model and save model, metrics and dataset."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]


def build_dataset(n=400, seed=42):
    rng = np.random.default_rng(seed)
    pass_rows = pd.DataFrame({
        "attendance": rng.integers(75, 100, n // 2),
        "internal_marks": rng.integers(60, 100, n // 2),
        "assignment_marks": rng.integers(60, 100, n // 2),
        "previous_score": rng.integers(55, 100, n // 2),
        "result": 1,
    })
    fail_rows = pd.DataFrame({
        "attendance": rng.integers(30, 70, n // 2),
        "internal_marks": rng.integers(10, 50, n // 2),
        "assignment_marks": rng.integers(10, 55, n // 2),
        "previous_score": rng.integers(10, 50, n // 2),
        "result": 0,
    })
    return pd.concat([pass_rows, fail_rows]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    df = build_dataset()
    df.to_csv("student_results.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["result"], test_size=0.25, random_state=42, stratify=df["result"]
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(X_test)))
    joblib.dump(model, "student_result_model.pkl")
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": round(accuracy, 4), "samples": len(df), "features": FEATURES}, f, indent=2)
    print(f"Model trained. Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
