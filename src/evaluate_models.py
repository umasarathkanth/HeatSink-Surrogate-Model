"""
evaluate_models.py

Evaluate the trained surrogate models using:
- MAE
- RMSE
- R² Score

The evaluation results are printed to the console
and saved as a CSV file.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import MODEL_DIR, SPLIT_DATA_DIR

# =============================================================================
# Paths
# =============================================================================

REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load Test Data
# =============================================================================

def load_test_data():
    """
    Load the saved test split.

    Returns
    -------
    tuple
        X_test, y_test
    """

    X_test = pd.read_csv(SPLIT_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(SPLIT_DATA_DIR / "y_test.csv")

    return X_test, y_test


# =============================================================================
# Evaluate
# =============================================================================

def evaluate_models() -> pd.DataFrame:
    """
    Evaluate all trained models.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics.
    """

    X_test, y_test = load_test_data()

    models = {
        "RandomForest": [
            "RandomForest_total_thermal_resistance.pkl",
            "RandomForest_junction_temperature.pkl",
        ],
        "XGBoost": [
            "XGBoost_total_thermal_resistance.pkl",
            "XGBoost_junction_temperature.pkl",
        ],
    }

    targets = [
        "total_thermal_resistance",
        "junction_temperature",
    ]

    results = []

    for model_name, model_files in models.items():

        for model_file, target in zip(model_files, targets):

            model = joblib.load(MODEL_DIR / model_file)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(
                y_test[target],
                predictions,
            )

            rmse = mean_squared_error(
                y_test[target],
                predictions,
            ) ** 0.5

            r2 = r2_score(
                y_test[target],
                predictions,
            )

            results.append(
                {
                    "Model": model_name,
                    "Target": target,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2 Score": r2,
                }
            )

    return pd.DataFrame(results)


# =============================================================================
# Save Results
# =============================================================================

def save_results(df: pd.DataFrame) -> None:
    """
    Save evaluation metrics.
    """

    output_path = METRICS_DIR / "model_metrics.csv"

    df.to_csv(
        output_path,
        index=False,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    results = evaluate_models()

    save_results(results)

    print("\n" + "=" * 75)
    print("MODEL EVALUATION RESULTS")
    print("=" * 75)
    print(results.to_string(index=False))
    print("=" * 75)

    print(
        f"\nMetrics saved to: {METRICS_DIR / 'model_metrics.csv'}"
    )


if __name__ == "__main__":
    main()