"""
evaluate_models.py

Evaluates trained surrogate models using:
- MAE
- RMSE
- R² Score
"""

from pathlib import Path
import pandas as pd
import joblib

from utils import (
    ensure_directory,
    save_dataframe,
    print_section,
    plot_correlation_heatmap,
    plot_feature_importance,    
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import ensure_directory, save_dataframe, print_section

from config import RAW_DATA_DIR, MODEL_DIR


# -----------------------------
# Load Test Data
# -----------------------------
def load_data():
    """
    Load dataset for evaluation.
    """
    df = pd.read_csv(RAW_DATA_DIR / "heat_sink_dataset.csv")
    return df


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_models() -> pd.DataFrame:

    print_section("Model Evaluation Started")

    df = load_data()

    features = ["tdp", "air_velocity", "k_tim"]

    X = df[features]

    targets = [
        "total_thermal_resistance",
        "junction_temperature",
    ]

    models = {
        "RandomForest": "RandomForest_{}",
        "XGBoost": "XGBoost_{}",
    }

    results = []

    for model_name, model_pattern in models.items():

        for target in targets:

            model_path = MODEL_DIR / f"{model_name}_{target}.pkl"

            model = joblib.load(model_path)

            y_true = df[target]
            y_pred = model.predict(X)

            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            r2 = r2_score(y_true, y_pred)

            results.append(
                {
                    "Model": model_name,
                    "Target": target,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2 Score": r2,
                }
            )

            print(f"Evaluated {model_name} on {target}")

    results_df = pd.DataFrame(results)
    
    # =============================
    # SAVE VISUALIZATIONS
    # =============================

    plot_dir = Path("reports/figures")
    ensure_directory(plot_dir)

    # 1. Correlation Heatmap
    plot_correlation_heatmap(
        df,
        plot_dir / "correlation_heatmap.png"
    )

    # 2. Feature Importance (Random Forest example)
    rf_model = joblib.load(
        MODEL_DIR / "RandomForest_total_thermal_resistance.pkl"
    )

    plot_feature_importance(
        rf_model,
        ["tdp", "air_velocity", "k_tim"],
        plot_dir / "rf_feature_importance.png",
        "Random Forest Feature Importance"
    )

    output_path = Path("reports/metrics/model_metrics.csv")

    ensure_directory(output_path.parent)
    save_dataframe(results_df, output_path)

    print_section("Model Evaluation Completed")

    print(results_df)

    print(f"\nSaved metrics to: {output_path}")

    return results_df


# -----------------------------
# Main
# -----------------------------
    def main():
        evaluate_models()


    if __name__ == "__main__":
        main()