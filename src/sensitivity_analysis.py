"""
sensitivity_analysis.py

Performs:
- Correlation analysis
- Feature importance analysis (RF + XGBoost)
- Visualization of parameter influence on outputs
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.config import MODEL_DIR, RAW_DATA_DIR

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    df = pd.read_csv(RAW_DATA_DIR / "heat_sink_dataset.csv")
    return df


# -----------------------------
# Correlation Analysis
# -----------------------------
def correlation_analysis(df: pd.DataFrame):
    corr = df.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()

    return corr


# -----------------------------
# Feature Importance
# -----------------------------
def feature_importance(model_path: Path, model_name: str):
    model = joblib.load(model_path)

    importances = model.feature_importances_
    features = ["tdp", "air_velocity", "k_tim"]

    df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    plt.figure()
    plt.bar(df["feature"], df["importance"])
    plt.title(f"{model_name} Feature Importance")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"reports/figures/{model_name}_feature_importance.png")
    plt.close()

    return df


# -----------------------------
# Main
# -----------------------------
def main():

    df = load_data()

    print("\nRunning Correlation Analysis...")
    corr = correlation_analysis(df)
    print(corr)

    print("\nRunning Feature Importance Analysis...")

    rf_model_path = MODEL_DIR / "RandomForest_total_thermal_resistance.pkl"
    xgb_model_path = MODEL_DIR / "XGBoost_total_thermal_resistance.pkl"

    rf_imp = feature_importance(rf_model_path, "RandomForest")
    xgb_imp = feature_importance(xgb_model_path, "XGBoost")

    print("\nRandom Forest Importance:\n", rf_imp)
    print("\nXGBoost Importance:\n", xgb_imp)

    print("\nAnalysis complete. Figures saved in reports/figures/")


if __name__ == "__main__":
    main()