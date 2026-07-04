"""
train_models.py

Trains surrogate ML models for:
- Total Thermal Resistance
- Junction Temperature

Models:
- Random Forest Regressor
- XGBoost Regressor
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

from utils import ensure_directory, print_section

from config import RAW_DATA_DIR, MODEL_DIR


# -----------------------------
# Load Dataset
# -----------------------------
def load_data() -> pd.DataFrame:
    """
    Load generated heat sink dataset.
    """
    return pd.read_csv(RAW_DATA_DIR / "heat_sink_dataset.csv")


# -----------------------------
# Train Models
# -----------------------------
def train_models() -> None:

    print_section("Model Training Started")

    ensure_directory(MODEL_DIR)

    df = load_data()

    features = ["tdp", "air_velocity", "k_tim"]

    X = df[features]

    targets = [
        "total_thermal_resistance",
        "junction_temperature",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df[targets],
        test_size=0.2,
        random_state=42,
    )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        ),
    }

    # Train and save models separately for each target
    for model_name, model in models.items():

        for target in targets:

            model.fit(X_train, y_train[target])

            model_path = MODEL_DIR / f"{model_name}_{target}.pkl"

            joblib.dump(model, model_path)

            print(f"Saved {model_name} -> {target} at {model_path}")


# -----------------------------
# Main
# -----------------------------
def main():

    train_models()

    print_section("Model Training Completed Successfully")


if __name__ == "__main__":
    main()