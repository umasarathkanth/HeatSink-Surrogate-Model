"""
train_models.py

Final training pipeline for Heat Sink Surrogate Model.

Responsibilities:
- Load dataset
- Validate dataset
- Train/test split
- Save splits
- Train ML models (Random Forest, XGBoost)
- Save models
- Save training metadata
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import (
    RAW_DATA_DIR,
    SPLIT_DATA_DIR,
    MODEL_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    RF_ESTIMATORS,
)


# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset() -> pd.DataFrame:
    """
    Load dataset from raw directory.
    """
    file_path = RAW_DATA_DIR / "heat_sink_dataset.csv"
    return pd.read_csv(file_path)


# -----------------------------
# Validation
# -----------------------------
def validate_dataset(df: pd.DataFrame) -> None:
    """
    Basic dataset validation checks.
    """
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    expected_cols = {
        "tdp",
        "air_velocity",
        "k_tim",
        "total_thermal_resistance",
        "junction_temperature",
    }

    if not expected_cols.issubset(set(df.columns)):
        raise ValueError("Dataset missing required columns")


# -----------------------------
# Split Data
# -----------------------------
def split_data(df: pd.DataFrame):
    """
    Split dataset into train and test sets.
    """

    X = df[["tdp", "air_velocity", "k_tim"]]

    y = df[
        [
            "total_thermal_resistance",
            "junction_temperature",
        ]
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test


# -----------------------------
# Save splits
# -----------------------------
def save_splits(X_train, X_test, y_train, y_test) -> None:
    """
    Save train/test splits to disk.
    """

    SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(SPLIT_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(SPLIT_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(SPLIT_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(SPLIT_DATA_DIR / "y_test.csv", index=False)


# -----------------------------
# Train Models
# -----------------------------
def train_models(X_train, y_train):
    """
    Train Random Forest and XGBoost models.
    """

    models = {}

    rf = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
    )

    # Train for each target separately (clean + explainable)
    targets = ["total_thermal_resistance", "junction_temperature"]

    for target in targets:

        rf_model = rf.__class__(**rf.get_params())
        rf_model.fit(X_train, y_train[target])

        xgb_model = xgb.__class__(**xgb.get_params())
        xgb_model.fit(X_train, y_train[target])

        models[f"RandomForest_{target}"] = rf_model
        models[f"XGBoost_{target}"] = xgb_model

    return models


# -----------------------------
# Save Models
# -----------------------------
def save_models(models: dict) -> None:
    """
    Save trained models to disk.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, path)


# -----------------------------
# Save Metadata
# -----------------------------
def save_metadata() -> None:
    """
    Save training configuration metadata.
    """

    metadata = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "features": ["tdp", "air_velocity", "k_tim"],
        "targets": [
            "total_thermal_resistance",
            "junction_temperature",
        ],
        "models": ["RandomForest", "XGBoost"],
    }

    with open(MODEL_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


# -----------------------------
# Main
# -----------------------------
def main():

    df = load_dataset()

    validate_dataset(df)

    X_train, X_test, y_train, y_test = split_data(df)

    save_splits(X_train, X_test, y_train, y_test)

    models = train_models(X_train, y_train)

    save_models(models)

    save_metadata()

    print("\nTraining completed successfully")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Splits saved to: {SPLIT_DATA_DIR}")


if __name__ == "__main__":
    main()