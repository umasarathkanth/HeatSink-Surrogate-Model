"""
eda_analysis.py

Exploratory Data Analysis (EDA) for Heat Sink Dataset
Script-based version (no notebooks)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils import print_section, ensure_directory
from src.config import RAW_DATA_DIR


def load_data():
    """
    Load dataset generated from parameter sweep.
    """
    return pd.read_csv(RAW_DATA_DIR / "heat_sink_dataset.csv")


def run_eda():

    print_section("EDA ANALYSIS STARTED")

    df = load_data()

    # -------------------------
    # Basic Statistics
    # -------------------------
    print("\nDataset Info:\n")
    print(df.info())

    print("\nStatistical Summary:\n")
    print(df.describe())

    # -------------------------
    # Correlation Heatmap
    # -------------------------
    ensure_directory(Path("reports/figures"))

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig("reports/figures/eda_correlation.png")
    plt.close()

    # -------------------------
    # Distributions
    # -------------------------
    df.hist(figsize=(10, 6))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()

    plt.savefig("reports/figures/eda_distributions.png")
    plt.close()

    # -------------------------
    # Key relationship plot
    # -------------------------
    plt.figure()

    plt.scatter(df["air_velocity"], df["junction_temperature"])
    plt.xlabel("Air Velocity")
    plt.ylabel("Junction Temperature")
    plt.title("Air Velocity vs Junction Temperature")

    plt.savefig("reports/figures/air_velocity_vs_temperature.png")
    plt.close()

    print_section("EDA COMPLETED")


if __name__ == "__main__":
    run_eda()