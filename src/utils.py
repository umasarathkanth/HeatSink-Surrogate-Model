"""
utils.py

Common utility functions used across the project.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_directory(directory: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory : Path
        Directory to create.
    """
    directory.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a DataFrame as a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : Path
        Output CSV path.
    """
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)


def save_json(data: dict, output_path: Path) -> None:
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    data : dict
        Dictionary to save.
    output_path : Path
        Output JSON path.
    """
    ensure_directory(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def print_section(title: str) -> None:
    """
    Print a formatted section header.

    Parameters
    ----------
    title : str
        Section title.
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    
    

# =========================
# VISUALIZATION FUNCTIONS
# =========================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)

    plt.figure(figsize=(8, 6))

    corr = df.corr(numeric_only=True)

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(model, feature_names, output_path: Path, title: str) -> None:
    ensure_directory(output_path.parent)

    importances = model.feature_importances_

    indices = np.argsort(importances)

    plt.figure(figsize=(6, 4))

    plt.barh(range(len(importances)), importances[indices])

    plt.yticks(range(len(importances)), np.array(feature_names)[indices])

    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()