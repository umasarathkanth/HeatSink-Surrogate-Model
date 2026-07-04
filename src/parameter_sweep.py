"""
parameter_sweep.py

Generates dataset using physics-based heat sink model
for surrogate machine learning training.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

from heat_sink_model import calculate_heat_sink


def generate_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate dataset using Latin Hypercube Sampling.

    Args:
        n_samples: Number of samples to generate.

    Returns:
        DataFrame containing inputs and outputs.
    """

    # -----------------------------
    # Parameter ranges
    # -----------------------------
    tdp_range = (30, 250)
    air_velocity_range = (0.5, 15)
    k_tim_range = (1, 12)

    # -----------------------------
    # LHS Sampling
    # -----------------------------
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n_samples)

    scaled = qmc.scale(
        sample,
        [tdp_range[0], air_velocity_range[0], k_tim_range[0]],
        [tdp_range[1], air_velocity_range[1], k_tim_range[1]]
    )

    data = []

    # -----------------------------
    # Physics simulation loop
    # -----------------------------
    for tdp, v_air, k_tim in scaled:

        result = calculate_heat_sink(
            tdp=tdp,
            air_velocity=v_air,
            k_tim=k_tim
        )

        data.append({
            "tdp": tdp,
            "air_velocity": v_air,
            "k_tim": k_tim,
            "reynolds_number": result["reynolds_number"],
            "nusselt_number": result["nusselt_number"],
            "heat_transfer_coefficient": result["heat_transfer_coefficient"],
            "tim_resistance": result["tim_resistance"],
            "conduction_resistance": result["conduction_resistance"],
            "convection_resistance": result["convection_resistance"],
            "total_thermal_resistance": result["total_thermal_resistance"],
            "junction_temperature": result["junction_temperature"],
        })

    return pd.DataFrame(data)


def save_dataset(df: pd.DataFrame, path: str = "data/heat_sink_dataset.csv"):
    """
    Save dataset to CSV.
    """
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")


if __name__ == "__main__":
    df = generate_dataset(n_samples=5000)
    save_dataset(df)
    print(f"Dataset shape: {df.shape}")
    print(df.head())