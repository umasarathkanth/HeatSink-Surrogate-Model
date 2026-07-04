"""
parameter_sweep.py

Generate a surrogate-model dataset using Latin Hypercube Sampling (LHS)
and the physics-based heat sink model.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import qmc

from config import (
    DATASET_SIZE,
    RAW_DATA_DIR,
    TDP_RANGE,
    AIR_VELOCITY_RANGE,
    TIM_CONDUCTIVITY_RANGE,
)

from heat_sink_model import calculate_heat_sink


def generate_dataset(n_samples: int = DATASET_SIZE) -> pd.DataFrame:
    """
    Generate a dataset using Latin Hypercube Sampling.

    Args:
        n_samples: Number of parameter combinations.

    Returns:
        Pandas DataFrame containing the generated dataset.
    """

    sampler = qmc.LatinHypercube(d=3)

    samples = sampler.random(n_samples)

    scaled_samples = qmc.scale(
        samples,
        [TDP_RANGE[0], AIR_VELOCITY_RANGE[0], TIM_CONDUCTIVITY_RANGE[0]],
        [TDP_RANGE[1], AIR_VELOCITY_RANGE[1], TIM_CONDUCTIVITY_RANGE[1]],
    )

    dataset = []

    for tdp, air_velocity, k_tim in scaled_samples:

        result = calculate_heat_sink(
            tdp=tdp,
            air_velocity=air_velocity,
            k_tim=k_tim,
        )

        dataset.append(
            {
                "tdp": tdp,
                "air_velocity": air_velocity,
                "k_tim": k_tim,
                "reynolds_number": result["reynolds_number"],
                "nusselt_number": result["nusselt_number"],
                "heat_transfer_coefficient": result[
                    "heat_transfer_coefficient"
                ],
                "tim_resistance": result["tim_resistance"],
                "conduction_resistance": result["conduction_resistance"],
                "convection_resistance": result["convection_resistance"],
                "total_thermal_resistance": result[
                    "total_thermal_resistance"
                ],
                "junction_temperature": result["junction_temperature"],
            }
        )

    return pd.DataFrame(dataset)


def save_dataset(df: pd.DataFrame) -> Path:
    """
    Save the generated dataset.

    Args:
        df: Dataset to save.

    Returns:
        Path to the saved CSV file.
    """

    output_path = RAW_DATA_DIR / "heat_sink_dataset.csv"

    df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":

    dataset = generate_dataset()

    output_file = save_dataset(dataset)

    print("=" * 60)
    print("Dataset Generation Completed")
    print("=" * 60)
    print(f"Dataset Shape : {dataset.shape}")
    print(f"Saved To      : {output_file}")
    print("\nFirst Five Samples:\n")
    print(dataset.head())