"""
heat_sink_model.py

Physics-based heat sink thermal model.

This module computes:
- Reynolds Number
- Nusselt Number
- Convective Heat Transfer Coefficient
- Thermal Resistances
- Junction Temperature
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeatSinkConfig:
    """
    Fixed heat sink and processor configuration.
    """

    # Processor
    die_length: float = 0.0525
    die_width: float = 0.045
    die_thickness: float = 0.0022

    # Ambient
    ambient_temperature: float = 25.0

    # Heat Sink
    sink_length: float = 90e-3
    sink_width: float = 116e-3
    sink_height: float = 27e-3

    fin_thickness: float = 0.8e-3
    number_of_fins: int = 60
    base_thickness: float = 2.5e-3

    # Material Properties
    aluminum_conductivity: float = 205.0

    # TIM
    tim_thickness: float = 0.1e-3

    # Junction-to-case resistance
    junction_case_resistance: float = 0.2

    # Air Properties @25°C
    air_conductivity: float = 0.0262
    air_kinematic_viscosity: float = 1.57e-5
    air_prandtl_number: float = 0.71
    



def calculate_heat_sink(
    tdp: float,
    air_velocity: float,
    k_tim: float,
    config: HeatSinkConfig = HeatSinkConfig(),
) -> dict:
    """
    Calculate the thermal performance of the heat sink.

    Args:
        tdp: Thermal Design Power (W).
        air_velocity: Air velocity across the heat sink (m/s).
        k_tim: Thermal conductivity of the TIM (W/m·K).
        config: Heat sink configuration.

    Returns:
        Dictionary containing intermediate calculations and final outputs.
    """

    # -----------------------------
    # Fin Geometry
    # -----------------------------
    fin_spacing = (
        config.sink_width
        - (config.number_of_fins * config.fin_thickness)
    ) / (config.number_of_fins - 1)

    fin_height = config.sink_height - config.base_thickness

    # -----------------------------
    # Reynolds Number
    # -----------------------------
    reynolds_number = (
        air_velocity * fin_spacing
    ) / config.air_kinematic_viscosity

    # -----------------------------
    # Nusselt Number
    # -----------------------------
    if reynolds_number < 2300:
        nusselt_number = (
            1.86
            * (
                (
                    reynolds_number
                    * config.air_prandtl_number
                    * (2 * fin_spacing)
                )
                / config.sink_length
            )
            ** (1 / 3)
        )
    else:
        nusselt_number = (
            0.023
            * (reynolds_number ** 0.8)
            * (config.air_prandtl_number ** 0.3)
        )

    # -----------------------------
    # Convective Heat Transfer
    # -----------------------------
    heat_transfer_coefficient = (
        nusselt_number
        * config.air_conductivity
    ) / (2 * fin_spacing)

    # -----------------------------
    # Surface Areas
    # -----------------------------
    fin_area = (
        config.number_of_fins
        * (2 * fin_height * config.sink_length)
    ) + (fin_spacing * config.sink_length)

    total_base_area = (
        config.sink_length
        * config.sink_width
    )

    exposed_base_area = (
        total_base_area
        - (
            config.fin_thickness
            * config.number_of_fins
            * config.sink_length
        )
    )

    total_convection_area = (
        fin_area
        + exposed_base_area
    )

    # -----------------------------
    # Thermal Resistances
    # -----------------------------
    convection_resistance = (
        1
        / (
            heat_transfer_coefficient
            * total_convection_area
        )
    )

    die_area = (
        config.die_length
        * config.die_width
    )

    tim_resistance = (
        config.tim_thickness
        / (
            k_tim
            * die_area
        )
    )

    conduction_resistance = (
        config.base_thickness
        / (
            config.aluminum_conductivity
            * die_area
        )
    )

    heat_sink_resistance = (
        conduction_resistance
        + convection_resistance
    )

    total_resistance = (
        config.junction_case_resistance
        + tim_resistance
        + heat_sink_resistance
    )

    junction_temperature = (
        config.ambient_temperature
        + (tdp * total_resistance)
    )

    return {
        "tdp": tdp,
        "air_velocity": air_velocity,
        "k_tim": k_tim,
        "reynolds_number": reynolds_number,
        "nusselt_number": nusselt_number,
        "heat_transfer_coefficient": heat_transfer_coefficient,
        "tim_resistance": tim_resistance,
        "conduction_resistance": conduction_resistance,
        "convection_resistance": convection_resistance,
        "heat_sink_resistance": heat_sink_resistance,
        "total_thermal_resistance": total_resistance,
        "junction_temperature": junction_temperature,
    }
    
    
    
if __name__ == "__main__":

    results = calculate_heat_sink(
        tdp=150,
        air_velocity=1.0,
        k_tim=4.0,
    )

    print("\n===== Heat Sink Model =====")

    for key, value in results.items():
        print(f"{key:30s}: {value:.6f}" if isinstance(value, float) else f"{key:30s}: {value}")
