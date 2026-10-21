
"""Calculate performance metrics for a simulated desalination system."""

import pandas as pd


def calculate_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Return sensor data with conductivity reduction and energy metrics."""

    required_columns = {
        "feed_conductivity_uS_cm",
        "treated_conductivity_uS_cm",
        "flow_rate_m3_h",
        "energy_kwh",
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Input data must not be empty.")

    if (data["feed_conductivity_uS_cm"] <= 0).any():
        raise ValueError("Feed conductivity must be positive.")

    if (data["treated_conductivity_uS_cm"] < 0).any():
        raise ValueError("Treated conductivity cannot be negative.")

    if (data["flow_rate_m3_h"] <= 0).any():
        raise ValueError("Flow rate must be positive.")

    if (data["energy_kwh"] < 0).any():
        raise ValueError("Energy consumption cannot be negative.")

    result = data.copy()

    result["conductivity_reduction_pct"] = (
        (
            result["feed_conductivity_uS_cm"]
            - result["treated_conductivity_uS_cm"]
        )
        / result["feed_conductivity_uS_cm"]
        * 100
    )

    result["specific_energy_kwh_m3"] = (
        result["energy_kwh"] / result["flow_rate_m3_h"]
    )

    return result
