
"""Generate simulated sensor readings for a desalination system."""

import numpy as np
import pandas as pd


def generate_sensor_data(
    n_readings: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a reproducible dataset of simulated plant measurements."""

    if n_readings <= 0:
        raise ValueError("n_readings must be greater than zero.")

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=n_readings,
        freq="h",
    )

    # Simulated feed-water conductivity, in microsiemens per centimetre.
    feed_conductivity = rng.normal(
        loc=5000, scale=150, size=n_readings
    )

    # Simulated treated-water conductivity.
    treated_conductivity = (
        feed_conductivity * 0.05
        + rng.normal(loc=0, scale=3, size=n_readings)
    )

    # Simulated operating measurements.
    pressure_bar = rng.normal(
        loc=5.0, scale=0.2, size=n_readings
    )
    flow_rate_m3_h = rng.normal(
        loc=2.0, scale=0.1, size=n_readings
    )
    energy_kwh = rng.normal(
        loc=4.0, scale=0.2, size=n_readings
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "feed_conductivity_uS_cm": feed_conductivity,
            "treated_conductivity_uS_cm": treated_conductivity,
            "pressure_bar": pressure_bar,
            "flow_rate_m3_h": flow_rate_m3_h,
            "energy_kwh": energy_kwh,
        }
    )
