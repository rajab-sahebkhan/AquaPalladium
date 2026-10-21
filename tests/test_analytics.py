
import pandas as pd
import pytest

from src.analytics import calculate_performance


def test_calculates_conductivity_reduction():
    data = pd.DataFrame({
        "feed_conductivity_uS_cm": [5000.0],
        "treated_conductivity_uS_cm": [250.0],
        "flow_rate_m3_h": [2.0],
        "energy_kwh": [4.0],
    })

    result = calculate_performance(data)

    assert result["conductivity_reduction_pct"].iloc[0] == pytest.approx(95.0)


def test_calculates_specific_energy_consumption():
    data = pd.DataFrame({
        "feed_conductivity_uS_cm": [5000.0],
        "treated_conductivity_uS_cm": [250.0],
        "flow_rate_m3_h": [2.0],
        "energy_kwh": [4.0],
    })

    result = calculate_performance(data)

    assert result["specific_energy_kwh_m3"].iloc[0] == pytest.approx(2.0)


def test_rejects_zero_flow_rate():
    data = pd.DataFrame({
        "feed_conductivity_uS_cm": [5000.0],
        "treated_conductivity_uS_cm": [250.0],
        "flow_rate_m3_h": [0.0],
        "energy_kwh": [4.0],
    })

    with pytest.raises(ValueError):
        calculate_performance(data)


def test_does_not_modify_original_data():
    data = pd.DataFrame({
        "feed_conductivity_uS_cm": [5000.0],
        "treated_conductivity_uS_cm": [250.0],
        "flow_rate_m3_h": [2.0],
        "energy_kwh": [4.0],
    })

    original_columns = list(data.columns)
    calculate_performance(data)

    assert list(data.columns) == original_columns
