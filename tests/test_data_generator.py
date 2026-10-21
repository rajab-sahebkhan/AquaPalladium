
from src.data_generator import generate_sensor_data


def test_generates_requested_number_of_readings():
    data = generate_sensor_data(n_readings=24)

    assert len(data) == 24


def test_repeated_runs_are_reproducible():
    first = generate_sensor_data(n_readings=10, seed=42)
    second = generate_sensor_data(n_readings=10, seed=42)

    assert first.equals(second)


def test_conductivity_values_are_positive():
    data = generate_sensor_data(n_readings=100)

    assert (data["feed_conductivity_uS_cm"] > 0).all()
    assert (data["treated_conductivity_uS_cm"] > 0).all()


def test_rejects_zero_readings():
    import pytest

    with pytest.raises(ValueError):
        generate_sensor_data(n_readings=0)
