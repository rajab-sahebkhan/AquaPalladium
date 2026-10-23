
import pandas as pd

from src.anomaly_detection import (
    calculate_production_discrepancy,
    flag_discrepancies,
)


def sample_data():
    """Create a small dataset with known expected outcomes."""
    return pd.DataFrame({
        "Feed_Flow_m3h": [20.0, 20.0, None],
        "Recovery_Pct": [50.0, 50.0, 50.0],
        "Water_Production_m3h": [10.0, 15.0, 10.0],
    })


def test_exact_match_has_zero_discrepancy():
    result = calculate_production_discrepancy(sample_data())

    assert result.loc[0, "Expected_Production_m3h"] == 10.0
    assert result.loc[0, "Production_Difference_m3h"] == 0.0
    assert result.loc[0, "Absolute_Difference_m3h"] == 0.0


def test_large_discrepancy_is_flagged():
    result = calculate_production_discrepancy(sample_data())
    result = flag_discrepancies(result, threshold=1.0)

    assert result.loc[1, "Anomaly_Status"] == (
        "Review: production discrepancy"
    )


def test_missing_required_input_is_not_checked():
    result = calculate_production_discrepancy(sample_data())
    result = flag_discrepancies(result, threshold=1.0)

    assert result.loc[2, "Anomaly_Status"] == "Not checked"


def test_original_dataframe_is_not_modified():
    original = sample_data()
    original_columns = original.columns.tolist()

    calculate_production_discrepancy(original)

    assert original.columns.tolist() == original_columns


def test_threshold_boundary_is_consistent():
    data = pd.DataFrame({
        "Feed_Flow_m3h": [20.0],
        "Recovery_Pct": [50.0],
        "Water_Production_m3h": [11.0],
    })

    result = calculate_production_discrepancy(data)
    result = flag_discrepancies(result, threshold=1.0)

    # A difference exactly equal to the threshold is not flagged.
    assert result.loc[0, "Anomaly_Status"] == "Consistent"
