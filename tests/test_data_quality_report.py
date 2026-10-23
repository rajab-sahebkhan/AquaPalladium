
import pandas as pd

from src.data_quality_report import generate_data_quality_report


def test_detects_constant_feature():
    df = pd.DataFrame({
        "Specific_Energy_kWh_per_m3": [5.0, 5.0, None, 5.0],
        "Feed_Flow_m3h": [10.0, 20.0, 30.0, 40.0],
    })

    report = generate_data_quality_report(df)
    energy = report.loc[
        report["feature"] == "Specific_Energy_kWh_per_m3"
    ].iloc[0]

    assert energy["unique_non_missing"] == 1
    assert energy["mean"] == 5.0
    assert energy["std"] == 0.0
    assert energy["missing_count"] == 1


def test_reports_missing_percentage():
    df = pd.DataFrame({
        "sensor": [1.0, None, 3.0, 4.0],
    })

    report = generate_data_quality_report(df)
    sensor = report.loc[report["feature"] == "sensor"].iloc[0]

    assert sensor["missing_count"] == 1
    assert sensor["missing_pct"] == 25.0


def test_preserves_all_features():
    df = pd.DataFrame({
        "flow": [1.0, 2.0],
        "pressure": [3.0, 4.0],
    })

    report = generate_data_quality_report(df)

    assert set(report["feature"]) == {"flow", "pressure"}
