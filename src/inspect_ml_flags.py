
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "outputs/anomaly_comparison/ml_only.csv"
)
BOTH_PATH = Path(
    "outputs/anomaly_comparison/flagged_by_both.csv"
)

FEATURES = [
    "Feed_Salinity_mgL",
    "Feed_Temperature_C",
    "Feed_Pressure_bar",
    "Recovery_Pct",
    "Membrane_Age_months",
    "SDI15",
    "Crossflow_Velocity_m_s",
    "pH",
    "Antiscalant_Dose_mgL",
    "System_Downtime_hr_per_month",
    "Cleaning_Frequency_per_year",
    "Feed_Flow_m3h",
]


def main():
    ml_only = pd.read_csv(INPUT_PATH)
    both = pd.read_csv(BOTH_PATH)

    print("=== ML-ONLY RECORDS ===")
    print(f"Rows: {len(ml_only):,}")

    print("\nMost unusual ML-only records:")
    print(
        ml_only.nsmallest(10, "IF_Anomaly_Score")[
            ["IF_Anomaly_Score"] + FEATURES[:6]
        ].to_string(index=False)
    )

    print("\n=== FEATURE COMPARISON ===")

    for label, frame in [
        ("ML only", ml_only),
        ("Flagged by both", both),
    ]:
        print(f"\n{label}: {len(frame)} rows")

        if frame.empty:
            print("No records to summarize.")
            continue

        print(
            frame[FEATURES]
            .median()
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
