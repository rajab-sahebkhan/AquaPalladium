
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/desalination_sensor_data.csv")

# Calculate the discrepancy between recorded and expected production.
df["Expected_Production_m3h"] = (
    df["Feed_Flow_m3h"] * df["Recovery_Pct"] / 100
)

df["Production_Difference_m3h"] = (
    df["Water_Production_m3h"] - df["Expected_Production_m3h"]
)

# Only rows with all three required values can be checked.
valid = df.dropna(
    subset=[
        "Feed_Flow_m3h",
        "Recovery_Pct",
        "Water_Production_m3h",
    ]
).copy()

# Label candidate discrepancies for investigation.
valid["Candidate_Discrepancy"] = (
    valid["Production_Difference_m3h"].abs() > 1
)

print("=== CANDIDATE DISCREPANCY RATE ===")
print(
    valid["Candidate_Discrepancy"]
    .value_counts()
    .rename(index={True: "Difference > 1", False: "Difference <= 1"})
)

print("\n=== COMPARE OPERATING CONDITIONS ===")
features = [
    "Feed_Flow_m3h",
    "Recovery_Pct",
    "Feed_Pressure_bar",
    "Membrane_Age_months",
    "SDI15",
    "Feed_Salinity_mgL",
    "Feed_Temperature_C",
]

summary = valid.groupby("Candidate_Discrepancy")[features].agg(
    ["count", "mean", "median"]
)

print(summary.to_string())

print("\n=== MISSINGNESS COMPARISON ===")
for col in features:
    missing_rate = valid.groupby("Candidate_Discrepancy")[col].apply(
        lambda s: s.isna().mean() * 100
    )
    print(f"\n{col} — missing %")
    print(missing_rate.round(2).to_string())

print("\n=== DISCREPANCY DIRECTION ===")
print(
    valid.loc[valid["Candidate_Discrepancy"],
              "Production_Difference_m3h"]
    .agg(["min", "median", "mean", "max"])
)
