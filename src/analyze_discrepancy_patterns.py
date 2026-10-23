
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/desalination_sensor_data.csv")

df["Expected_Production_m3h"] = (
    df["Feed_Flow_m3h"] * df["Recovery_Pct"] / 100
)

df["Production_Difference_m3h"] = (
    df["Water_Production_m3h"] - df["Expected_Production_m3h"]
)

df["Absolute_Difference_m3h"] = (
    df["Production_Difference_m3h"].abs()
)

df = df.dropna(
    subset=[
        "Feed_Flow_m3h",
        "Recovery_Pct",
        "Water_Production_m3h",
    ]
).copy()

# Create feed-flow groups.
df["Feed_Flow_Group"] = pd.qcut(
    df["Feed_Flow_m3h"],
    q=5,
    duplicates="drop",
)

print("=== DISCREPANCY BY FEED-FLOW GROUP ===")
flow_summary = df.groupby(
    "Feed_Flow_Group", observed=True
).agg(
    rows=("Production_Difference_m3h", "size"),
    mean_difference=("Production_Difference_m3h", "mean"),
    median_difference=("Production_Difference_m3h", "median"),
    mean_absolute_difference=("Absolute_Difference_m3h", "mean"),
    p95_absolute_difference=(
        "Absolute_Difference_m3h",
        lambda s: s.quantile(0.95),
    ),
    over_1_m3h=(
        "Absolute_Difference_m3h",
        lambda s: (s > 1).mean() * 100,
    ),
)

print(flow_summary.round(3).to_string())

# Check correlation, without treating it as causation.
print("\n=== CORRELATIONS ===")
print(
    df[
        [
            "Feed_Flow_m3h",
            "Membrane_Age_months",
            "Recovery_Pct",
            "Production_Difference_m3h",
            "Absolute_Difference_m3h",
        ]
    ].corr().round(3).to_string()
)

# Check whether membrane-age results vary by feed-flow group.
df["Membrane_Age_Group"] = pd.cut(
    df["Membrane_Age_months"],
    bins=[-np.inf, 6, 12, 24, 36, np.inf],
    labels=["0–6", "6–12", "12–24", "24–36", "36+"],
    right=False,
)

print("\n=== DISCREPANCY BY MEMBRANE AGE ===")
age_summary = df.groupby(
    "Membrane_Age_Group", observed=True
).agg(
    rows=("Production_Difference_m3h", "size"),
    median_age=("Membrane_Age_months", "median"),
    mean_absolute_difference=("Absolute_Difference_m3h", "mean"),
    over_1_m3h=(
        "Absolute_Difference_m3h",
        lambda s: (s > 1).mean() * 100,
    ),
)

print(age_summary.round(3).to_string())
