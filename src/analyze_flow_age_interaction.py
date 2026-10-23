
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/desalination_sensor_data.csv")

df["Expected_Production_m3h"] = (
    df["Feed_Flow_m3h"] * df["Recovery_Pct"] / 100
)

df["Absolute_Difference_m3h"] = (
    df["Water_Production_m3h"] - df["Expected_Production_m3h"]
).abs()

df = df.dropna(
    subset=[
        "Feed_Flow_m3h",
        "Recovery_Pct",
        "Water_Production_m3h",
        "Membrane_Age_months",
    ]
).copy()

df["Flow_Group"] = pd.qcut(
    df["Feed_Flow_m3h"],
    q=4,
    labels=["Low", "Medium", "High", "Very high"],
)

df["Age_Group"] = pd.cut(
    df["Membrane_Age_months"],
    bins=[-np.inf, 6, 12, 24, 36, np.inf],
    labels=["0–6", "6–12", "12–24", "24–36", "36+"],
    right=False,
)

df["Flagged"] = df["Absolute_Difference_m3h"] > 1

grouped = df.groupby(
    ["Age_Group", "Flow_Group"],
    observed=True,
).agg(
    rows=("Flagged", "size"),
    flagged_rows=("Flagged", "sum"),
    median_abs_error=("Absolute_Difference_m3h", "median"),
    p99_abs_error=("Absolute_Difference_m3h", lambda s: s.quantile(0.99)),
)

grouped["flagged_pct"] = (
    grouped["flagged_rows"] / grouped["rows"] * 100
)

print("=== FLOW AND AGE INTERACTION ===")
print(grouped.round(3).to_string())

print("\n=== FLAGGED ROW COUNTS ===")
print("Total rows:", len(df))
print("Flagged rows:", int(df["Flagged"].sum()))
print("Overall flag rate (%):", round(df["Flagged"].mean() * 100, 3))
