
from pathlib import Path

import pandas as pd

INPUT_PATH = Path(
    "outputs/stable_anomaly_investigation/all_investigation_results.csv"
)
OUTPUT_DIR = Path("outputs/stable_anomaly_investigation")

FEATURES = [
    "System_Downtime_hr_per_month",
    "Membrane_Age_months",
    "Crossflow_Velocity_m_s",
    "Feed_Flow_m3h",
    "Recovery_Pct",
    "Feed_Pressure_bar",
]


def main():
    df = pd.read_csv(INPUT_PATH)
    df["stable_anomaly"] = (
        df["stable_anomaly"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    normal = df.loc[~df["stable_anomaly"], FEATURES]
    anomaly = df.loc[df["stable_anomaly"], FEATURES]

    # Compare distribution percentiles, not just medians.
    rows = []

    for feature in FEATURES:
        for group_name, values in [
            ("Other observations", normal[feature]),
            ("Stable anomalies", anomaly[feature]),
        ]:
            rows.append({
                "feature": feature,
                "group": group_name,
                "count": values.notna().sum(),
                "missing_pct": values.isna().mean() * 100,
                "p10": values.quantile(0.10),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
            })

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "anomaly_percentiles.csv",
        index=False,
    )

    # Spearman correlations capture monotonic relationships.
    correlations = df[FEATURES].corr(method="spearman")
    correlations.to_csv(
        OUTPUT_DIR / "operating_parameter_correlations.csv"
    )

    # Compare correlations between operating parameters within each group.
    normal_corr = normal.corr(method="spearman")
    anomaly_corr = anomaly.corr(method="spearman")

    correlation_changes = (
        anomaly_corr - normal_corr
    )

    correlation_changes.to_csv(
        OUTPUT_DIR / "correlation_difference.csv"
    )

    print("\n=== PERCENTILE COMPARISON ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== OVERALL SPEARMAN CORRELATIONS ===")
    print(correlations.round(2).to_string())

    print("\nReports saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
