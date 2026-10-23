
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


DATA_PATH = Path("data/raw/desalination_sensor_data.csv")
OUTPUT_DIR = Path("outputs/stable_anomaly_investigation")

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

CONTAMINATIONS = [0.01, 0.02, 0.05]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()

    # Exclude features that contain no usable values.
    usable_features = [
        col for col in FEATURES if X[col].notna().any()
    ]
    X = X[usable_features]

    # Median imputation for missing sensor values.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Remove constant features after imputation.
    varying_mask = np.ptp(X_imputed, axis=0) > 0
    X_imputed = X_imputed[:, varying_mask]
    model_features = np.array(usable_features)[varying_mask].tolist()

    # Keep original row numbers so we can trace records to the source CSV.
    results = pd.DataFrame({
        "original_row": np.arange(len(df))
    })

    for contamination in CONTAMINATIONS:
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )

        predictions = model.fit_predict(X_imputed)
        label = f"flag_{int(contamination * 100)}pct"

        results[label] = predictions == -1

    flag_columns = [
        f"flag_{int(c * 100)}pct"
        for c in CONTAMINATIONS
    ]

    results["times_flagged"] = results[flag_columns].sum(axis=1)
    results["stable_anomaly"] = results[flag_columns].all(axis=1)

    # Add the original feature values for interpretation.
    results = pd.concat(
        [results, df[model_features].reset_index(drop=True)],
        axis=1,
    )

    stable = results[results["stable_anomaly"]].copy()
    comparison = results.copy()
    comparison["group"] = np.where(
        comparison["stable_anomaly"],
        "Stable anomaly",
        "Not stable anomaly",
    )

    # Save records for later inspection and dashboard use.
    stable.to_csv(
        OUTPUT_DIR / "stable_anomaly_records.csv",
        index=False,
    )

    results.to_csv(
        OUTPUT_DIR / "all_investigation_results.csv",
        index=False,
    )

    # Compare feature distributions and missingness.
    summary_rows = []

    for feature in model_features:
        normal_values = df.loc[
            ~results["stable_anomaly"], feature
        ]
        anomaly_values = df.loc[
            results["stable_anomaly"], feature
        ]

        normal_median = normal_values.median()
        anomaly_median = anomaly_values.median()

        normal_missing = normal_values.isna().mean() * 100
        anomaly_missing = anomaly_values.isna().mean() * 100

        summary_rows.append({
            "feature": feature,
            "normal_median": normal_median,
            "stable_anomaly_median": anomaly_median,
            "median_difference": anomaly_median - normal_median,
            "normal_missing_pct": normal_missing,
            "anomaly_missing_pct": anomaly_missing,
            "normal_count": normal_values.notna().sum(),
            "anomaly_count": anomaly_values.notna().sum(),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "feature_comparison.csv",
        index=False,
    )

    print("\n=== STABLE ANOMALY INVESTIGATION ===")
    print(f"Total observations: {len(df):,}")

    for column in flag_columns:
        print(f"{column}: {results[column].sum():,}")

    print(f"Stable anomalies: {len(stable):,}")
    print(
        "Stable anomaly percentage: "
        f"{len(stable) / len(df) * 100:.3f}%"
    )

    print("\n=== FEATURE COMPARISON ===")
    print(
        summary.round(3).to_string(index=False)
    )

    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
