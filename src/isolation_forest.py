
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/raw/desalination_sensor_data.csv")
OUTPUT_DIR = Path("outputs/isolation_forest")

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
    df = pd.read_csv(DATA_PATH)

    # Keep original row numbers so flagged records can be traced.
    X = df[FEATURES].copy()
    valid_rows = X.notna().any(axis=1)

    # Impute missing sensor values with each feature's median.
    # Drop features that are entirely missing, if any.
    imputer = SimpleImputer(strategy="median", keep_empty_features=False)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=FEATURES,
        index=df.index,
    )

    # Remove constant features after imputation.
    usable_features = X_imputed.columns[
        X_imputed.nunique() > 1
    ].tolist()
    X_model = X_imputed[usable_features]

    model = make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
        ),
    )

    model.fit(X_model)

    # Isolation Forest returns -1 for anomalies and +1 for inliers.
    predictions = model.predict(X_model)

    # Lower decision scores indicate more unusual observations.
    scores = model.decision_function(X_model)

    results = df.copy()
    results["IF_Anomaly_Score"] = scores
    results["IF_Prediction"] = predictions
    results["IF_Status"] = "Normal"
    results.loc[predictions == -1, "IF_Status"] = "Review: ML anomaly"

    # Rows with no observed feature values cannot be meaningfully assessed.
    results.loc[~valid_rows, "IF_Status"] = "Not assessed"

    # Do not label rows with no usable input as ML anomalies.
    results.loc[~valid_rows, "IF_Prediction"] = 1
    results.loc[~valid_rows, "IF_Anomaly_Score"] = float("nan")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "isolation_forest_results.csv",
        index=False,
    )

    flagged = results[results["IF_Status"] == "Review: ML anomaly"]
    flagged.to_csv(
        OUTPUT_DIR / "flagged_ml_anomalies.csv",
        index=False,
    )

    print("=== AQUAPALLADIUM: ISOLATION FOREST ===")
    print(f"Dataset rows: {len(df):,}")
    print(f"Features used: {usable_features}")
    print(f"Rows with at least one observed feature: {valid_rows.sum():,}")
    print("\n=== MODEL RESULTS ===")
    print(results["IF_Status"].value_counts().to_string())
    print("\nMost unusual observations:")
    print(
        results.nsmallest(10, "IF_Anomaly_Score")[
            ["IF_Anomaly_Score", "IF_Status"] + FEATURES[:5]
        ].to_string(index=False)
    )
    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
