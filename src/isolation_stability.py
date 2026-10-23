
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


DATA_PATH = Path("data/raw/desalination_sensor_data.csv")
OUTPUT_DIR = Path("outputs/isolation_stability")

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
    X = df[FEATURES].copy()

    # Exclude features that are completely missing.
    X = X.dropna(axis=1, how="all")

    # Use the same imputed feature matrix for all experiments.
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    # Remove constant columns.
    X_imputed = X_imputed.loc[
        :, X_imputed.nunique() > 1
    ]

    output = pd.DataFrame(index=df.index)
    output["original_row"] = df.index

    for contamination in [0.01, 0.02, 0.05]:
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        model.fit(X_imputed)

        predictions = model.predict(X_imputed)
        scores = model.decision_function(X_imputed)

        suffix = str(int(contamination * 100))

        output[f"score_{suffix}pct"] = scores
        output[f"flag_{suffix}pct"] = predictions == -1

        print(
            f"Contamination {contamination:.0%}: "
            f"{(predictions == -1).sum():,} flagged"
        )

    flag_columns = [
        "flag_1pct",
        "flag_2pct",
        "flag_5pct",
    ]

    output["times_flagged"] = output[flag_columns].sum(axis=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        OUTPUT_DIR / "contamination_stability.csv",
        index=False,
    )

    stable = output[output["times_flagged"] == len(flag_columns)]

    print("\n=== STABILITY SUMMARY ===")
    print(f"Flagged at all three settings: {len(stable):,}")

    print("\nMost unusual observations by the 2% model:")
    print(
        output.nsmallest(10, "score_2pct")[
            [
                "original_row",
                "score_1pct",
                "score_2pct",
                "score_5pct",
                "times_flagged",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
