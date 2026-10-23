"""
Robustness audit for AquaPalladium cleaning-frequency prediction.

Run from project root:
    python -m src.audit_cleaning_robustness

Outputs:
    outputs/cleaning_robustness/seed_comparison.csv
    outputs/cleaning_robustness/feature_importance_by_seed.csv
    outputs/cleaning_robustness/flow_target_relationship.csv
    outputs/cleaning_robustness/summary.txt

This is an exploratory audit. It cannot establish dataset provenance or prove causality.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "desalination_sensor_data.csv"
OUT_DIR = ROOT / "outputs" / "cleaning_robustness"
TARGET = "Cleaning_Frequency_per_year"
FEATURES = [
    "Feed_Flow_m3h", "Membrane_Age_months", "SDI15",
    "Feed_Pressure_bar", "Feed_Salinity_mgL", "Recovery_Pct",
]
ALL_FEATURES = [
    "Feed_Salinity_mgL", "Feed_Temperature_C", "Feed_Pressure_bar",
    "Recovery_Pct", "Membrane_Age_months", "SDI15",
    "Crossflow_Velocity_m_s", "pH", "Antiscalant_Dose_mgL",
    "System_Downtime_hr_per_month", "Feed_Flow_m3h",
]
SEEDS = [7, 21, 42, 84, 123]

def model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = sorted(set([TARGET] + FEATURES + ALL_FEATURES))
    df = pd.read_csv(DATA_PATH, low_memory=False)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    data = df[columns].replace([np.inf, -np.inf], np.nan)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).reset_index(drop=True)
    y = data[TARGET].astype(float)

    results = []
    importance_rows = []
    for seed in SEEDS:
        idx = np.arange(len(data))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        baseline = DummyRegressor(strategy="median")
        baseline.fit(np.zeros((len(train_idx), 1)), y_train)
        baseline_pred = baseline.predict(np.zeros((len(test_idx), 1)))
        results.append({
            "seed": seed, "model": "median_baseline", "n_features": 0,
            "MAE": mean_absolute_error(y_test, baseline_pred),
            "R2": r2_score(y_test, baseline_pred),
        })

        for model_name, feature_list in [
            ("six_feature_model", FEATURES),
            ("all_eleven_model", ALL_FEATURES),
        ]:
            fitted = model()
            fitted.fit(data.iloc[train_idx][feature_list], y_train)
            pred = fitted.predict(data.iloc[test_idx][feature_list])
            results.append({
                "seed": seed, "model": model_name, "n_features": len(feature_list),
                "MAE": mean_absolute_error(y_test, pred),
                "R2": r2_score(y_test, pred),
            })
            if model_name == "six_feature_model":
                perm = permutation_importance(
                    fitted, data.iloc[test_idx][feature_list], y_test,
                    scoring="neg_mean_absolute_error", n_repeats=3,
                    random_state=seed, n_jobs=-1,
                )
                for j, feature in enumerate(feature_list):
                    importance_rows.append({
                        "seed": seed, "feature": feature,
                        "mae_increase_when_shuffled": perm.importances_mean[j],
                        "importance_std": perm.importances_std[j],
                    })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_DIR / "seed_comparison.csv", index=False)
    imp_df = pd.DataFrame(importance_rows)
    imp_df.to_csv(OUT_DIR / "feature_importance_by_seed.csv", index=False)

    # Descriptive target relationship and frequency of repeated target values per flow bin.
    flow = data[["Feed_Flow_m3h", TARGET]].copy()
    flow["flow_bin"] = pd.qcut(flow["Feed_Flow_m3h"], q=10, duplicates="drop")
    flow_summary = flow.groupby("flow_bin", observed=True).agg(
        rows=(TARGET, "size"),
        flow_min=("Feed_Flow_m3h", "min"),
        flow_median=("Feed_Flow_m3h", "median"),
        flow_max=("Feed_Flow_m3h", "max"),
        cleaning_mean=(TARGET, "mean"),
        cleaning_median=(TARGET, "median"),
        cleaning_std=(TARGET, "std"),
        distinct_target_values=(TARGET, "nunique"),
    ).reset_index()
    flow_summary["flow_bin"] = flow_summary["flow_bin"].astype(str)
    flow_summary.to_csv(OUT_DIR / "flow_target_relationship.csv", index=False)

    summary = result_df.groupby("model").agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
    ).reset_index()
    lines = [
        f"Usable target rows: {len(data):,}",
        "Random 80/20 holdout repeated with seeds: " + ", ".join(map(str, SEEDS)),
        "",
        "Performance summary across seeds (mean ± standard deviation):",
        summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        "Individual split results:",
        result_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        "Interpretation: stable scores across seeds support robustness to random splitting only. "
        "They do not establish performance on future periods or other plants.",
        "The flow-bin summary is descriptive and cannot confirm whether the target was generated "
        "from a hidden formula. Provenance must be checked from dataset documentation or source.",
        f"Outputs: {OUT_DIR}",
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nFeature importance by seed:", OUT_DIR / "feature_importance_by_seed.csv")
    print("Flow/target summary:", OUT_DIR / "flow_target_relationship.csv")

if __name__ == "__main__":
    main()
