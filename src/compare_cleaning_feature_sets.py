"""
Compare smaller cleaning-frequency feature sets against the full model.

Run from the project root:
    python -m src.compare_cleaning_feature_sets

Outputs:
    outputs/cleaning_feature_sets/model_comparison.csv
    outputs/cleaning_feature_sets/permutation_importance_full_model.csv
    outputs/cleaning_feature_sets/summary.txt

Exploratory prediction only; this does not establish causality.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "desalination_sensor_data.csv"
OUT_DIR = ROOT / "outputs" / "cleaning_feature_sets"
TARGET = "Cleaning_Frequency_per_year"

FEATURE_SETS = {
    "top_four": ["Feed_Flow_m3h", "Membrane_Age_months", "SDI15", "Feed_Pressure_bar"],
    "top_six": ["Feed_Flow_m3h", "Membrane_Age_months", "SDI15", "Feed_Pressure_bar",
                "Feed_Salinity_mgL", "Recovery_Pct"],
    "all_eleven": ["Feed_Salinity_mgL", "Feed_Temperature_C", "Feed_Pressure_bar",
                   "Recovery_Pct", "Membrane_Age_months", "SDI15",
                   "Crossflow_Velocity_m_s", "pH", "Antiscalant_Dose_mgL",
                   "System_Downtime_hr_per_month", "Feed_Flow_m3h"],
    "without_feed_flow": ["Feed_Salinity_mgL", "Feed_Temperature_C", "Feed_Pressure_bar",
                          "Recovery_Pct", "Membrane_Age_months", "SDI15",
                          "Crossflow_Velocity_m_s", "pH", "Antiscalant_Dose_mgL",
                          "System_Downtime_hr_per_month"],
}

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    needed = sorted({col for cols in FEATURE_SETS.values() for col in cols} | {TARGET})
    df = pd.read_csv(DATA_PATH, low_memory=False)
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    data = df[needed].replace([np.inf, -np.inf], np.nan)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).reset_index(drop=True)
    y = data[TARGET].astype(float)
    idx = np.arange(len(data))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    results, fitted = [], {}
    for name, features in FEATURE_SETS.items():
        model = make_model()
        model.fit(data.iloc[train_idx][features], y_train)
        pred = model.predict(data.iloc[test_idx][features])
        results.append({
            "model": name, "n_features": len(features), "features": ", ".join(features),
            "test_rows": len(test_idx),
            "MAE_cleanings_per_year": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred),
        })
        fitted[name] = model

    comparison = pd.DataFrame(results).sort_values("MAE_cleanings_per_year")
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    full_features = FEATURE_SETS["all_eleven"]
    perm = permutation_importance(
        fitted["all_eleven"], data.iloc[test_idx][full_features], y_test,
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=42, n_jobs=-1
    )
    importance = pd.DataFrame({
        "feature": full_features,
        "mae_increase_when_shuffled": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("mae_increase_when_shuffled", ascending=False)
    importance.to_csv(OUT_DIR / "permutation_importance_full_model.csv", index=False)

    full_mae = float(comparison.loc[comparison.model == "all_eleven", "MAE_cleanings_per_year"].iloc[0])
    top4_mae = float(comparison.loc[comparison.model == "top_four", "MAE_cleanings_per_year"].iloc[0])
    noflow_mae = float(comparison.loc[comparison.model == "without_feed_flow", "MAE_cleanings_per_year"].iloc[0])
    lines = [
        f"Usable target rows: {len(data):,}",
        f"Train rows: {len(train_idx):,}; test rows: {len(test_idx):,}",
        "Split: random 80/20, random_state=42; same rows for every model.",
        "",
        "Model comparison (lower MAE is better; higher R2 is better):",
        comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        f"Top-four MAE minus full-model MAE: {top4_mae - full_mae:.4f}",
        f"Without-feed-flow MAE minus full-model MAE: {noflow_mae - full_mae:.4f}",
        "",
        "Caution: random holdout results do not prove causality or guarantee generalization "
        "to another plant or time period. Synthetic/formula-generated labels or related rows "
        "could make scores optimistic.",
        f"Outputs: {OUT_DIR}",
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nFull-model permutation importance saved to:", OUT_DIR / "permutation_importance_full_model.csv")

if __name__ == "__main__":
    main()
