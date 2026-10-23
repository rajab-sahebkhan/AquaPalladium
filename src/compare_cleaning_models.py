"""
Compare cleaning-frequency prediction models on the same held-out rows.

Run from the project root:
    python -m src.compare_cleaning_models

Writes:
    outputs/cleaning_model_comparison/model_comparison.csv
    outputs/cleaning_model_comparison/permutation_importance.csv
    outputs/cleaning_model_comparison/flow_bin_summary.csv
    outputs/cleaning_model_comparison/summary.txt

Important:
- This is predictive analysis, not proof of a causal relationship.
- Random splitting may overestimate generalization if rows are synthetic,
  duplicated in time, or generated from a shared underlying formula.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "desalination_sensor_data.csv"
OUT_DIR = ROOT / "outputs" / "cleaning_model_comparison"
TARGET = "Cleaning_Frequency_per_year"

FEATURE_SETS = {
    "median_baseline": [],
    "feed_flow_only_linear": ["Feed_Flow_m3h"],
    "feed_flow_only_random_forest": ["Feed_Flow_m3h"],
    "top_three_random_forest": ["Feed_Flow_m3h", "Membrane_Age_months", "SDI15"],
    "all_features_random_forest": [
        "Feed_Salinity_mgL", "Feed_Temperature_C", "Feed_Pressure_bar",
        "Recovery_Pct", "Membrane_Age_months", "SDI15",
        "Crossflow_Velocity_m_s", "pH", "Antiscalant_Dose_mgL",
        "System_Downtime_hr_per_month", "Feed_Flow_m3h",
    ],
}

def make_model(name, features, y_train):
    if name == "median_baseline":
        return DummyRegressor(strategy="median")
    if name == "feed_flow_only_linear":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )),
    ])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place desalination_sensor_data.csv in data/raw/."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = [TARGET] + sorted({f for fs in FEATURE_SETS.values() for f in fs})
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    usable = df[required].replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET]).copy()
    y = pd.to_numeric(usable[TARGET], errors="coerce")
    valid = y.notna()
    usable = usable.loc[valid].copy()
    y = y.loc[valid].astype(float)

    # One common split and one common target population for fair comparisons.
    indices = np.arange(len(usable))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    rows, fitted = [], {}

    for name, features in FEATURE_SETS.items():
        model = make_model(name, features, y_train)
        if name == "median_baseline":
            model.fit(np.zeros((len(train_idx), 1)), y_train)
            pred = model.predict(np.zeros((len(test_idx), 1)))
        else:
            X_train = usable.iloc[train_idx][features]
            X_test = usable.iloc[test_idx][features]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        rows.append({
            "model": name,
            "features": ", ".join(features) if features else "(training-set median only)",
            "n_features": len(features),
            "test_rows": len(test_idx),
            "MAE_cleanings_per_year": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred),
        })
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values("MAE_cleanings_per_year")
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # Permutation importance for the full model, measured as MAE increase after shuffling.
    full_features = FEATURE_SETS["all_features_random_forest"]
    full_model = fitted["all_features_random_forest"]
    perm = permutation_importance(
        full_model, usable.iloc[test_idx][full_features], y_test,
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=42, n_jobs=-1
    )
    importance = pd.DataFrame({
        "feature": full_features,
        "mae_increase_when_shuffled": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("mae_increase_when_shuffled", ascending=False)
    importance.to_csv(OUT_DIR / "permutation_importance.csv", index=False)

    # Inspect target values by feed-flow decile; this describes patterns, not causes.
    flow_df = usable[["Feed_Flow_m3h", TARGET]].copy()
    flow_df["flow_decile"] = pd.qcut(flow_df["Feed_Flow_m3h"], q=10, duplicates="drop")
    bins = flow_df.groupby("flow_decile", observed=True).agg(
        rows=(TARGET, "size"),
        flow_median=("Feed_Flow_m3h", "median"),
        cleaning_mean=(TARGET, "mean"),
        cleaning_median=(TARGET, "median"),
        cleaning_std=(TARGET, "std"),
    ).reset_index()
    bins["flow_decile"] = bins["flow_decile"].astype(str)
    bins.to_csv(OUT_DIR / "flow_bin_summary.csv", index=False)

    best = comparison.iloc[0]
    full = comparison.loc[comparison["model"] == "all_features_random_forest"].iloc[0]
    baseline = comparison.loc[comparison["model"] == "median_baseline"].iloc[0]
    report = [
        f"Usable target rows: {len(usable):,}",
        f"Train rows: {len(train_idx):,}; test rows: {len(test_idx):,}",
        "Split: random 80/20, random_state=42",
        "",
        "Model comparison (lower MAE is better; higher R2 is better):",
        comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        f"Best MAE model: {best['model']}",
        f"Full model MAE improvement over median baseline: "
        f"{(baseline['MAE_cleanings_per_year'] - full['MAE_cleanings_per_year']):.4f}",
        "",
        "Interpretation caution: this only tests predictive performance on a random holdout. "
        "It does not establish that feed flow causes cleaning frequency, and results may be "
        "optimistic if the dataset was generated from a formula or contains related rows.",
        f"Outputs: {OUT_DIR}",
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("\nPermutation importance saved to:", OUT_DIR / "permutation_importance.csv")
    print("Feed-flow bins saved to:", OUT_DIR / "flow_bin_summary.csv")

if __name__ == "__main__":
    main()
