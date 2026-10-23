"""
Compare feed-flow-only, six-feature, and flow-bin baseline models.

Run from the project root:
    python -m src.compare_flow_conditional_models

Outputs:
    outputs/flow_conditional_models/model_comparison.csv
    outputs/flow_conditional_models/flow_bin_target_summary.csv
    outputs/flow_conditional_models/summary.txt

All models use the same random 80/20 split. This is exploratory prediction,
not causal validation. The bin model assigns the training-set mean target for
each feed-flow quantile bin, then uses the overall training mean for unseen bins.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "desalination_sensor_data.csv"
OUT_DIR = ROOT / "outputs" / "flow_conditional_models"
TARGET = "Cleaning_Frequency_per_year"
FLOW = "Feed_Flow_m3h"
SIX_FEATURES = [
    FLOW, "Membrane_Age_months", "SDI15",
    "Feed_Pressure_bar", "Feed_Salinity_mgL", "Recovery_Pct",
]

def rf_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [TARGET] + SIX_FEATURES
    df = pd.read_csv(DATA_PATH, low_memory=False)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    data = df[columns].replace([np.inf, -np.inf], np.nan)
    for c in columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=[TARGET]).reset_index(drop=True)
    y = data[TARGET].astype(float)

    idx = np.arange(len(data))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    results = []
    # Median baseline
    median_value = float(y_train.median())
    pred = np.full(len(test), median_value)
    results.append({
        "model": "training_median_baseline", "n_features": 0,
        "MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred),
    })

    # Feed-flow-only RF
    flow_model = rf_pipeline()
    flow_model.fit(train[[FLOW]], y_train)
    pred_flow = flow_model.predict(test[[FLOW]])
    results.append({
        "model": "feed_flow_only_random_forest", "n_features": 1,
        "MAE": mean_absolute_error(y_test, pred_flow), "R2": r2_score(y_test, pred_flow),
    })

    # Six-feature RF
    six_model = rf_pipeline()
    six_model.fit(train[SIX_FEATURES], y_train)
    pred_six = six_model.predict(test[SIX_FEATURES])
    results.append({
        "model": "six_feature_random_forest", "n_features": len(SIX_FEATURES),
        "MAE": mean_absolute_error(y_test, pred_six), "R2": r2_score(y_test, pred_six),
    })

    # Quantile-bin baseline. Bin edges are derived from training data only.
    train_flow = train[FLOW]
    quantiles = np.linspace(0, 1, 11)
    edges = np.unique(train_flow.dropna().quantile(quantiles).to_numpy())
    if len(edges) < 3:
        raise ValueError("Not enough unique feed-flow values to construct bins.")
    edges[0], edges[-1] = -np.inf, np.inf
    train_bins = pd.cut(train_flow, bins=edges, include_lowest=True)
    overall_mean = float(y_train.mean())
    bin_means = y_train.groupby(train_bins, observed=True).mean()
    test_bins = pd.cut(test[FLOW], bins=edges, include_lowest=True)
    pred_bins = test_bins.map(bin_means).astype("float64").fillna(overall_mean).to_numpy(dtype=float)
    results.append({
        "model": "feed_flow_quantile_bin_mean", "n_features": 1,
        "MAE": mean_absolute_error(y_test, pred_bins), "R2": r2_score(y_test, pred_bins),
    })

    comparison = pd.DataFrame(results).sort_values("MAE")
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # Descriptive full-dataset bin summary for interpreting the shape.
    tmp = data[[FLOW, TARGET]].dropna().copy()
    tmp["flow_decile"] = pd.qcut(tmp[FLOW], q=10, duplicates="drop")
    summary = tmp.groupby("flow_decile", observed=True).agg(
        rows=(TARGET, "size"),
        flow_min=(FLOW, "min"),
        flow_median=(FLOW, "median"),
        flow_max=(FLOW, "max"),
        cleaning_mean=(TARGET, "mean"),
        cleaning_median=(TARGET, "median"),
        cleaning_std=(TARGET, "std"),
        zero_fraction=(TARGET, lambda x: float((x == 0).mean())),
        six_fraction=(TARGET, lambda x: float((x == 6).mean())),
        target_unique_values=(TARGET, "nunique"),
    ).reset_index()
    summary["flow_decile"] = summary["flow_decile"].astype(str)
    summary.to_csv(OUT_DIR / "flow_bin_target_summary.csv", index=False)

    lines = [
        f"Rows with target: {len(data):,}",
        f"Train rows: {len(train_idx):,}; test rows: {len(test_idx):,}",
        "Split: random 80/20, random_state=42",
        "",
        "Model comparison (lower MAE is better; higher R2 is better):",
        comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "Interpretation: if the six-feature model beats the flow-only and bin baselines, "
        "the additional inputs help prediction on this random holdout. This does not prove "
        "causality or rule out synthetic target-generation patterns.",
        f"Outputs: {OUT_DIR}",
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nFlow-bin summary:", OUT_DIR / "flow_bin_target_summary.csv")

if __name__ == "__main__":
    main()
