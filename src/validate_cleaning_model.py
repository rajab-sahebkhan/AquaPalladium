"""Validate AquaPalladium's cleaning-frequency model.

Run from the repository root:
    python -m src.validate_cleaning_model

This is a reproducible exploratory check, not evidence of real-world maintenance accuracy.
It uses a deterministic sample to keep cross-validation fast on a laptop.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "desalination_sensor_data.csv"
OUT_DIR = ROOT / "outputs" / "cleaning_model_validation"

TARGET = "Cleaning_Frequency_per_year"
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
    "Feed_Flow_m3h",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run this from the AquaPalladium repository."
        )

    df = pd.read_csv(DATA_PATH)
    features = [c for c in FEATURES if c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Required target column missing: {TARGET}")
    if len(features) < 2:
        raise ValueError("Not enough expected feature columns were found.")

    data = df[features + [TARGET]].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET].between(0, 30)]

    if len(data) < 500:
        raise ValueError(f"Too few usable rows for validation: {len(data)}")

    print("\nAquaPalladium — cleaning-frequency validation")
    print("=" * 52)
    print(f"Usable target rows: {len(data):,}")
    print(f"Predictors: {len(features)}")
    print("\nTarget distribution:")
    distribution = data[TARGET].value_counts(normalize=True).sort_index()
    for value, share in distribution.items():
        print(f"  {value:g} cleanings/year: {share:.1%}")

    # A fixed sample keeps five-fold validation practical on ordinary laptops.
    sample_size = min(25_000, len(data))
    sample = data.sample(n=sample_size, random_state=42) if len(data) > sample_size else data
    X = sample[features]
    y = sample[TARGET]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=35,
            min_samples_leaf=5,
            max_samples=0.75,
            random_state=42,
            n_jobs=2,
        )),
    ])

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=1,
        return_train_score=False,
    )

    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        baseline_prediction = np.full(len(test_idx), y_train.median())
        baseline_mae = mean_absolute_error(y_test, baseline_prediction)
        model_mae = -results["test_mae"][fold - 1]
        model_r2 = results["test_r2"][fold - 1]
        fold_rows.append({
            "fold": fold,
            "model_mae": model_mae,
            "median_baseline_mae": baseline_mae,
            "mae_improvement": baseline_mae - model_mae,
            "model_r2": model_r2,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
        })

    fold_df = pd.DataFrame(fold_rows)
    print(f"\nFive-fold cross-validation on {len(sample):,} sampled rows:")
    print(f"  MAE: {fold_df['model_mae'].mean():.3f} ± {fold_df['model_mae'].std(ddof=0):.3f} cleanings/year")
    print(f"  R²:  {fold_df['model_r2'].mean():.3f} ± {fold_df['model_r2'].std(ddof=0):.3f}")
    print(f"  Median-baseline MAE: {fold_df['median_baseline_mae'].mean():.3f}")
    print(f"  Mean MAE improvement over baseline: {fold_df['mae_improvement'].mean():.3f}")

    # Fit once on the sample to inspect model importance (importance is descriptive, not causal).
    pipeline.fit(X, y)
    importances = pd.DataFrame({
        "feature": features,
        "importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(OUT_DIR / "cross_validation_folds.csv", index=False)
    importances.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    distribution.rename("share").rename_axis(TARGET).reset_index().to_csv(
        OUT_DIR / "target_distribution.csv", index=False
    )

    summary = {
        "usable_rows": int(len(data)),
        "validation_sample_rows": int(len(sample)),
        "n_features": len(features),
        "cv_folds": 5,
        "mae_mean": float(fold_df["model_mae"].mean()),
        "mae_std": float(fold_df["model_mae"].std(ddof=0)),
        "baseline_mae_mean": float(fold_df["median_baseline_mae"].mean()),
        "r2_mean": float(fold_df["model_r2"].mean()),
        "r2_std": float(fold_df["model_r2"].std(ddof=0)),
        "random_seed": 42,
        "caveat": "Random-fold validation on unverified data; not temporal or real-world validation.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTop model features (not causal explanations):")
    print(importances.head(8).to_string(index=False, formatters={"importance": "{:.3f}".format}))
    print(f"\nSaved outputs to: {OUT_DIR.relative_to(ROOT)}")
    print("\nImportant: random-fold validation can overstate performance if rows are related or synthetic.")


if __name__ == "__main__":
    main()
