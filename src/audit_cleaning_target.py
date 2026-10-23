"""Audit possible target leakage and relationships for cleaning-frequency prediction.

Run from the repository root:
    python -m src.audit_cleaning_target

Writes tables to outputs/cleaning_target_audit/.
This is exploratory. Correlations do not prove leakage or causality.
"""
from pathlib import Path
import json

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
OUT_DIR = ROOT / "outputs" / "cleaning_target_audit"
TARGET = "Cleaning_Frequency_per_year"
FEATURES = [
    "Feed_Salinity_mgL", "Feed_Temperature_C", "Feed_Pressure_bar",
    "Recovery_Pct", "Membrane_Age_months", "SDI15",
    "Crossflow_Velocity_m_s", "pH", "Antiscalant_Dose_mgL",
    "System_Downtime_hr_per_month", "Feed_Flow_m3h",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw = pd.read_csv(DATA_PATH)
    if TARGET not in raw:
        raise ValueError(f"Missing target: {TARGET}")

    features = [c for c in FEATURES if c in raw.columns]
    data = raw[features + [TARGET]].copy()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET].between(0, 30)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(data)
    y = data[TARGET]

    # Target distribution
    target_dist = y.value_counts(dropna=False).sort_index().rename_axis(TARGET).reset_index(name="count")
    target_dist["share_pct"] = target_dist["count"] / max(n, 1) * 100
    target_dist.to_csv(OUT_DIR / "target_distribution.csv", index=False)

    # Pairwise correlations and target-binned summaries
    rows = []
    for col in features:
        pair = data[[col, TARGET]].dropna()
        if len(pair) < 3 or pair[col].nunique() < 2:
            continue
        rows.append({
            "feature": col,
            "n_complete": len(pair),
            "pearson_r": pair[col].corr(pair[TARGET], method="pearson"),
            "spearman_r": pair[col].corr(pair[TARGET], method="spearman"),
            "feature_unique_values": pair[col].nunique(),
            "target_mean_by_feature_corr": pair.groupby(col, observed=True)[TARGET].mean().corr(
                pd.Series(range(pair.groupby(col, observed=True).ngroups)),
                method="spearman"
            ) if pair[col].nunique() > 2 else np.nan,
        })
    corr = pd.DataFrame(rows)
    if not corr.empty:
        corr["abs_spearman"] = corr["spearman_r"].abs()
        corr = corr.sort_values("abs_spearman", ascending=False)
    corr.to_csv(OUT_DIR / "feature_target_correlations.csv", index=False)

    # Missingness association: can missingness itself signal the target?
    missing_rows = []
    for col in features:
        flag = raw[col].isna().astype(int)
        target_aligned = pd.to_numeric(raw[TARGET], errors="coerce")
        valid = target_aligned.notna()
        if valid.sum() > 0 and flag[valid].nunique() > 1:
            missing_rows.append({
                "feature": col,
                "missing_rate_pct": float(flag[valid].mean() * 100),
                "mean_target_when_present": float(target_aligned[valid & (flag == 0)].mean()),
                "mean_target_when_missing": float(target_aligned[valid & (flag == 1)].mean()) if (valid & (flag == 1)).any() else np.nan,
                "missing_flag_target_spearman": float(flag[valid].corr(target_aligned[valid], method="spearman")),
            })
    pd.DataFrame(missing_rows).to_csv(OUT_DIR / "missingness_target_association.csv", index=False)

    # Deterministic random holdout; compare model to training-median baseline.
    # Sample to keep runtime predictable.
    sample = data.sample(n=min(30000, len(data)), random_state=42) if len(data) > 30000 else data
    X = sample[features]
    y_sample = sample[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sample, test_size=0.2, random_state=42
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=80, min_samples_leaf=5, random_state=42, n_jobs=2
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline_pred = np.full(len(y_test), y_train.median())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    model_mae = mean_absolute_error(y_test, pred)
    metrics = {
        "usable_rows": int(n),
        "holdout_sample_rows": int(len(sample)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model_mae": float(model_mae),
        "baseline_mae": float(baseline_mae),
        "mae_improvement_pct": float((baseline_mae - model_mae) / baseline_mae * 100) if baseline_mae else None,
        "model_r2": float(r2_score(y_test, pred)),
        "train_median_target": float(y_train.median()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "caveat": "Random holdout on unverified dataset; not temporal or real-world validation.",
    }
    (OUT_DIR / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Permutation importance measures test-set score drop after shuffling each feature.
    perm = permutation_importance(
        model, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=4, random_state=42, n_jobs=1,
    )
    perm_df = pd.DataFrame({
        "feature": features,
        "mae_increase_when_shuffled": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("mae_increase_when_shuffled", ascending=False)
    perm_df.to_csv(OUT_DIR / "permutation_importance.csv", index=False)

    print("\nAquaPalladium — cleaning target audit")
    print("=" * 48)
    print(f"Usable rows: {n:,}")
    print("\nTarget counts:")
    print(target_dist.to_string(index=False, formatters={"share_pct": "{:.2f}%".format}))
    print("\nFeature-target Spearman correlations (largest absolute first):")
    if corr.empty:
        print("No usable correlations.")
    else:
        print(corr[["feature", "n_complete", "pearson_r", "spearman_r"]].head(11).to_string(
            index=False, formatters={"pearson_r": "{:.3f}".format, "spearman_r": "{:.3f}".format}
        ))
    print("\nRandom holdout comparison:")
    print(f"  Model MAE:    {model_mae:.3f}")
    print(f"  Baseline MAE: {baseline_mae:.3f}")
    print(f"  Model R²:     {metrics['model_r2']:.3f}")
    print(f"  MAE gain:     {metrics['mae_improvement_pct']:.1f}%")
    print("\nPermutation importance (positive = shuffling worsened MAE):")
    print(perm_df.head(8).to_string(
        index=False, formatters={
            "mae_increase_when_shuffled": "{:.4f}".format,
            "importance_std": "{:.4f}".format,
        }
    ))
    print(f"\nSaved audit outputs to {OUT_DIR.relative_to(ROOT)}")
    print("Note: correlations/importance identify patterns, not causal effects or proof of leakage.")


if __name__ == "__main__":
    main()
