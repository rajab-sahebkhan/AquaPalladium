"""
Inspect cleaning-frequency target patterns for possible simple mappings.

Run from project root:
    python -m src.audit_cleaning_target_patterns

Outputs:
    outputs/cleaning_target_patterns/target_by_feature_bins.csv
    outputs/cleaning_target_patterns/target_by_age_sdi_flow.csv
    outputs/cleaning_target_patterns/summary.txt

This script checks descriptive patterns only. It cannot prove data leakage or
reconstruct target generation without dataset documentation.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "desalination_sensor_data.csv"
OUT_DIR = ROOT / "outputs" / "cleaning_target_patterns"
TARGET = "Cleaning_Frequency_per_year"
FEATURES = [
    "Feed_Flow_m3h", "Membrane_Age_months", "SDI15",
    "Feed_Pressure_bar", "Feed_Salinity_mgL", "Recovery_Pct",
]

def bin_summary(df, feature, q=10):
    tmp = df[[feature, TARGET]].copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    tmp[TARGET] = pd.to_numeric(tmp[TARGET], errors="coerce")
    tmp = tmp.dropna()
    if tmp[feature].nunique() < 2:
        return pd.DataFrame()
    try:
        tmp["feature_bin"] = pd.qcut(tmp[feature], q=q, duplicates="drop")
    except ValueError:
        tmp["feature_bin"] = pd.cut(tmp[feature], bins=min(q, tmp[feature].nunique()))
    result = tmp.groupby("feature_bin", observed=True).agg(
        rows=(TARGET, "size"),
        feature_min=(feature, "min"),
        feature_median=(feature, "median"),
        feature_max=(feature, "max"),
        target_mean=(TARGET, "mean"),
        target_median=(TARGET, "median"),
        target_std=(TARGET, "std"),
        target_unique_values=(TARGET, "nunique"),
        target_zero_fraction=(TARGET, lambda x: float((x == 0).mean())),
        target_six_fraction=(TARGET, lambda x: float((x == 6).mean())),
    ).reset_index()
    result.insert(0, "feature", feature)
    result["feature_bin"] = result["feature_bin"].astype(str)
    return result

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    needed = [TARGET] + FEATURES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    data = df[needed].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])

    summaries = [bin_summary(data, feature) for feature in FEATURES]
    summaries = [s for s in summaries if not s.empty]
    all_bins = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    all_bins.to_csv(OUT_DIR / "target_by_feature_bins.csv", index=False)

    # A 2D cross-tab is descriptive: does the target pattern vary by both age and SDI15?
    tmp = data[["Feed_Flow_m3h", "Membrane_Age_months", "SDI15", TARGET]].copy()
    for col in ["Feed_Flow_m3h", "Membrane_Age_months", "SDI15"]:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna()
    for col in ["Feed_Flow_m3h", "Membrane_Age_months", "SDI15"]:
        tmp[col + "_bin"] = pd.qcut(tmp[col], q=5, duplicates="drop")
    two_d = tmp.groupby(
        ["Membrane_Age_months_bin", "SDI15_bin", "Feed_Flow_m3h_bin"], observed=True
    ).agg(
        rows=(TARGET, "size"),
        cleaning_mean=(TARGET, "mean"),
        cleaning_median=(TARGET, "median"),
        cleaning_std=(TARGET, "std"),
        cleaning_unique_values=(TARGET, "nunique"),
    ).reset_index()
    for col in ["Membrane_Age_months_bin", "SDI15_bin", "Feed_Flow_m3h_bin"]:
        two_d[col] = two_d[col].astype(str)
    two_d.to_csv(OUT_DIR / "target_by_age_sdi_flow.csv", index=False)

    lines = [
        f"Rows with target: {len(data):,}",
        f"Target unique values: {sorted(data[TARGET].dropna().unique().tolist())}",
        f"Target mean: {data[TARGET].mean():.4f}; median: {data[TARGET].median():.4f}",
        "",
        "Pearson and Spearman correlations with target:",
    ]
    for col in FEATURES:
        pair = data[[col, TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
        pearson = pair[col].corr(pair[TARGET], method="pearson")
        spearman = pair[col].corr(pair[TARGET], method="spearman")
        lines.append(f"{col}: Pearson={pearson:.4f}; Spearman={spearman:.4f}; paired_rows={len(pair):,}")
    lines += [
        "",
        "Inspect target_by_feature_bins.csv for abrupt changes, flat regions, or target values "
        "that are almost uniquely determined within feature bins.",
        "Inspect target_by_age_sdi_flow.csv for patterns across combinations of flow, age, and SDI15.",
        "Caution: bin-level patterns are not proof of leakage or of a target-generation formula. "
        "Confirm provenance from the dataset documentation/source if possible.",
        f"Outputs: {OUT_DIR}",
    ]
    (OUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nFeature-bin summaries:", OUT_DIR / "target_by_feature_bins.csv")
    print("Combined-bin summary:", OUT_DIR / "target_by_age_sdi_flow.csv")

if __name__ == "__main__":
    main()
