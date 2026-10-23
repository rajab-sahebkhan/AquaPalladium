"""Investigate cleaning-frequency patterns against key operating variables.

Run from the repository root:
    python -m src.analyze_cleaning_relationships

Outputs are descriptive exploratory analyses. They do not establish causation.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "desalination_sensor_data.csv"
OUT_DIR = ROOT / "outputs" / "cleaning_relationships"

TARGET = "Cleaning_Frequency_per_year"
FEATURES = ["Feed_Flow_m3h", "Membrane_Age_months", "SDI15"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = [c for c in [TARGET, *FEATURES] if c not in df.columns]
    if required:
        raise ValueError(f"Missing expected columns: {required}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    use = df[[TARGET, *FEATURES]].apply(pd.to_numeric, errors="coerce")
    use = use[use[TARGET].between(0, 30)]

    # Summaries by target class
    summaries = []
    for feature in FEATURES:
        grouped = use.groupby(TARGET)[feature]
        stats = grouped.agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        ).reset_index()
        stats.insert(0, "feature", feature)
        summaries.append(stats)
    pd.concat(summaries, ignore_index=True).to_csv(
        OUT_DIR / "feature_by_cleaning_frequency.csv", index=False
    )

    # Binned feed-flow relationship: quantile bins avoid equal-width bins dominated by outliers.
    flow_data = use[[TARGET, "Feed_Flow_m3h"]].dropna()
    flow_data = flow_data[flow_data["Feed_Flow_m3h"] >= 0]
    flow_data["feed_flow_bin"] = pd.qcut(
        flow_data["Feed_Flow_m3h"], q=10, duplicates="drop"
    )
    flow_bins = flow_data.groupby("feed_flow_bin", observed=True).agg(
        rows=(TARGET, "size"),
        flow_min=("Feed_Flow_m3h", "min"),
        flow_median=("Feed_Flow_m3h", "median"),
        flow_max=("Feed_Flow_m3h", "max"),
        cleaning_mean=(TARGET, "mean"),
        cleaning_median=(TARGET, "median"),
        cleaning_std=(TARGET, "std"),
    ).reset_index()
    flow_bins["feed_flow_bin"] = flow_bins["feed_flow_bin"].astype(str)
    flow_bins.to_csv(OUT_DIR / "cleaning_by_feed_flow_bins.csv", index=False)

    # Two-dimensional grouped summaries: membrane age × SDI15.
    two = use[[TARGET, "Membrane_Age_months", "SDI15"]].dropna().copy()
    two = two[(two["Membrane_Age_months"] >= 0) & (two["SDI15"] >= 0)]
    two["membrane_age_band"] = pd.cut(
        two["Membrane_Age_months"], bins=[-0.001, 6, 12, 24, 36, 48, 60, np.inf],
        labels=["0–6", "6–12", "12–24", "24–36", "36–48", "48–60", "60+"],
    )
    two["sdi15_band"] = pd.cut(
        two["SDI15"], bins=[-0.001, 1, 2, 3, 4, 5, 10, np.inf],
        labels=["0–1", "1–2", "2–3", "3–4", "4–5", "5–10", "10+"],
    )
    grouped_2d = two.groupby(["membrane_age_band", "sdi15_band"], observed=True).agg(
        rows=(TARGET, "size"),
        cleaning_mean=(TARGET, "mean"),
        cleaning_median=(TARGET, "median"),
    ).reset_index()
    grouped_2d.to_csv(OUT_DIR / "age_sdi15_cleaning_grid.csv", index=False)

    # Feature correlations against target, including Pearson and Spearman.
    corr_rows = []
    for feature in FEATURES:
        pair = use[[feature, TARGET]].dropna()
        corr_rows.append({
            "feature": feature,
            "n": len(pair),
            "pearson_r": pair[feature].corr(pair[TARGET], method="pearson"),
            "spearman_r": pair[feature].corr(pair[TARGET], method="spearman"),
            "feature_unique": pair[feature].nunique(),
            "target_mean": pair[TARGET].mean(),
        })
    pd.DataFrame(corr_rows).to_csv(OUT_DIR / "key_feature_correlations.csv", index=False)

    # Check for near-deterministic target by feature bins using a simple diagnostic:
    # target class spread within quantile bins and number of distinct targets observed.
    deterministic_rows = []
    for feature in FEATURES:
        pair = use[[feature, TARGET]].dropna()
        try:
            pair["bin"] = pd.qcut(pair[feature], q=10, duplicates="drop")
        except ValueError:
            continue
        for bin_label, group in pair.groupby("bin", observed=True):
            counts = group[TARGET].value_counts(normalize=True)
            deterministic_rows.append({
                "feature": feature,
                "bin": str(bin_label),
                "rows": len(group),
                "target_unique_values": group[TARGET].nunique(),
                "dominant_target_share_pct": counts.iloc[0] * 100 if len(counts) else np.nan,
                "target_mean": group[TARGET].mean(),
                "target_std": group[TARGET].std(),
            })
    pd.DataFrame(deterministic_rows).to_csv(
        OUT_DIR / "target_concentration_by_feature_bin.csv", index=False
    )

    print("\nAquaPalladium — cleaning-frequency relationship audit")
    print("=" * 58)
    print("\nFeature-target correlations:")
    print(pd.DataFrame(corr_rows).to_string(
        index=False, formatters={"pearson_r": "{:.3f}".format, "spearman_r": "{:.3f}".format}
    ))
    print("\nMean cleaning frequency by feed-flow quantile bin:")
    print(flow_bins[["feed_flow_bin", "rows", "flow_median", "cleaning_mean", "cleaning_median"]].to_string(
        index=False, formatters={
            "flow_median": "{:.2f}".format,
            "cleaning_mean": "{:.3f}".format,
            "cleaning_median": "{:.1f}".format,
        }
    ))
    print(f"\nSaved summaries to: {OUT_DIR.relative_to(ROOT)}")
    print("Interpretation caution: grouped patterns are observational and do not prove causality or leakage.")


if __name__ == "__main__":
    main()
