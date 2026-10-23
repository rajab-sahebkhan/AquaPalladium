
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "desalination_sensor_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparison_plots"

TARGET = "Water_Production_m3h"
FEED_FLOW = "Feed_Flow_m3h"
MEMBRANE_AGE = "Membrane_Age_months"

MAX_SCATTER_POINTS = 5000
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load and clean the data
# --------------------------------------------------

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"\nCSV not found:\n{CSV_PATH}\n\n"
            "Place desalination_sensor_data.csv inside "
            "the project's data/raw/ folder."
        )

    df = pd.read_csv(CSV_PATH)

    # Convert columns to numeric where possible.
    # Invalid values become NaN rather than crashing.
    numeric = df.apply(pd.to_numeric, errors="coerce")

    # Replace infinite values with missing values.
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    # Remove completely empty and constant columns.
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

    if TARGET not in numeric.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found.\n"
            f"Available columns: {numeric.columns.tolist()}"
        )

    print(f"Original dataset shape: {df.shape}")
    print(f"Usable numerical columns: {len(numeric.columns)}")
    print(f"Rows available: {len(numeric):,}")

    return numeric


# --------------------------------------------------
# 3. Scatterplots against water production
# --------------------------------------------------

def plot_target_relationships(df):
    features = [
        "Membrane_Age_months",
        "Feed_Flow_m3h",
        "Feed_Pressure_bar",
        "Feed_Salinity_mgL",
        "Feed_Temperature_C",
        "Recovery_Pct",
        "SDI15",
        "Crossflow_Velocity_m_s",
        "Cleaning_Frequency_per_year",
        "System_Downtime_hr_per_month",
        "pH",
    ]

    available = [
        col for col in features
        if col in df.columns and col != TARGET
    ]

    for feature in available:
        pair = df[[feature, TARGET]].dropna()

        if len(pair) < 2:
            print(f"Skipping {feature}: insufficient data.")
            continue

        if len(pair) > MAX_SCATTER_POINTS:
            pair = pair.sample(
                MAX_SCATTER_POINTS,
                random_state=RANDOM_STATE
            )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            pair[feature],
            pair[TARGET],
            s=12,
            alpha=0.25
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("Water production (m³/h)")
        ax.set_title(f"{TARGET} vs {feature}")
        ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"production_vs_{feature}.png",
            dpi=150
        )
        plt.close(fig)

    print(f"Created {len(available)} production scatterplots.")


# --------------------------------------------------
# 4. Median production vs membrane age,
#    grouped by feed-flow range
# --------------------------------------------------

def plot_age_comparison(df):
    required = [MEMBRANE_AGE, FEED_FLOW, TARGET]

    if not all(col in df.columns for col in required):
        print("Skipping age comparison: required columns missing.")
        return

    data = df[required].dropna().copy()

    # Quartiles create four approximately equal-sized groups.
    data["Feed_Flow_Group"] = pd.qcut(
        data[FEED_FLOW],
        q=4,
        labels=["Low", "Medium-low", "Medium-high", "High"],
        duplicates="drop"
    )

    # Five-month age intervals.
    data["Age_Bin"] = (
        np.floor(data[MEMBRANE_AGE] / 5) * 5
    ).astype(int)

    summary = (
        data.groupby(
            ["Feed_Flow_Group", "Age_Bin"],
            observed=True
        )[TARGET]
        .agg(["count", "median", "mean"])
        .reset_index()
    )

    summary = summary[summary["count"] >= 20]

    summary.to_csv(
        OUTPUT_DIR / "age_production_summary.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for group, subset in summary.groupby(
        "Feed_Flow_Group", observed=True
    ):
        ax.plot(
            subset["Age_Bin"],
            subset["median"],
            marker="o",
            label=f"{group} feed flow"
        )

    ax.set_xlabel("Membrane age (months)")
    ax.set_ylabel("Median water production (m³/h)")
    ax.set_title("Median Production vs Membrane Age")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "median_production_vs_age_by_flow.png",
        dpi=150
    )
    plt.close(fig)

    print("Created membrane-age comparison plot and summary.")


# --------------------------------------------------
# 5. Correlation heatmap
# --------------------------------------------------

def plot_correlation_matrix(df):
    corr = df.corr(min_periods=2)

    fig, ax = plt.subplots(figsize=(13, 10))

    image = ax.imshow(
        corr.to_numpy(),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    columns = corr.columns.tolist()

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        columns, rotation=75, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=8)

    # Annotate correlations.
    for i in range(len(columns)):
        for j in range(len(columns)):
            value = corr.iloc[i, j]

            if pd.notna(value):
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7
                )

    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("AquaPalladium — Correlation Matrix")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "correlation_matrix.png",
        dpi=150
    )
    plt.close(fig)

    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    print("Created correlation heatmap and CSV.")


# --------------------------------------------------
# 6. Variable distributions
# --------------------------------------------------

def plot_distributions(df):
    features = [
        "Membrane_Age_months",
        "Feed_Flow_m3h",
        "Feed_Pressure_bar",
        "Feed_Salinity_mgL",
        "Water_Production_m3h",
        "SDI15",
    ]

    available = [col for col in features if col in df.columns]

    for feature in available:
        values = df[feature].dropna()

        if values.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(values, bins=50)
        ax.set_xlabel(feature)
        ax.set_ylabel("Observations")
        ax.set_title(f"Distribution of {feature}")
        ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"distribution_{feature}.png",
            dpi=150
        )
        plt.close(fig)

    print(f"Created {len(available)} distribution plots.")


# --------------------------------------------------
# 7. Run the complete plotting pipeline
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    plot_target_relationships(df)
    plot_age_comparison(df)
    plot_correlation_matrix(df)
    plot_distributions(df)

    print("\nComparison plotting complete!")
    print(f"Results saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
