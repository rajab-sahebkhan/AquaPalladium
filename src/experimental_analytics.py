
"""Quality checks and performance analysis for experimental SWRO data."""

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/wave_swro_operating_conditions.csv")
OUTPUT_DIR = Path("outputs/experimental")


def analyse_experiments(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate measurements and summarise performance by experiment type."""
    df = data.copy()

    numeric_columns = [
        "feed_flow_l_min",
        "permeate_flow_l_min",
        "feed_pressure_bar",
        "brine_pressure_bar",
        "permeate_salinity_ppm",
        "brine_salinity_ppm",
        "temperature_c",
        "reported_recovery_pct",
        "reported_salt_rejection_pct",
    ]

    # Convert measurement columns to numeric without changing the raw CSV.
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Calculate recovery only where both flow measurements are available.
    valid_flow = (
        df["feed_flow_l_min"].gt(0)
        & df["permeate_flow_l_min"].notna()
        & df["permeate_flow_l_min"].ge(0)
    )

    df["calculated_recovery_pct"] = float("nan")
    df.loc[valid_flow, "calculated_recovery_pct"] = (
        100
        * df.loc[valid_flow, "permeate_flow_l_min"]
        / df.loc[valid_flow, "feed_flow_l_min"]
    )

    # Compare calculated recovery with the value reported by the workbook.
    df["recovery_difference_percentage_points"] = (
        df["calculated_recovery_pct"] - df["reported_recovery_pct"]
    )

    summaries = []
    for experiment, group in df.groupby("experiment_type", dropna=False):
        row = {
            "experiment_type": experiment,
            "observations": len(group),
            "valid_flow_pairs": int(valid_flow.loc[group.index].sum()),
            "mean_feed_pressure_bar": group["feed_pressure_bar"].mean(),
            "mean_permeate_flow_l_min": group["permeate_flow_l_min"].mean(),
            "mean_permeate_salinity_ppm": group["permeate_salinity_ppm"].mean(),
            "mean_reported_recovery_pct": group["reported_recovery_pct"].mean(),
            "mean_calculated_recovery_pct": group["calculated_recovery_pct"].mean(),
            "mean_recovery_difference_pp": (
                group["recovery_difference_percentage_points"].mean()
            ),
        }
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    return df, summary


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}. "
            "Run python -m src.load_wave_dataset first."
        )

    data = pd.read_csv(INPUT_FILE)
    analysed, summary = analyse_experiments(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysed.to_csv(OUTPUT_DIR / "analysed_observations.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)

    print("Dataset shape:", analysed.shape)
    print("\nColumn data types:")
    print(analysed.dtypes.to_string())

    print("\nMissing values:")
    print(analysed.isna().sum().to_string())

    print("\nPerformance summary:")
    print(summary.round(3).to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
