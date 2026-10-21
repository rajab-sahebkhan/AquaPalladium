
"""Compare SWRO performance across operating conditions."""

from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data/processed/wave_swro_operating_conditions.csv")
OUTPUT_DIR = Path("outputs/experimental")


def main() -> None:
    data = pd.read_csv(INPUT_FILE)

    metrics = {
        "permeate_flow_l_min": "Permeate flow (L/min)",
        "permeate_salinity_ppm": "Permeate salinity (ppm)",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for column, label in metrics.items():
        summary = (
            data.groupby("experiment_type")[column]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .round(3)
        )

        print(f"\n{'=' * 65}")
        print(label)
        print(summary.to_string())

        summary.to_csv(
            OUTPUT_DIR / f"{column}_group_summary.csv"
        )

    # Compare group medians without assuming normally distributed data.
    print("\nPairwise median differences")
    for column, label in metrics.items():
        print(f"\n{label}")

        medians = data.groupby("experiment_type")[column].median()

        for i, first in enumerate(medians.index):
            for second in medians.index[i + 1:]:
                difference = medians[first] - medians[second]
                print(
                    f"{first} - {second}: "
                    f"{difference:.3f}"
                )

    print(f"\nSummary files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
