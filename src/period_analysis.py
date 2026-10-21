
"""Explore period-related changes within each amplitude setting."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("data/processed/wave_swro_operating_conditions.csv")
OUTPUT_DIR = Path("outputs/experimental")


def plot_permeate_flow(summary: pd.DataFrame) -> None:
    """Plot permeate flow against period for each amplitude."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for amplitude, group in summary.groupby(level="amplitude_l_min"):
        periods = group.index.get_level_values("period_s")
        flows = group["permeate_flow_l_min"]

        ax.plot(
            periods,
            flows,
            marker="o",
            label=f"Amplitude {amplitude:g} L/min",
        )

    ax.set_title("Permeate Flow vs Period")
    ax.set_xlabel("Period (seconds)")
    ax.set_ylabel("Permeate flow (L/min)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(plot_dir / "period_vs_permeate_flow.png", dpi=150)
    plt.close(fig)


def plot_salinity(summary: pd.DataFrame) -> None:
    """Plot permeate salinity against period for each amplitude."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for amplitude, group in summary.groupby(level="amplitude_l_min"):
        periods = group.index.get_level_values("period_s")
        salinity = group["permeate_salinity_ppm"]

        ax.plot(
            periods,
            salinity,
            marker="o",
            label=f"Amplitude {amplitude:g} L/min",
        )

    ax.set_title("Permeate Salinity vs Period")
    ax.set_xlabel("Period (seconds)")
    ax.set_ylabel("Permeate salinity (ppm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "period_vs_salinity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(INPUT_FILE)
    data = data[data["sheet"] == "Mean=10L_min"].copy()

    metrics = [
        "feed_pressure_bar",
        "permeate_flow_l_min",
        "permeate_salinity_ppm",
    ]

    summary = (
        data.groupby(["amplitude_l_min", "period_s"])[metrics]
        .mean()
        .round(3)
    )

    print("\nPerformance by amplitude and period:")
    print(summary.to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "period_analysis.csv")
    plot_permeate_flow(summary)
    plot_salinity(summary)

    print(f"\nSaved summary to {OUTPUT_DIR / 'period_analysis.csv'}")


if __name__ == "__main__":
    main()

print(f"\nSaved summary to {OUTPUT_DIR / 'period_analysis.csv'}")