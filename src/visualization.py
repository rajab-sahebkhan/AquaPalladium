
"""Visualise performance across experimental SWRO conditions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("outputs/experimental/analysed_observations.csv")
OUTPUT_DIR = Path("outputs/experimental/plots")


def plot_permeate_flow(data: pd.DataFrame) -> None:
    """Compare permeate flow across experimental conditions."""
    summary = data.groupby("experiment_type")[
        "permeate_flow_l_min"
    ].agg(["mean", "std"])

    summary = summary.reindex(
        ["steady_constant_flow", "sinusoidal", "rectified_sinusoidal"]
    )

    ax = summary["mean"].plot(
        kind="bar",
        yerr=summary["std"],
        capsize=4,
    )

    ax.set_title("Permeate Flow by Operating Condition")
    ax.set_xlabel("Operating condition")
    ax.set_ylabel("Permeate flow (L/min)")
    ax.tick_params(axis="x", labelrotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "permeate_flow.png", dpi=150)
    plt.close()


def plot_permeate_salinity(data: pd.DataFrame) -> None:
    """Compare permeate salinity across experimental conditions."""
    summary = data.groupby("experiment_type")[
        "permeate_salinity_ppm"
    ].agg(["mean", "std"])

    summary = summary.reindex(
        ["steady_constant_flow", "sinusoidal", "rectified_sinusoidal"]
    )

    ax = summary["mean"].plot(
        kind="bar",
        yerr=summary["std"],
        capsize=4,
    )

    ax.set_title("Permeate Salinity by Operating Condition")
    ax.set_xlabel("Operating condition")
    ax.set_ylabel("Permeate salinity (ppm)")
    ax.tick_params(axis="x", labelrotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "permeate_salinity.png", dpi=150)
    plt.close()


def plot_pressure_vs_flow(data: pd.DataFrame) -> None:
    """Explore the relationship between pressure and permeate flow."""
    sinusoidal = data[data["experiment_type"] == "sinusoidal"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for sheet, group in sinusoidal.groupby("sheet"):
        ax.scatter(
            group["feed_pressure_bar"],
            group["permeate_flow_l_min"],
            label=sheet.replace("Mean=", "").replace("L_min", " L/min"),
            alpha=0.8,
        )

    ax.set_title("Feed Pressure vs Permeate Flow")
    ax.set_xlabel("Feed pressure (bar)")
    ax.set_ylabel("Permeate flow (L/min)")
    ax.legend(title="Nominal mean flow")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pressure_vs_permeate_flow.png", dpi=150)
    plt.close(fig)


def plot_feed_flow_vs_salinity(data: pd.DataFrame) -> None:
    """Explore the relationship between feed flow and permeate salinity."""
    sinusoidal = data[data["experiment_type"] == "sinusoidal"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for sheet, group in sinusoidal.groupby("sheet"):
        ax.scatter(
            group["feed_flow_l_min"],
            group["permeate_salinity_ppm"],
            label=sheet.replace("Mean=", "").replace("L_min", " L/min"),
            alpha=0.8,
        )

    ax.set_title("Feed Flow vs Permeate Salinity")
    ax.set_xlabel("Feed flow (L/min)")
    ax.set_ylabel("Permeate salinity (ppm)")
    ax.legend(title="Nominal mean flow")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feed_flow_vs_salinity.png", dpi=150)
    plt.close(fig)



def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Analysis file not found: {INPUT_FILE}. "
            "Run python -m src.experimental_analytics first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_FILE)

    plot_permeate_flow(data)
    plot_permeate_salinity(data)
    plot_pressure_vs_flow(data)
    plot_feed_flow_vs_salinity(data)

    print(f"Charts saved to: {OUTPUT_DIR.resolve()}")
    print("- permeate_flow.png")
    print("- permeate_salinity.png")
    print("- pressure_vs_permeate_flow.png")
    print("- feed_flow_vs_salinity.png")
    


if __name__ == "__main__":
    main()
