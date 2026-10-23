
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = Path(
    "outputs/stable_anomaly_investigation/all_investigation_results.csv"
)
OUTPUT_DIR = Path("outputs/stable_anomaly_investigation/plots")

FEATURES = {
    "System_Downtime_hr_per_month": "System downtime (hours/month)",
    "Membrane_Age_months": "Membrane age (months)",
    "Crossflow_Velocity_m_s": "Crossflow velocity (m/s)",
    "Feed_Flow_m3h": "Feed flow (m³/h)",
    "Recovery_Pct": "Recovery (%)",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    df["Group"] = df["stable_anomaly"].map({
        True: "Stable anomalies",
        False: "Other observations",
        "True": "Stable anomalies",
        "False": "Other observations",
    })

    for feature, label in FEATURES.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        groups = [
            df.loc[df["Group"] == "Other observations", feature].dropna(),
            df.loc[df["Group"] == "Stable anomalies", feature].dropna(),
        ]

        ax.boxplot(
            groups,
            tick_labels=["Other observations", "Stable anomalies"],
            showfliers=False,
        )

        ax.set_title(f"{label}: comparison by anomaly group")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        filename = feature.lower() + "_comparison.png"
        fig.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close(fig)

        print(f"Saved: {OUTPUT_DIR / filename}")

    print("\nAll five comparison plots generated.")


if __name__ == "__main__":
    main()
