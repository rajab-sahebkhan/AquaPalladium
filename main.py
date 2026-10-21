
"""Run the AquaPalladium desalination monitoring prototype."""

from pathlib import Path

from src.analytics import calculate_performance
from src.data_generator import generate_sensor_data


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    sensor_data = generate_sensor_data(n_readings=100)
    results = calculate_performance(sensor_data)

    print("\nAquaPalladium — Desalination Monitoring")
    print("=" * 44)
    print(f"Simulated readings: {len(results)}")
    print(
        "Average conductivity reduction: "
        f"{results['conductivity_reduction_pct'].mean():.2f}%"
    )
    print(
        "Average specific energy consumption: "
        f"{results['specific_energy_kwh_m3'].mean():.2f} kWh/m³"
    )

    output_file = output_dir / "processed_sensor_data.csv"
    results.to_csv(output_file, index=False)

    print(f"\nProcessed data saved to: {output_file}")


if __name__ == "__main__":
    main()
