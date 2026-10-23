from pathlib import Path
import pandas as pd


def load_data():
    """Load the raw desalination dataset."""
    path = "data/raw/desalination_sensor_data.csv"
    return pd.read_csv(path)


def calculate_production_discrepancy(df):
    """Calculate formula-derived production and discrepancy."""
    result = df.copy()

    result["Expected_Production_m3h"] = (
        result["Feed_Flow_m3h"]
        * result["Recovery_Pct"]
        / 100
    )

    result["Production_Difference_m3h"] = (
        result["Water_Production_m3h"]
        - result["Expected_Production_m3h"]
    )

    result["Absolute_Difference_m3h"] = (
        result["Production_Difference_m3h"].abs()
    )

    return result


def flag_discrepancies(df, threshold=1.0):
    """Flag records with large production discrepancies."""
    result = df.copy()

    required = [
        "Feed_Flow_m3h",
        "Recovery_Pct",
        "Water_Production_m3h",
    ]

    checkable = result[required].notna().all(axis=1)

    result["Anomaly_Status"] = "Not checked"
    result.loc[checkable, "Anomaly_Status"] = "Consistent"

    flagged = (
        checkable
        & (result["Absolute_Difference_m3h"] > threshold)
    )

    result.loc[flagged, "Anomaly_Status"] = (
        "Review: production discrepancy"
    )

    return result


def main():
    df = load_data()

    result = calculate_production_discrepancy(df)
    result = flag_discrepancies(result, threshold=1.0)

    print("=== AQUAPALLADIUM: ANOMALY INVESTIGATION ===")
    print(f"Total rows: {len(result):,}")

    print("\n=== STATUS COUNTS ===")
    print(result["Anomaly_Status"].value_counts().to_string())

    print("\n=== LARGEST DISCREPANCIES ===")
    flagged = result[
        result["Anomaly_Status"]
        == "Review: production discrepancy"
    ]

    columns = [
        "Feed_Flow_m3h",
        "Recovery_Pct",
        "Water_Production_m3h",
        "Expected_Production_m3h",
        "Production_Difference_m3h",
        "Absolute_Difference_m3h",
        "Anomaly_Status",
    ]

    print(
        flagged.nlargest(10, "Absolute_Difference_m3h")[
            columns
        ].to_string(index=False)
    )

        # Create the output directory if it does not exist.
    output_dir = Path("outputs/anomaly_detection")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export every record with its calculated fields and status.
    result.to_csv(
        output_dir / "all_anomaly_results.csv",
        index=False,
    )

    # Export only records that need review.
    flagged.to_csv(
        output_dir / "flagged_records.csv",
        index=False,
    )

    # Export a compact status summary.
    summary = (
        result["Anomaly_Status"]
        .value_counts()
        .rename_axis("Anomaly_Status")
        .reset_index(name="Count")
    )

    summary.to_csv(
        output_dir / "anomaly_summary.csv",
        index=False,
    )

    print("\n=== REPORTS EXPORTED ===")
    print(f"Output directory: {output_dir}")
    print("1. all_anomaly_results.csv")
    print("2. flagged_records.csv")
    print("3. anomaly_summary.csv")

if __name__ == "__main__":
    main()
