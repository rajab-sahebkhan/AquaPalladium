
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/desalination_sensor_data.csv")
OUTPUT_DIR = Path("outputs/data_quality")



def generate_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness, uniqueness, and basic statistics."""
    numeric = df.select_dtypes(include="number")

    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
        "unique_non_missing": df.nunique(dropna=True),
    })

    report["mean"] = numeric.mean()
    report["std"] = numeric.std()
    report["min"] = numeric.min()
    report["median"] = numeric.median()
    report["max"] = numeric.max()

    report.index.name = "feature"
    return report.reset_index()



def main():
    df = pd.read_csv(DATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = generate_data_quality_report(df)

    report_path = OUTPUT_DIR / "data_quality_report.csv"
    report.to_csv(report_path, index=False)

    print("=== AQUAPALLADIUM DATA QUALITY REPORT ===")
    print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n=== FEATURES WITH MOST MISSING VALUES ===")
    print(
        report.sort_values("missing_count", ascending=False)[
            ["feature", "missing_count", "missing_pct"]
        ].head(10).to_string(index=False)
    )

    print("\n=== CONSTANT FEATURES ===")
    constants = report.loc[
        (report["unique_non_missing"] == 1)
        & (report["missing_count"] < len(df)),
        ["feature", "unique_non_missing", "mean", "std"],
    ]

    if constants.empty:
        print("No constant features found.")
    else:
        print(constants.to_string(index=False))

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
