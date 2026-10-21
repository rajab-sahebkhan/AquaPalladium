
"""Audit the raw desalination dataset before cleaning."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/desalination_sensor_data.csv")


def audit_data(path: Path = DATA_PATH) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Place the CSV in data/raw/."
        )

    data = pd.read_csv(path)

    print("\nAquaPalladium — Raw Data Audit")
    print("=" * 40)

    print(f"\nShape: {data.shape}")
    print("\nColumn types:")
    print(data.dtypes)

    print("\nMissing values:")
    missing = data.isna().sum()
    missing_pct = (missing / len(data) * 100).round(2)

    print(
        pd.DataFrame({
            "missing_count": missing,
            "missing_percent": missing_pct,
        }).sort_values("missing_percent", ascending=False)
    )

    print(f"\nExact duplicate rows: {data.duplicated().sum()}")

    print("\nDistinct values per column:")
    print(data.nunique())

    print("\nNumeric summary:")
    print(data.describe().round(3).T)

    print("\nCorrelation with water production:")
    print(
        data.corr(numeric_only=True)["Water_Production_m3h"]
        .sort_values(ascending=False)
        .round(3)
    )


if __name__ == "__main__":
    audit_data()
