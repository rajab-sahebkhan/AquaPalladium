
"""Load and standardise the experimental wave-powered SWRO workbooks."""

from pathlib import Path
import re

import pandas as pd


DATA_DIR = Path("data/raw/wave_swro")
OUTPUT_DIR = Path("data/processed")

COLUMN_NAMES = {
    "Tp (sec)": "period_s",
    "Amplitude (L/min)": "amplitude_l_min",
    "Feed flow (L/min)": "feed_flow_l_min",
    "Brine flow (L/min)": "brine_flow_l_min",
    "Feed pressure (Bar)": "feed_pressure_bar",
    "Brine Pressure (Bar)": "brine_pressure_bar",
    "Pressure (Bar)": "feed_pressure_bar",
    "Flow (L/min)": "feed_flow_l_min",
    "Permeate flow (L/min)": "permeate_flow_l_min",
    "Permeate salinity (ppm)": "permeate_salinity_ppm",
    "Brine salinity (ppm)": "brine_salinity_ppm",
    "Temp (degree C)": "temperature_c",
    "% recovery": "reported_recovery_pct",
    "salt rejection": "reported_salt_rejection_pct",
    "% recovery salt rejection": "reported_recovery_pct",
}



def read_experiment_sheet(
    workbook: Path,
    sheet_name: str,
) -> pd.DataFrame:
    """Read an experiment sheet containing possible repeated headers."""
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)

    # Find the first row containing a recognised measurement header.
    header_row = None
    for i, row in raw.iterrows():
        values = {
            str(value).strip()
            for value in row
            if pd.notna(value)
        }
        if (
            "Feed flow (L/min)" in values
            or "Pressure (Bar)" in values
        ):
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            f"Could not find a recognised header in "
            f"{workbook.name} / {sheet_name}"
        )

    headers = [
        str(value).strip() if pd.notna(value) else None
        for value in raw.iloc[header_row].tolist()
    ]

    # Reuse the first header for the whole sheet.
    data = raw.iloc[header_row + 1:].copy()
    data.columns = headers

    # Remove blank separators and repeated header rows.
    data = data.dropna(how="all")
    first_header = headers[0]
    if first_header is not None:
        data = data[
            data[first_header].astype(str).str.strip() != first_header
        ]

    data = data.loc[:, data.columns.notna()]
    data = data.rename(columns=COLUMN_NAMES)

    # Convert measurements to numbers, preserving missing values.
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(how="all")
    return data.reset_index(drop=True)




def load_wave_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load operating, constant-pressure, and membrane-integrity data."""
    required = [
        "steady_flow.xlsx",
        "sinusoidal.xlsx",
        "rectified_sinusoidal.xlsx",
        "membrane_integrity.xlsx",
    ]
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing workbook(s) in {DATA_DIR}: {missing}"
        )

    # Dataset 1: steady constant-flow experiment.
    operating_parts = []

    frame = read_experiment_sheet(
        DATA_DIR / "steady_flow.xlsx", "const_flow"
    )
    frame["workbook"] = "steady_flow.xlsx"
    frame["sheet"] = "const_flow"
    frame["experiment_type"] = "steady_constant_flow"
    operating_parts.append(frame)

    # Add the three sinusoidal-flow experiments.
    sinusoidal_path = DATA_DIR / "sinusoidal.xlsx"
    for sheet in pd.ExcelFile(sinusoidal_path).sheet_names:
        frame = read_experiment_sheet(sinusoidal_path, sheet)
        frame["workbook"] = sinusoidal_path.name
        frame["sheet"] = sheet
        frame["experiment_type"] = "sinusoidal"

        match = re.search(r"Mean\s*=\s*(\d+(?:\.\d+)?)L_min", sheet)
        frame["nominal_mean_flow_l_min"] = (
            float(match.group(1)) if match else pd.NA
        )
        operating_parts.append(frame)

    # Add rectified sinusoidal-flow experiment.
    rectified_path = DATA_DIR / "rectified_sinusoidal.xlsx"
    sheet = pd.ExcelFile(rectified_path).sheet_names[0]
    frame = read_experiment_sheet(rectified_path, sheet)
    frame["workbook"] = rectified_path.name
    frame["sheet"] = sheet
    frame["experiment_type"] = "rectified_sinusoidal"
    operating_parts.append(frame)

    operating = pd.concat(
        operating_parts, ignore_index=True, sort=False
    )

    # Dataset 2: constant-pressure experiment.
    # It has a different schema, so keep it separate.
    constant_pressure = read_experiment_sheet(
        DATA_DIR / "steady_flow.xlsx", "const_pr"
    )
    constant_pressure = constant_pressure.rename(
        columns={
            "feed_pressure_bar": "pressure_bar",
            "feed_flow_l_min": "flow_l_min",
        }
    )
    constant_pressure["workbook"] = "steady_flow.xlsx"
    constant_pressure["sheet"] = "const_pr"
    constant_pressure["experiment_type"] = "steady_constant_pressure"

    # Dataset 3: membrane-integrity experiments.
    integrity_path = DATA_DIR / "membrane_integrity.xlsx"
    integrity_parts = []

    for sheet in pd.ExcelFile(integrity_path).sheet_names:
        frame = read_experiment_sheet(integrity_path, sheet)
        frame["workbook"] = integrity_path.name
        frame["sheet"] = sheet
        frame["membrane_condition"] = sheet
        integrity_parts.append(frame)

    integrity = pd.concat(
        integrity_parts, ignore_index=True, sort=False
    )

    return operating, constant_pressure, integrity


def main() -> None:
    """Load the experimental workbooks and save cleaned datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    operating, constant_pressure, integrity = load_wave_dataset()

    operating.to_csv(
        OUTPUT_DIR / "wave_swro_operating_conditions.csv",
        index=False,
    )
    constant_pressure.to_csv(
        OUTPUT_DIR / "wave_swro_constant_pressure.csv",
        index=False,
    )
    integrity.to_csv(
        OUTPUT_DIR / "wave_swro_membrane_integrity.csv",
        index=False,
    )

    print(f"Operating dataset: {operating.shape}")
    print(operating.groupby("experiment_type").size().to_string())

    print(f"\nConstant-pressure dataset: {constant_pressure.shape}")

    print(f"\nMembrane-integrity dataset: {integrity.shape}")
    print(integrity.groupby("membrane_condition").size().to_string())

    print(f"\nSaved processed data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
