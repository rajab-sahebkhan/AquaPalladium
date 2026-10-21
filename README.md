# AquaPalladium 🌊

**Experimental desalination analytics using Python, pandas, and Matplotlib.**

AquaPalladium is an independent data-analysis project exploring the performance of seawater reverse osmosis (SWRO) under different operating conditions. It combines experimental measurements, data cleaning, performance calculations, statistical summaries, and visualizations in a reproducible Python workflow.

The project also includes a synthetic sensor-data prototype for exploring desalination monitoring concepts.

## Project Objectives

- Load and standardize experimental desalination data from Excel workbooks.
- Compare operating conditions using pressure, permeate flow, and permeate salinity.
- Calculate water recovery from feed and permeate flow measurements.
- Investigate relationships between operating parameters.
- Produce reproducible summaries and visualizations.
- Develop a foundation for future desalination monitoring and anomaly-detection experiments.

## Dataset

The experimental analysis uses measurements from a wave-powered SWRO desalination study.

The processed data is organized into three datasets:

| Dataset | Observations | Purpose |
|---|---:|---|
| Operating conditions | 117 | Compare steady, sinusoidal, and rectified-sinusoidal flow conditions |
| Constant-pressure experiment | 47 | Analyse a separate experiment with a different measurement schema |
| Membrane-integrity experiments | 59 | Explore measurements under different membrane conditions |

The operating-condition dataset contains 47 steady-flow observations, 45 sinusoidal observations across three nominal mean-flow settings, and 25 rectified-sinusoidal observations.

**Data provenance:** This is experimental wave-powered SWRO data. It is not proprietary DION+ plant data and does not directly represent DION+'s membrane capacitive deionization (MCDI) or closed-circuit reverse osmosis (CCRO) technology.

Check the source dataset's licence and redistribution conditions before publishing the raw data.

## Technology Stack

- Python
- pandas
- NumPy
- Matplotlib
- openpyxl
- pytest

## Project Structure

```text
AquaPalladium/
├── data/
│   ├── raw/
│   │   └── wave_swro/
│   └── processed/
├── outputs/
│   └── experimental/
│       └── plots/
├── src/
│   ├── analytics.py
│   ├── data_audit.py
│   ├── data_generator.py
│   ├── experimental_analytics.py
│   ├── load_wave_dataset.py
│   ├── period_analysis.py
│   ├── statistical_analysis.py
│   └── visualization.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Load and process the experimental data

```powershell
python -m src.load_wave_dataset
```

This generates the cleaned operating-condition, constant-pressure, and membrane-integrity CSV files in `data/processed/`.

### 4. Run the performance analysis

```powershell
python -m src.experimental_analytics
```

### 5. Generate the performance visualizations

```powershell
python -m src.visualization
```

### 6. Analyse sinusoidal operating periods

```powershell
python -m src.period_analysis
```

### 7. Generate descriptive statistical summaries

```powershell
python -m src.statistical_analysis
```

### 8. Run the tests

```powershell
python -m pytest -v
```

## Initial Findings

The current exploratory analysis shows:

- Sinusoidal operation has the highest mean permeate flow and lowest mean permeate salinity among the three operating-condition groups in this dataset.
- Within the sinusoidal experiments, higher nominal mean-flow settings are associated with higher observed permeate flow and lower average salinity.
- In the `Mean=10L_min` experiment, amplitude settings form distinct groups with different feed pressures, permeate flows, and salinity levels.
- Within each amplitude group in that experiment, changes across the measured periods are comparatively small and do not show a consistent improvement in all metrics.

These findings are descriptive. They do not establish causation or identify a universally optimal operating condition.

## Methodology

The experimental workflow:

1. Reads Excel sheets and identifies measurement headers.
2. Removes blank separators and repeated header rows.
3. Standardizes measurement-column names and converts values to numeric types.
4. Separates datasets with different measurement schemas.
5. Calculates recovery as permeate flow divided by feed flow, multiplied by 100.
6. Summarizes measurements by operating condition.
7. Explores correlations and creates plots for performance comparison.

## Limitations

- The dataset is relatively small and comes from a specific experimental setup.
- Rows may represent sequential measurements rather than independent experimental replicates.
- Operating parameters can change together, so correlations do not establish causal effects.
- Reported recovery matches the recovery calculated from the flow measurements in the current operating dataset; this is not independent validation.
- The analysis does not yet establish statistical significance or identify a validated optimal operating condition.
- The dataset does not establish long-term membrane degradation or real-time performance.

## Future Work

- Improve data-quality validation and automated tests for the experimental loader.
- Add clearer plots showing individual observations and within-group variation.
- Investigate the effects of pressure, amplitude, and period while accounting for the experimental design.
- Add reproducible statistical comparisons when the design supports them.
- Explore anomaly detection using simulated or appropriately sourced sensor data.
- Improve documentation of data provenance and licensing.

## License

See [`LICENSE`](LICENSE) for the project's licensing terms. The project code's licence does not automatically grant redistribution rights for third-party datasets.
