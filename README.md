# AquaPalladium 🌊

**Desalination analytics, sensor monitoring, and exploratory machine learning using Python.**

AquaPalladium is an independent portfolio project exploring seawater reverse osmosis (SWRO) performance and data-driven desalination monitoring. It combines experimental data processing, statistical analysis, interactive visualization, data-quality checks, anomaly detection, and exploratory machine learning in a reproducible Python workflow.

The project has two complementary workstreams:

- **Experimental SWRO analysis:** process and analyze measurements from wave-powered desalination experiments to compare operating conditions, permeate flow, salinity, pressure, and recovery.
- **Sensor analytics prototype:** explore operating-data quality, anomaly detection, and cleaning-frequency prediction through an interactive Streamlit dashboard.

The sensor analytics workstream is exploratory. Its dataset provenance and target-generation process have not yet been independently verified, so its results should not be interpreted as validated industrial predictions.

## Project objectives

- Load, clean, and standardize experimental desalination data.
- Compare operating conditions using pressure, permeate flow, and permeate salinity.
- Calculate water recovery from feed and permeate flow measurements.
- Investigate relationships between operating parameters.
- Build an interactive dashboard for operational-data exploration.
- Detect unusual sensor records using Isolation Forest.
- Evaluate cleaning-frequency prediction against a simple baseline.
- Audit missingness, feature relationships, model stability, and data-quality discrepancies.
- Maintain reproducible analysis scripts and automated tests.

## Interactive dashboard

The Streamlit application provides five views:

1. **Executive overview** â€” a high-level summary of the available sensor data.
2. **Operating explorer** â€” exploration of operating conditions and relationships between variables.
3. **Cleaning intelligence** â€” exploratory cleaning-frequency prediction and model metrics.
4. **Anomaly intelligence** â€” inspection of unusual records flagged by an unsupervised anomaly-detection model.
5. **Data quality** â€” inspection of missing values and other data-quality issues.

### Run the dashboard

From the project root, activate the virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the application:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that address in your browser.

## Datasets and provenance

### Experimental wave-powered SWRO data

The experimental analysis uses measurements from a wave-powered SWRO desalination study. The processed operating-condition data is organized into three datasets:

| Dataset | Observations | Purpose |
|---|---:|---|
| Operating conditions | 117 | Compare steady, sinusoidal, and rectified-sinusoidal flow conditions |
| Constant-pressure experiment | 47 | Analyze a separate experiment with a different measurement schema |
| Membrane-integrity experiments | 59 | Explore measurements under different membrane conditions |

The operating-condition dataset contains 47 steady-flow observations, 45 sinusoidal observations across three nominal mean-flow settings, and 25 rectified-sinusoidal observations.

**Data provenance:** This experimental dataset is not proprietary DION+ plant data. It does not directly represent DION+'s membrane capacitive deionization (MCDI) or closed-circuit reverse osmosis (CCRO) technology.

### Sensor-data prototype

The sensor analytics workflow uses `data/raw/desalination_sensor_data.csv`, which contains 156,234 records and 14 columns in the inspected local version.

The dataset includes variables such as feed salinity, temperature, pressure, recovery, membrane age, SDI15, crossflow velocity, antiscalant dose, downtime, feed flow, specific energy, water production, and cleaning frequency.

The dataset's provenance and the mechanism used to generate its cleaning-frequency target have not been confirmed. Results therefore demonstrate exploratory modeling on the available dataset, not validated performance on a real desalination plant.

Raw datasets are not automatically included in version control. Before publishing or redistributing any dataset, verify its source, licence, and redistribution conditions.

## Technology stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Plotly
- Streamlit
- openpyxl
- pytest

## Project structure

```text
AquaPalladium/
├── data/
│   ├── raw/
│   â””── processed/
├── docs/
├── outputs/
├── src/
│   ├── analytics.py
│   ├── data_audit.py
│   ├── data_generator.py
│   ├── experimental_analytics.py
│   ├── load_wave_dataset.py
│   ├── period_analysis.py
│   ├── statistical_analysis.py
│   ├── visualization.py
│   ├── anomaly_detection.py
│   ├── data_quality_report.py
│   â””── ... exploratory audit scripts
├── tests/
├── app.py
├── main.py
├── requirements.txt
â””── README.md
```

Generated reports and intermediate outputs are stored under `outputs/`. The precise set of analysis scripts may evolve as the project develops.

## Experimental analysis workflow

Run these commands from the repository root when the required experimental workbook files are available:

```powershell
python -m src.load_wave_dataset
python -m src.experimental_analytics
python -m src.visualization
python -m src.period_analysis
python -m src.statistical_analysis
```

The workflow loads and standardizes Excel measurements, separates datasets with different schemas, calculates performance metrics, summarizes operating conditions, and produces visualizations.

## Sensor analytics and model evaluation

The project includes scripts for exploring cleaning-frequency prediction, anomaly detection, and data quality.

### Cleaning-frequency model audit

```powershell
python -m src.audit_cleaning_target
python -m src.audit_cleaning_target_patterns
python -m src.audit_cleaning_robustness
```

These scripts examine target distributions, feature relationships, missingness, model performance against a median baseline, permutation importance, and sensitivity to different random train/test splits.

### Anomaly detection and data quality

The repository also contains scripts for Isolation Forest anomaly detection, anomaly stability comparisons, production-flow discrepancy analysis, and data-quality reporting. Run individual scripts from the repository root after checking their docstrings for expected inputs and output locations.

## Exploratory model results

In the inspected version of the sensor dataset, the cleaning-frequency Random Forest evaluation produced the following results across five random 80/20 train/test splits:

| Model | Mean MAE | MAE standard deviation | Mean R² |
|---|---:|---:|---:|
| Median baseline | 1.0799 | 0.0076 | Approximately 0 |
| Six-feature model | 0.6387 | 0.0030 | 0.6504 |
| Eleven-feature model | 0.6335 | 0.0029 | 0.6570 |

The eleven-feature model improves on the median baseline in these evaluations. The six-feature model performs similarly with fewer predictors.

These are exploratory random-holdout results. They do not establish generalization to future operating periods, other desalination plants, or real-world cleaning schedules. Feature importance describes the fitted model's predictive reliance on variables; it does not establish causality.

## Initial experimental findings

The experimental SWRO analysis indicates that:

- Sinusoidal operation has the highest mean permeate flow and lowest mean permeate salinity among the three operating-condition groups in this dataset.
- Within the sinusoidal experiments, higher nominal mean-flow settings are associated with higher observed permeate flow and lower average salinity.
- In the `Mean=10L_min` experiment, amplitude settings form distinct groups with different feed pressures, permeate flows, and salinity levels.
- Within each amplitude group in that experiment, changes across measured periods are comparatively small and do not show consistent improvement across all metrics.

These findings are descriptive. They do not establish causation or identify a universally optimal operating condition.

## Methodology

The experimental workflow:

1. Reads Excel sheets and identifies measurement headers.
2. Removes blank separators and repeated header rows.
3. Standardizes measurement-column names and converts values to numeric types.
4. Separates datasets with different measurement schemas.
5. Calculates recovery as permeate flow divided by feed flow, multiplied by 100.
6. Summarizes measurements by operating condition.
7. Explores correlations and produces plots for performance comparison.

The sensor analytics workflow additionally applies data-quality checks, median imputation where configured, Random Forest regression for cleaning-frequency estimation, and Isolation Forest for unsupervised anomaly detection.

## Testing

Run the automated test suite:

```powershell
python -m pytest -q
```

Check installed dependency consistency:

```powershell
python -m pip check
```

Tests cover selected analytics, data generation, anomaly detection, and data-quality functionality.

## Limitations

- The experimental dataset is relatively small and comes from a specific experimental setup.
- Observations may be sequential measurements rather than independent experimental replicates.
- Operating parameters can change together, so correlations do not establish causal effects.
- Recovery calculated from flow measurements is not independent validation of those measurements.
- The sensor dataset's provenance and cleaning-target generation process remain unverified.
- Random holdout validation does not establish future-period or cross-plant performance.
- Anomaly flags identify unusual records, not confirmed equipment faults.
- The analysis does not yet establish a validated optimal operating condition or industrially validated cleaning schedule.

## Future work

- Improve automated validation of experimental data loaders.
- Investigate experimental effects while accounting for the measurement and experimental design.
- Verify the source and generation process of the sensor dataset.
- Evaluate cleaning-frequency predictions by target class and with time-aware validation where timestamps and suitable data are available.
- Test anomaly-detection stability and review flagged records against verified operational evidence.
- Improve documentation, reproducibility, and automated testing.
- Add validated operational data only when it can be used under appropriate access and redistribution conditions.

## License

See [`LICENSE`](LICENSE) for the project's code licensing terms. The project's code licence does not automatically grant redistribution rights for third-party datasets.
