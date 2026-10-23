# AquaPalladium — Cleaning Frequency Prediction

## 1. Overview

AquaPalladium is a Python-based desalination analytics project that explores sensor data, identifies unusual operating patterns, and predicts cleaning frequency using machine learning.

This document describes the cleaning-frequency prediction component, its evaluation, the features that contribute to predictions, and the limitations that must be considered before interpreting results operationally.

**Important:** The available dataset's provenance and target-generation process have not been verified. Results should therefore be treated as exploratory machine-learning findings, not validated recommendations for a real desalination plant.

## 2. Objective

The objective is to predict `Cleaning_Frequency_per_year` from available desalination operating parameters.

The model is intended to demonstrate how historical sensor data can be used to estimate a maintenance-related target and how predictions can be evaluated against simple baselines.

This is a supervised regression task because the target is represented by numerical values from 0 to 6.

## 3. Dataset

The primary dataset is:

`data/raw/desalination_sensor_data.csv`

The dataset contains approximately 156,234 records and 14 columns.

### Target variable

`Cleaning_Frequency_per_year`

The target has seven observed values, ranging from 0 to 6.

| Cleaning frequency | Percentage of usable records |
|---:|---:|
| 0 | 13.50% |
| 1 | 27.07% |
| 2 | 27.01% |
| 3 | 17.97% |
| 4 | 9.13% |
| 5 | 3.66% |
| 6 | 1.66% |

There are 148,408 records with a usable target value.

The mean cleaning frequency is approximately 2.00 per year, and the median is 2.

### Input features

The model uses the following 11 predictors:

- `Feed_Salinity_mgL`
- `Feed_Temperature_C`
- `Feed_Pressure_bar`
- `Recovery_Pct`
- `Membrane_Age_months`
- `SDI15`
- `Crossflow_Velocity_m_s`
- `pH`
- `Antiscalant_Dose_mgL`
- `System_Downtime_hr_per_month`
- `Feed_Flow_m3h`

`Specific_Energy_kWh_per_m3` is excluded because it is constant at 5.0 for all non-missing observations and therefore provides no useful variation for prediction.

`Water_Production_m3h` is not included as an input feature.

## 4. Model and preprocessing

The current model is a Random Forest Regressor.

Random Forest combines predictions from multiple decision trees. Each tree learns patterns from the training data, and the final prediction aggregates the individual tree predictions.

The implementation uses:

- Median imputation for missing input values.
- A Random Forest Regressor.
- A fixed random seed for reproducibility.
- A random train/test split for evaluation.

The robustness audit also evaluates the model using multiple random split seeds.

Median imputation replaces missing numerical values with the median learned from the training data. This allows the model to process incomplete records, although it can conceal useful information about missingness.

The evaluated model configurations use 35 trees in the cross-validation and robustness experiments. Other experiments use different configurations; metrics should always be reported alongside the specific evaluation procedure.

## 5. Evaluation metrics

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and actual cleaning frequency.

For example, an MAE of 0.64 means predictions differ from actual values by approximately 0.64 cleaning-frequency units on average.

Lower MAE is better.

### R² score

R² measures how much variation in the target is explained by the model relative to a mean-prediction baseline.

- An R² of 1 indicates perfect predictions.
- An R² of 0 indicates performance comparable to predicting the target mean.
- A negative R² indicates performance worse than that baseline on the evaluated data.

R² does not establish that the model captures real physical mechanisms.

## 6. Model evaluation results

The following results were obtained from experiments using the available dataset.

### Main model comparison

| Model | MAE | R² |
|---|---:|---:|
| Random Forest, all 11 features | 0.6353 | 0.6520 |
| Random Forest, six selected features | 0.6415 | 0.6449 |
| Random Forest, top three features | 0.8795 | 0.3733 |
| Median-prediction baseline | 1.0759 | Approximately 0 |
| Linear regression using feed flow only | 1.0791 | 0.0680 |
| Random Forest using feed flow only | 1.2028 | -0.1733 |

The six-feature model uses feed flow, membrane age, SDI15, feed pressure, feed salinity, and recovery.

The all-feature model performs better than the simple baselines in this random holdout experiment. The six-feature model achieves similar performance with fewer predictors.

The feed-flow-only models do not reproduce the performance of the multivariable models.

### Repeated holdout evaluation

The robustness audit repeated the 80/20 train/test split with five random seeds: 7, 21, 42, 84, and 123.

| Model | Mean MAE | MAE standard deviation | Mean R² |
|---|---:|---:|---:|
| Random Forest, all 11 features | 0.6335 | 0.0029 | 0.6570 |
| Random Forest, six selected features | 0.6387 | 0.0030 | 0.6504 |
| Median baseline | 1.0799 | 0.0076 | Approximately 0 |

The results are consistent across these random splits. However, repeated random holdouts do not test performance on a genuinely independent plant, a later time period, or a different data source.

### Cross-validation

A separate five-fold cross-validation experiment on a deterministic 25,000-record sample produced:

- Mean MAE: 0.720
- MAE standard deviation: 0.008
- Mean R²: 0.570
- R² standard deviation: 0.005
- Median-baseline MAE: 1.078

These figures differ from the random holdout results because the experiments use different evaluation procedures and sample sizes. They should not be presented as if they came from the same test set.

## 7. Feature importance

Permutation importance estimates how much a model's evaluation error increases when one feature's values are shuffled.

A larger increase in MAE suggests the model relies more heavily on that feature for its predictions.

The robustness audit produced the following approximate ranking:

| Feature | Approximate increase in test MAE |
|---|---:|
| Feed flow | 1.486 |
| Membrane age | 1.147 |
| SDI15 | 0.745 |
| Feed pressure | 0.351 |
| Feed salinity | 0.063 |
| Recovery | 0.032 |

The ranking was consistent across the five evaluated random splits.

These values are changes in MAE, not percentages and not causal effects.

A feature can be important to the model without having a strong individual correlation with the target. This can happen when the model learns nonlinear relationships or interactions between variables.

Feature importance does not prove that changing a particular operating parameter will cause cleaning frequency to change.

## 8. Target-pattern investigation

Exploratory analysis found a negative association between feed flow and cleaning frequency.

The Pearson correlation was approximately -0.270, while the Spearman correlation was approximately -0.241.

Across feed-flow quantile groups, mean cleaning frequency declined from approximately 2.56 in the lowest-flow group to 1.17 in the highest-flow group.

The relationship was not strictly monotonic across every group.

Other individual correlations, including those for membrane age and SDI15, were close to zero. Nevertheless, these features had substantial permutation importance in the fitted model.

This difference suggests that individual correlations alone do not describe all the patterns learned by the model. The underlying data-generation process remains unknown, so these patterns require further investigation.

## 9. Candidate model selection

The six-feature model is a promising candidate for the dashboard because it uses fewer inputs while retaining similar performance to the full model.

However, the observed differences are small, and the six-feature model has not been established as universally superior.

A final choice should consider:

- Performance on a fixed, reproducible test set.
- Cross-validation results.
- Prediction stability.
- Data leakage and target-generation audits.
- Ease of explaining the model.
- Whether the selected features are available and reliable in the intended application.

The all-feature model can remain the current benchmark until these checks are complete.

## 10. Known limitations

1. **Dataset provenance:** The origin and collection methodology have not been verified.
2. **Target validity:** It is unknown whether the cleaning-frequency values represent observed maintenance records, generated values, or another process.
3. **Generalization:** Random splits may overestimate performance on future operating periods or other plants.
4. **Missing values:** Median imputation can obscure informative missingness patterns.
5. **Correlation versus causation:** Predictive importance does not establish physical causality.
6. **Operational validation:** The model has not been validated against confirmed maintenance outcomes from a real desalination plant.
7. **Prediction interpretation:** The regressor can return fractional predictions even though the recorded target values are integers from 0 to 6.
8. **Data quality:** Other dataset issues, including a constant specific-energy column and recovery values concentrated at 50%, require investigation.

Consequently, the model should not yet be used to prescribe cleaning schedules or make operational maintenance decisions.

## 11. Implementation and reproducibility

Relevant project scripts include:

- `src/validate_cleaning_model.py` — cross-validation experiment.
- `src/audit_cleaning_target.py` — target and baseline audit.
- `src/analyze_cleaning_relationships.py` — exploratory relationship analysis.
- `src/compare_cleaning_models.py` — comparison of feature sets and baselines.
- `src/compare_cleaning_feature_sets.py` — feature-set comparison.
- `src/audit_cleaning_robustness.py` — repeated holdout and permutation-importance audit.
- `src/audit_cleaning_target_patterns.py` — target distribution and feature-pattern analysis.
- `src/compare_flow_conditional_models.py` — comparison with feed-flow-based baselines.

Run scripts from the repository root in the project's activated Python virtual environment. For example:

```powershell
python src/validate_cleaning_model.py
python src/audit_cleaning_robustness.py
```

Exact results may vary if the dataset, preprocessing, model configuration, or evaluation procedure changes.

## 12. Next steps

Before presenting the cleaning model as a finished portfolio feature:

1. Integrate the selected model into the Streamlit dashboard.
2. Verify that the displayed metrics correspond to the correct evaluation experiment.
3. Test the full application, including anomaly detection and data-quality pages.
4. Review the data-generation process and investigate possible leakage.
5. Add appropriate explanations for predictions and feature importance.
6. Document the dataset's provenance as soon as it can be established.
7. Consider a time-based evaluation if reliable timestamps and suitable data coverage are available.
8. Keep the model framed as an exploratory prediction tool until it is independently validated.

## Conclusion

AquaPalladium's cleaning-frequency model demonstrates measurable predictive performance on the available dataset and performs substantially better than a median-prediction baseline in the evaluated random splits.

The results justify further investigation and dashboard integration. They do not, by themselves, establish that the model can predict real-world membrane-cleaning requirements.

The immediate priority is to finish the dashboard, reproduce the evaluation results, document the limitations, and prepare a clear demonstration of the complete project.
