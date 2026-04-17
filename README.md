# Forecasting Federal Funds Rate

This project provides a machine learning framework for forecasting Federal Open Market Committee (FOMC) rate decisions (**Higher/Raise**, **Same/Hold**, **Lower/Cut**). It leverages historical macroeconomic indicators and learned policy regimes, and evaluates various models using strict time-series walk-forward validation to ensure no future data leakage.

## Codebase Structure

### `data/`
Contains both raw macroeconomic datasets downloaded from FRED and the processed datasets ready for model training:
- **Raw Data**: `CPI All Urban Consumers.csv`, `Unemployment Rate UNRATE.csv`, `WTI Crude Oil Prices.csv`, and `Federal Funds Target Rate.csv`.
- **Processed Data**: Files like `df_model.csv`, `fedrate_all.csv`, and `state_all.csv` which contain the merged, normalized, and engineered features.

### `analysis/`
Contains Jupyter Notebooks detailing the data processing pipeline, modeling experiments, and evaluation steps.

#### Data Preparation & Feature Engineering
- **`01_process_fed_rate.ipynb`**: Processes raw FOMC meeting/rate data and exports the clean meeting-level series to `data/processed_fed_meetings.csv`.
- **`02_process_economic_variables.ipynb`**: Joins the processed meeting calendar to macroeconomic inputs and prepares the economic variables used downstream for modeling.
- **`03.1_state_identification.ipynb`**: Identifies economic policy regimes (tightening, plateau, easing) directly from the Federal Funds Rate series using a Dirichlet-Multinomial drift-score model, creating soft probability states that are then used as features.

#### Predictive Modeling
- **`baseline_ordinal_logit.ipynb` / `baseline_multinomial_logit.ipynb`**: Baseline logistic regression models setup to predict the ordered or multi-class FOMC outcomes and output probabilities.
- **`03_boosting_models_trained_on_economic_variables.ipynb`**: The core predictive pipeline. It trains four gradient boosting classifiers (XGBoost, LightGBM, CatBoost, HistGradientBoosting) using walk-forward validation and performs Bayesian hyperparameter tuning using Optuna.
- **`03.2_compare_regime_feature_as_input.ipynb`**: Compares the tuned boosting models when trained with versus without the engineered regime-state feature.
- **`04_construct_calibrated_likelihood_layer.ipynb`**: Converts walk-forward boosting model probabilities into a calibrated likelihood layer and exports the calibration artifacts used downstream.
- **`05_bayesian_update.ipynb`**: Combines market prior information with the calibrated likelihood layer to produce a posterior probability forecast for the next meeting.

#### Evaluation & Diagnostics
- **`04_construct_calibrated_likelihood_layer.ipynb`**: Also includes the supporting probability diagnostics, calibration checks, and model-comparison views used to validate the likelihood layer.

### `experiments/`
Contains isolated exploratory runs that should not mutate the main notebook
pipeline or production artifacts.

- **`pca_tree_optimization/`**: Adds the PCA replacement process for the
  redundant `unemployment_gap_lag1` / `output_gap_pct_lag1` pair, expands
  aligned baselines, and records tree-model optimization checks for class
  imbalance, hard feature selection, compact lag features, two-stage labels,
  and tighter regularization.

## Methodology Highlights
1. **Walk-Forward Validation**: Because the data represents a time-series forecasting problem, models are trained recursively in an expanding window. To predict meeting $t$, only data from $t-1$ and earlier is utilized.
2. **Probability Distributions**: Models don't just output discrete classes, but rather a probability distribution `[P(Lower), P(Same), P(Higher)]` which gives a measure of the model's confidence and expected policy direction.
3. **Regime Context**: Instead of solely relying on recent raw numbers, the models are fed "regime" states to understand the current longer-term momentum of monetary policy (e.g., if the Fed is consistently raising rates). 

## Reproducible Pipeline

Run the full notebook workflow, CatBoost artifact build, and isolated PCA/tree
optimization experiment with one command:

```bash
/opt/anaconda3/bin/python scripts/run_reproducible_pipeline.py
```

The runner executes notebooks in dependency order, stores executed notebook
copies and logs under `artifacts/pipeline_runs/<run-id>/`, and leaves a
`manifest.json` for report reproducibility. To inspect the planned commands
without running the full workflow:

```bash
/opt/anaconda3/bin/python scripts/run_reproducible_pipeline.py --dry-run
```

## Running Without Local Data Files

The active notebooks now load their CSV inputs from GitHub raw URLs under
`https://raw.githubusercontent.com/echochoho1010/forecasting_fed_rate/master/data/`,
so they can run even if the repository data files have not been downloaded locally.
