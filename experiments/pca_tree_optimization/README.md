# PCA and Tree-Model Optimization Experiment

This folder keeps exploratory modeling work out of the main notebook pipeline.
It adds a small, reproducible experiment for the feedback items around baseline
quality, class imbalance, redundant macro features, compact lag choices,
two-stage labels, and tighter tree regularization.

## What This Tests

The experiment uses the existing `data/df_model.csv` file and preserves the
same expanding-window walk-forward validation protocol used elsewhere in the
repo. The main notebooks and production artifacts are not modified.

The script evaluates:

- **Aligned baselines**: majority class, last decision, and balanced
  multinomial logistic regression, all emitted on the same
  `[Lower, Same, Higher]` probability scale.
- **Class imbalance handling**: macro-F1 is the primary metric and tree models
  use balanced class weighting.
- **PCA replacement**: `unemployment_gap_lag1` and `output_gap_pct_lag1` are
  replaced by one walk-forward-fitted `cycle_gap_pc1` feature.
- **Hard feature selection**: one of the redundant pair is dropped instead of
  using PCA.
- **Compact economic concepts**: each macro concept keeps a lag-1 level plus a
  change feature instead of carrying lags 1, 2, 3, and 4 together.
- **Two-stage labels**: first classify `Same` versus `Changed`, then classify
  `Lower` versus `Higher` conditional on a predicted change.
- **Tighter trees**: lower depth, fewer leaves, larger minimum leaf size, and
  stronger L2 regularization.
- **Small CatBoost check**: a tight CatBoost variant on full features and on
  the PC1 replacement feature set, when CatBoost is installed.

The PCA process is intentionally fitted inside each walk-forward training fold.
This avoids using future observations to create the test-row PC1 value.

## Run

From the repository root:

```bash
/opt/anaconda3/bin/python experiments/pca_tree_optimization/run_experiment.py
```

Optional output directory:

```bash
/opt/anaconda3/bin/python experiments/pca_tree_optimization/run_experiment.py \
  --output-dir experiments/pca_tree_optimization/output
```

## Outputs

The script writes only inside this experiment folder by default:

- `metrics.csv`: model-level metrics for comparison.
- `predictions.csv`: long-form walk-forward predictions and probabilities.
- `pca_pair_loadings.csv`: final full-sample diagnostic loadings for the
  two-feature PC1.
- `pca_pair_meta.json`: PCA diagnostic metadata.
- `feature_sets.json`: feature columns used by each feature-set variant.
- `summary.md`: report-friendly notes and the ranked results table.

## Reading The PCA Result

The sign of PCA is arbitrary, so this experiment orients PC1 so that a higher
`cycle_gap_pc1` means a stronger real-economy cycle: higher output gap and lower
unemployment gap. That makes the single PC easier to interpret in downstream
tables.
