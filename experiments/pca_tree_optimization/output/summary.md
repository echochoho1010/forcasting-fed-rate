# PCA and Tree-Model Optimization Summary

Primary metric: macro-F1 on the expanding walk-forward window.

All models and baselines emit probabilities on the same `[Lower, Same, Higher]` scale.

## Ranked Results

| experiment | group | accuracy | balanced_accuracy | f1_macro | f1_weighted | pred_lower | pred_same | pred_higher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_last_decision | baseline | 0.6667 | 0.5831 | 0.5831 | 0.6667 | 11 | 65 | 20 |
| catboost_tight_full | tree | 0.5938 | 0.6762 | 0.5741 | 0.6113 | 15 | 34 | 47 |
| catboost_tight_pair_pc1 | tree | 0.5938 | 0.6647 | 0.5717 | 0.6131 | 15 | 36 | 45 |
| hgb_default_full | tree | 0.5104 | 0.6006 | 0.5012 | 0.5301 | 17 | 31 | 48 |
| hgb_tight_full | tree | 0.4167 | 0.5754 | 0.4263 | 0.3968 | 15 | 16 | 65 |
| hgb_tight_pair_pc1 | tree | 0.4167 | 0.5754 | 0.4263 | 0.3968 | 15 | 16 | 65 |
| hgb_tight_hard_select_unemployment_gap | tree | 0.4167 | 0.5754 | 0.4263 | 0.3968 | 15 | 16 | 65 |
| hgb_tight_hard_select_output_gap | tree | 0.4167 | 0.5754 | 0.4224 | 0.3959 | 16 | 16 | 64 |
| baseline_multinomial_logit_balanced | baseline | 0.4271 | 0.5711 | 0.4202 | 0.4280 | 35 | 22 | 39 |
| hgb_tight_concept_lag1_change | tree | 0.3750 | 0.5318 | 0.3826 | 0.3546 | 18 | 14 | 64 |
| hgb_tight_concept_lag1_change_pc1 | tree | 0.3750 | 0.5066 | 0.3682 | 0.3618 | 18 | 15 | 63 |
| two_stage_hgb_tight_pair_pc1 | two_stage | 0.3021 | 0.4928 | 0.3051 | 0.2962 | 55 | 12 | 29 |
| two_stage_hgb_tight_concept_pc1 | two_stage | 0.3021 | 0.4928 | 0.3051 | 0.2962 | 55 | 12 | 29 |
| baseline_majority_expanding | baseline | 0.6771 | 0.3333 | 0.2692 | 0.5467 | 0 | 96 | 0 |

## Experiment Notes

### baseline_majority_expanding

Expanding majority-class baseline with smoothed 3-class probabilities.

Macro-F1=0.2692, accuracy=0.6771, predicted counts Lower/Same/Higher=0/96/0.

### baseline_last_decision

Persistence baseline that repeats the previous FOMC decision.

Macro-F1=0.5831, accuracy=0.6667, predicted counts Lower/Same/Higher=11/65/20.

### baseline_multinomial_logit_balanced

Balanced multinomial logistic regression on the full feature set.

Macro-F1=0.4202, accuracy=0.4271, predicted counts Lower/Same/Higher=35/22/39.

### hgb_default_full

Current-style HistGradientBoosting setup on the full feature set.

Macro-F1=0.5012, accuracy=0.5104, predicted counts Lower/Same/Higher=17/31/48.

### hgb_tight_full

Tighter HistGradientBoosting on the full feature set.

Macro-F1=0.4263, accuracy=0.4167, predicted counts Lower/Same/Higher=15/16/65.

### hgb_tight_pair_pc1

Tighter tree with the redundant gap pair replaced by PC1.

Macro-F1=0.4263, accuracy=0.4167, predicted counts Lower/Same/Higher=15/16/65.

### hgb_tight_hard_select_output_gap

Tighter tree that keeps output_gap_pct_lag1 and drops unemployment_gap_lag1.

Macro-F1=0.4224, accuracy=0.4167, predicted counts Lower/Same/Higher=16/16/64.

### hgb_tight_hard_select_unemployment_gap

Tighter tree that keeps unemployment_gap_lag1 and drops output_gap_pct_lag1.

Macro-F1=0.4263, accuracy=0.4167, predicted counts Lower/Same/Higher=15/16/65.

### hgb_tight_concept_lag1_change

Tighter tree with compact lag-1 plus change features.

Macro-F1=0.3826, accuracy=0.3750, predicted counts Lower/Same/Higher=18/14/64.

### hgb_tight_concept_lag1_change_pc1

Compact lag-1 plus change features with the redundant gap pair replaced by PC1.

Macro-F1=0.3682, accuracy=0.3750, predicted counts Lower/Same/Higher=18/15/63.

### two_stage_hgb_tight_pair_pc1

Two-stage changed-vs-same then lower-vs-higher model using PC1 features.

Macro-F1=0.3051, accuracy=0.3021, predicted counts Lower/Same/Higher=55/12/29.

### two_stage_hgb_tight_concept_pc1

Two-stage model using compact concept features and PC1.

Macro-F1=0.3051, accuracy=0.3021, predicted counts Lower/Same/Higher=55/12/29.

### catboost_tight_full

Small tight CatBoost check on the full feature set.

Macro-F1=0.5741, accuracy=0.5938, predicted counts Lower/Same/Higher=15/34/47.

### catboost_tight_pair_pc1

Small tight CatBoost check with the redundant gap pair replaced by PC1.

Macro-F1=0.5717, accuracy=0.5938, predicted counts Lower/Same/Higher=15/36/45.
