from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path("/Users/echohuang/Documents/forecasting_fed_rate")
ANALYSIS = ROOT / "analysis"


CALIBRATION_MD = """---
## 9. Decile Calibration Curves and Export

This section converts the walk-forward probability outputs into a reusable decile-based calibration map.

For each model and class:
- predicted probabilities are grouped into fixed decile buckets
- the realized hit rate is computed inside each bucket
- the calibration table is exported to `calibration_map.csv`

The exported mapping is later consumed by `05_bayesian_update.ipynb` as a calibrated likelihood layer.
"""


CALIBRATION_CODE = """# ── 9. Decile calibration map + multi-model calibration curves ───────────────
from pathlib import Path

CALIBRATION_BUCKETS = np.linspace(0.0, 1.0, 11)
CLASS_ORDER = ['Lower', 'Same', 'Higher']


def assign_decile_bucket(prob):
    bucket_idx = min(int(np.floor(prob * 10)), 9)
    bucket_low = bucket_idx / 10
    bucket_high = bucket_low + 0.1
    return bucket_low, bucket_high


calibration_rows = []
for model_name, result in final_results.items():
    probs = np.asarray(result['probas'], dtype=float)
    acts = np.asarray(result['actuals_enc'], dtype=int)
    y_onehot = label_binarize(acts, classes=[0, 1, 2])

    for class_idx, class_name in enumerate(CLASS_ORDER):
        class_probs = probs[:, class_idx]
        bucket_idx = np.minimum((class_probs * 10).astype(int), 9)

        for decile in range(10):
            bucket_low = decile / 10
            bucket_high = bucket_low + 0.1
            mask = bucket_idx == decile
            count = int(mask.sum())

            predicted_mean = float(class_probs[mask].mean()) if count else np.nan
            actual_freq = float(y_onehot[mask, class_idx].mean()) if count else np.nan

            calibration_rows.append({
                'model': model_name,
                'class': class_name,
                'bucket_low': bucket_low,
                'bucket_high': bucket_high,
                'predicted_mean': predicted_mean,
                'actual_freq': actual_freq,
                'count': count,
            })

df_calibration = pd.DataFrame(calibration_rows)
calibration_out = Path('calibration_map.csv')
df_calibration.to_csv(calibration_out, index=False)

print(f'Calibration map saved to {calibration_out.resolve()}')
display(df_calibration.head(12))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
palette = sns.color_palette('tab10', n_colors=len(final_results))

for class_idx, class_name in enumerate(CLASS_ORDER):
    ax = axes[class_idx]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Perfect calibration')

    for color, (model_name, _) in zip(palette, final_results.items()):
        df_plot = (
            df_calibration[
                (df_calibration['model'] == model_name) &
                (df_calibration['class'] == class_name) &
                (df_calibration['count'] > 0)
            ]
            .sort_values('bucket_low')
        )
        if df_plot.empty:
            continue

        ax.plot(
            df_plot['predicted_mean'],
            df_plot['actual_freq'],
            marker='o',
            linewidth=1.8,
            color=color,
            label=model_name,
        )

    ax.set_title(f'{class_name} calibration')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

axes[0].legend(fontsize=8, loc='upper left')
fig.suptitle('Decile calibration curves by class', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
"""


REGIME_ACCURACY_MD = """---
## 12. Regime-Specific Accuracy Export

This section summarizes accuracy by policy regime using the same walk-forward predictions already computed above.

For each model variant, the notebook:
- aligns each out-of-sample meeting with its pre-meeting realtime regime label
- computes accuracy within each regime bucket
- exports the result to `regime_accuracy_by_model.csv`

`05_bayesian_update.ipynb` can then use the current regime to up-weight or down-weight the model likelihood.
"""


REGIME_ACCURACY_CODE = """# ── 12. Regime-specific accuracy export ───────────────────────────────────────
from pathlib import Path

regime_eval = (
    df_regime.loc[INITIAL_TRAIN_SIZE:, ['meeting_date', 'regime_simple']]
    .reset_index(drop=True)
    .copy()
)

regime_accuracy_rows = []
for model_name, result in all_results.items():
    correct = (np.asarray(result['actuals_enc']) == np.asarray(result['preds_enc'])).astype(int)
    if len(correct) != len(regime_eval):
        raise ValueError(
            f'Regime alignment mismatch for {model_name}: '
            f'{len(correct)} predictions vs {len(regime_eval)} regime rows'
        )

    df_eval = regime_eval.copy()
    df_eval['correct'] = correct

    for regime_name, group in df_eval.groupby('regime_simple', dropna=False):
        regime_accuracy_rows.append({
            'model': model_name,
            'regime_simple': regime_name,
            'accuracy': float(group['correct'].mean()),
            'count': int(len(group)),
        })

df_regime_accuracy = (
    pd.DataFrame(regime_accuracy_rows)
    .sort_values(['model', 'regime_simple'])
    .reset_index(drop=True)
)

regime_accuracy_out = Path('regime_accuracy_by_model.csv')
df_regime_accuracy.to_csv(regime_accuracy_out, index=False)

print(f'Regime accuracy table saved to {regime_accuracy_out.resolve()}')
display(df_regime_accuracy)
"""


NEW_NOTEBOOK_CELLS = [
    ("markdown", """# 05 Bayesian Update

This notebook combines three information sources into a final posterior probability for the next FOMC decision:

1. a market-implied prior from CME FedWatch
2. a calibrated likelihood from the latest boosting-model output
3. dynamic weighting based on meeting proximity and regime-specific model accuracy

The output is a normalized posterior distribution over:
- Lower
- Same
- Higher
"""),
    ("code", """import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from IPython.display import display

sns.set_style('whitegrid')
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}
CLASS_ORDER = ['Lower', 'Same', 'Higher']
CLASS_TO_INDEX = {'Lower': 0, 'Same': 1, 'Higher': 2}
INITIAL_TRAIN_SIZE = 40

NOTEBOOK_ROOT = Path.cwd()
PROJECT_ROOT_CANDIDATES = [
    NOTEBOOK_ROOT,
    NOTEBOOK_ROOT.parent,
    Path('/Users/echohuang/Documents/forecasting_fed_rate'),
]
PROJECT_ROOT = next((path.resolve() for path in PROJECT_ROOT_CANDIDATES if path.exists()), NOTEBOOK_ROOT.resolve())
print(f'Working directory: {NOTEBOOK_ROOT.resolve()}')
print(f'Project root     : {PROJECT_ROOT}')
"""),
    ("markdown", """## 1. Load calibration data and the latest walk-forward model output

This section loads:
- the decile calibration map exported from `04_construct_calibrated_likelihood_layer.ipynb`
- the cached walk-forward probability outputs from the same likelihood-layer workflow

XGBoost is used as the primary model when available.
If XGBoost is unavailable, the notebook falls back to the average probability vector across all available boosting models.
"""),
    ("code", """def resolve_existing_path(*candidates, required=True):
    search_roots = [NOTEBOOK_ROOT, PROJECT_ROOT, PROJECT_ROOT / 'analysis', PROJECT_ROOT / 'data']
    seen = set()

    for candidate in candidates:
        candidate_path = Path(candidate)
        expanded = [candidate_path]
        if not candidate_path.is_absolute():
            expanded.extend(root / candidate_path for root in search_roots)

        for path in expanded:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                return resolved

    if required:
        raise FileNotFoundError(f'None of the candidate paths exist: {candidates}')
    return None


cache_path = resolve_existing_path(
    'final_results_cache.pkl',
    'data/final_results_cache.pkl',
    'analysis/final_results_cache.pkl',
)
df_model_path = resolve_existing_path(
    'data/df_model.csv',
    'df_model.csv',
    'analysis/df_model.csv',
)
calibration_path = resolve_existing_path(
    'calibration_map.csv',
    'data/calibration_map.csv',
    'analysis/calibration_map.csv',
    required=True,
)

model_cache = pd.read_pickle(cache_path)
final_results = model_cache['final_results']
df_model = pd.read_csv(df_model_path, parse_dates=['meeting_date'])
calibration_map = pd.read_csv(calibration_path)

available_models = list(final_results.keys())
print(f'Calibration map: {calibration_path.resolve()}')
print(f'Model cache     : {cache_path.resolve()}')
print(f'Available models: {available_models}')

if 'XGBoost' in final_results:
    selected_model = 'XGBoost'
    latest_model_probs = np.asarray(final_results['XGBoost']['probas'][-1], dtype=float)
    likelihood_source = 'XGBoost'
else:
    selected_model = 'Ensemble'
    prob_stack = np.stack([np.asarray(final_results[name]['probas'][-1], dtype=float)
                           for name in available_models])
    latest_model_probs = prob_stack.mean(axis=0)
    likelihood_source = 'Ensemble average'

latest_meeting_date = pd.to_datetime(df_model['meeting_date'].iloc[-1])
print(f'Selected likelihood source: {likelihood_source}')
print(f'Latest walk-forward meeting: {latest_meeting_date.date()}')
print('Latest raw model probabilities:')
display(pd.DataFrame([latest_model_probs], columns=CLASS_ORDER))
"""),
    ("markdown", """## 2. Fetch the CME FedWatch prior for the next meeting

The notebook first attempts to fetch a market-implied prior from the CME endpoint:

`https://www.cmegroup.com/CmeWS/mvc/ProductCalendar/Future/SR3`

Because the exact response structure may change, the parser is intentionally defensive.
If the API is unavailable or the payload cannot be parsed into `Lower / Same / Higher`, the notebook falls back to:
- a manually provided prior dictionary, or
- interactive user input inside the notebook
"""),
    ("code", """CME_API_URL = 'https://www.cmegroup.com/CmeWS/mvc/ProductCalendar/Future/SR3'
MANUAL_PRIOR = None
MANUAL_MEETING_DATE = None


def _as_float(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace('%', '').replace(',', '')
        if cleaned == '':
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _slug(text):
    return ''.join(ch.lower() for ch in str(text) if ch.isalnum())


def _walk_objects(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_objects(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_objects(item)


def _parse_meeting_date(candidate):
    try:
        return pd.to_datetime(candidate).normalize()
    except Exception:
        return None


def normalize_probability_vector(prob_dict):
    ordered = np.array([float(prob_dict[name]) for name in CLASS_ORDER], dtype=float)
    if np.all(ordered >= 0) and ordered.sum() > 1.0001:
        ordered = ordered / 100.0
    if np.any(ordered < 0):
        raise ValueError('Probabilities must be non-negative.')
    if ordered.sum() <= 0:
        raise ValueError('Probability vector must have a positive sum.')
    ordered = ordered / ordered.sum()
    return dict(zip(CLASS_ORDER, ordered))


def parse_cme_prior(payload):
    key_map = {
        'lower': 'Lower',
        'cut': 'Lower',
        'cuts': 'Lower',
        'easing': 'Lower',
        'same': 'Same',
        'hold': 'Same',
        'holds': 'Same',
        'nochange': 'Same',
        'unchanged': 'Same',
        'higher': 'Higher',
        'hike': 'Higher',
        'hikes': 'Higher',
        'raise': 'Higher',
        'raises': 'Higher',
        'tightening': 'Higher',
    }

    for obj in _walk_objects(payload):
        mapped = {}
        meeting_date = None

        for key, value in obj.items():
            value_num = _as_float(value)
            key_slug = _slug(key)

            if meeting_date is None and 'date' in key_slug:
                parsed_date = _parse_meeting_date(value)
                if parsed_date is not None:
                    meeting_date = parsed_date

            for alias, class_name in key_map.items():
                if alias in key_slug and value_num is not None:
                    mapped[class_name] = value_num

        if set(mapped) == set(CLASS_ORDER):
            return normalize_probability_vector(mapped), meeting_date

    raise ValueError('Could not parse Lower / Same / Higher probabilities from CME payload.')


def prompt_manual_prior():
    print('CME API unavailable. Please enter manual prior probabilities for the next FOMC meeting.')
    lower = float(input('P(Lower): '))
    same = float(input('P(Same): '))
    higher = float(input('P(Higher): '))
    meeting_date = input('Next meeting date (YYYY-MM-DD): ').strip()
    return normalize_probability_vector({'Lower': lower, 'Same': same, 'Higher': higher}), pd.to_datetime(meeting_date).normalize()


try:
    response = requests.get(
        CME_API_URL,
        headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'},
        timeout=20,
    )
    response.raise_for_status()
    cme_payload = response.json()
    prior_probs, cme_meeting_date = parse_cme_prior(cme_payload)
    print('Loaded prior from CME FedWatch API.')
except Exception as exc:
    print(f'CME API request or parsing failed: {exc}')
    if MANUAL_PRIOR is not None and MANUAL_MEETING_DATE is not None:
        prior_probs = normalize_probability_vector(MANUAL_PRIOR)
        cme_meeting_date = pd.to_datetime(MANUAL_MEETING_DATE).normalize()
        print('Using MANUAL_PRIOR and MANUAL_MEETING_DATE fallback.')
    else:
        try:
            prior_probs, cme_meeting_date = prompt_manual_prior()
        except Exception as input_exc:
            raise RuntimeError(
                'CME API failed and manual input was not supplied. '
                'Set MANUAL_PRIOR and MANUAL_MEETING_DATE, or rerun this cell interactively.'
            ) from input_exc

display(pd.DataFrame([prior_probs]))
print(f'CME/Manual next meeting date: {pd.Timestamp(cme_meeting_date).date()}')
"""),
    ("markdown", """## 3. Load the current regime and regime-specific model accuracy

The notebook expects `regime_accuracy_by_model.csv` exported from `03.2_compare_regime_feature_as_input.ipynb`.
It uses that file to adjust the likelihood weight by the model's historical performance in the current regime.
"""),
    ("code", """fedrate_path = resolve_existing_path('data/fedrate_all.csv', 'fedrate_all.csv')
state_path = resolve_existing_path('data/state_all.csv', 'state_all.csv')
regime_accuracy_candidates = [
    Path('regime_accuracy_by_model.csv'),
    Path('data/regime_accuracy_by_model.csv'),
    Path('analysis/regime_accuracy_by_model.csv'),
]

fedrate_all = pd.read_csv(fedrate_path, parse_dates=['observation_date'])
state_all = pd.read_csv(state_path, parse_dates=['observation_date'])

fedrate_all_with_regime = fedrate_all.merge(
    state_all[['observation_date', 'regime_rt']].rename(columns={'regime_rt': 'regime_simple'}),
    on='observation_date',
    how='left',
)
fedrate_all_with_regime = fedrate_all_with_regime.sort_values('observation_date').reset_index(drop=True)

current_regime = fedrate_all_with_regime['regime_simple'].dropna().iloc[-1]
current_regime_date = fedrate_all_with_regime['observation_date'].dropna().iloc[-1]
print(f'Current realtime regime: {current_regime} (as of {current_regime_date.date()})')

regime_accuracy_path = None
for candidate in regime_accuracy_candidates:
    if candidate.exists():
        regime_accuracy_path = candidate
        break

if regime_accuracy_path is None:
    raise FileNotFoundError(
        'regime_accuracy_by_model.csv not found. Run 03.2_compare_regime_feature_as_input.ipynb first.'
    )

regime_accuracy = pd.read_csv(regime_accuracy_path)
print(f'Loaded regime accuracy from {regime_accuracy_path.resolve()}')

display(regime_accuracy)


def resolve_regime_accuracy(regime_accuracy_table, model_name, current_regime_name, fallback_accuracy):
    candidate_names = []
    if model_name == 'Ensemble':
        candidate_names = ['Ensemble']
    else:
        candidate_names = [
            f'{model_name} (+ regime)',
            f'{model_name} (base)',
            model_name,
        ]

    for candidate_name in candidate_names:
        subset = regime_accuracy_table[
            (regime_accuracy_table['model'] == candidate_name) &
            (regime_accuracy_table['regime_simple'] == current_regime_name)
        ]
        if not subset.empty:
            return float(subset.iloc[0]['accuracy']), candidate_name

    return float(fallback_accuracy), 'overall fallback'


overall_accuracy = (
    float(final_results['XGBoost']['accuracy'])
    if 'XGBoost' in final_results
    else float(np.mean([final_results[name]['accuracy'] for name in available_models]))
)
current_regime_accuracy, regime_accuracy_source = resolve_regime_accuracy(
    regime_accuracy,
    selected_model,
    current_regime,
    overall_accuracy,
)
print(f'Regime accuracy source : {regime_accuracy_source}')
print(f'Current regime accuracy: {current_regime_accuracy:.4f}')
"""),
    ("markdown", """## 4. Convert the latest model output into a calibrated likelihood and compute dynamic weights

The calibrated likelihood is produced by:
- taking the latest model probability for each class
- placing it into the corresponding decile bucket
- replacing the raw probability with the observed hit rate for that bucket

Dynamic weighting uses:
- time-to-meeting: closer meetings put more weight on the market prior
- regime-specific accuracy: stronger model performance in the current regime increases the likelihood weight

The regime adjustment uses a small multiplicative heuristic:
- `+20%` to the likelihood weight if regime accuracy is above `0.60`
- `-20%` if regime accuracy is below `0.40`
"""),
    ("code", """def find_next_meeting_date(cme_date=None, manual_date=None):
    today = pd.Timestamp.today().normalize()
    candidate_dates = []

    if cme_date is not None:
        candidate_dates.append(pd.to_datetime(cme_date).normalize())

    processed_candidates = [
        Path('data/processed_fed_meetings.csv'),
        Path('processed_fed_meetings.csv'),
    ]
    for candidate in processed_candidates:
        if candidate.exists():
            df_meetings = pd.read_csv(candidate, parse_dates=['meeting_date'])
            future_dates = df_meetings.loc[df_meetings['meeting_date'] >= today, 'meeting_date']
            candidate_dates.extend(list(pd.to_datetime(future_dates).dt.normalize()))
            break

    if manual_date is not None:
        candidate_dates.append(pd.to_datetime(manual_date).normalize())

    candidate_dates = [date for date in candidate_dates if pd.notna(date) and date >= today]
    if not candidate_dates:
        raise RuntimeError(
            'Could not determine the next meeting date from CME or the local meeting calendar. '
            'Provide MANUAL_MEETING_DATE and rerun the notebook.'
        )

    return min(candidate_dates)


def lookup_calibrated_value(calibration_table, model_name, class_name, probability_value):
    bucket_low = np.floor(min(max(probability_value, 0.0), 0.999999) * 10) / 10
    bucket_high = bucket_low + 0.1

    exact = calibration_table[
        (calibration_table['model'] == model_name) &
        (calibration_table['class'] == class_name) &
        np.isclose(calibration_table['bucket_low'], bucket_low) &
        np.isclose(calibration_table['bucket_high'], bucket_high)
    ]

    if not exact.empty and int(exact.iloc[0]['count']) > 0 and pd.notna(exact.iloc[0]['actual_freq']):
        return float(exact.iloc[0]['actual_freq'])

    populated = calibration_table[
        (calibration_table['model'] == model_name) &
        (calibration_table['class'] == class_name) &
        (calibration_table['count'] > 0) &
        calibration_table['actual_freq'].notna()
    ].copy()

    if populated.empty:
        return float(probability_value)

    populated['distance'] = (populated['predicted_mean'] - probability_value).abs()
    populated = populated.sort_values(['distance', 'count'], ascending=[True, False])
    return float(populated.iloc[0]['actual_freq'])


def build_calibrated_likelihood():
    if selected_model != 'Ensemble':
        calibrated = [
            lookup_calibrated_value(calibration_map, selected_model, class_name, latest_model_probs[idx])
            for idx, class_name in enumerate(CLASS_ORDER)
        ]
        calibrated = np.array(calibrated, dtype=float)
        return calibrated / calibrated.sum()

    per_model = []
    for model_name in available_models:
        latest_probs = np.asarray(final_results[model_name]['probas'][-1], dtype=float)
        calibrated = [
            lookup_calibrated_value(calibration_map, model_name, class_name, latest_probs[idx])
            for idx, class_name in enumerate(CLASS_ORDER)
        ]
        per_model.append(np.array(calibrated, dtype=float))

    calibrated = np.vstack(per_model).mean(axis=0)
    return calibrated / calibrated.sum()


next_meeting_date = find_next_meeting_date(cme_date=cme_meeting_date, manual_date=MANUAL_MEETING_DATE)
today = pd.Timestamp.today().normalize()
days_to_meeting = int((next_meeting_date - today).days)

prior_weight = float(np.clip(1.0 - (days_to_meeting / 42.0), 0.2, 0.8))
likelihood_weight = 1.0 - prior_weight

if current_regime_accuracy > 0.60:
    likelihood_weight = min(0.8, likelihood_weight * 1.2)
elif current_regime_accuracy < 0.40:
    likelihood_weight = max(0.2, likelihood_weight * 0.8)
prior_weight = 1.0 - likelihood_weight

prior_vector = np.array([prior_probs[name] for name in CLASS_ORDER], dtype=float)
likelihood_vector = build_calibrated_likelihood()
raw_posterior = (prior_weight * prior_vector) + (likelihood_weight * likelihood_vector)
posterior_vector = raw_posterior / raw_posterior.sum()

summary_table = pd.DataFrame({
    'class': CLASS_ORDER,
    'P_prior': prior_vector,
    'P_likelihood': likelihood_vector,
    'P_posterior': posterior_vector,
})

print(f'Today               : {today.date()}')
print(f'Next meeting        : {next_meeting_date.date()}')
print(f'Days to meeting     : {days_to_meeting}')
print(f'Current regime      : {current_regime}')
print(f'Prior weight        : {prior_weight:.4f}')
print(f'Likelihood weight   : {likelihood_weight:.4f}')
print(f'Likelihood source   : {likelihood_source}')
print(f'Regime acc. source  : {regime_accuracy_source}')
print(f'Regime accuracy     : {current_regime_accuracy:.4f}')
display(summary_table)
"""),
    ("markdown", """## 5. Plot the prior, likelihood, and posterior side by side, then export the final output

The final output contains one row with:
- the current date
- days to the next meeting
- the current realtime regime
- the dynamic weights
- the prior / likelihood / posterior probability vector
"""),
    ("code", """plot_table = summary_table.melt(
    id_vars='class',
    value_vars=['P_prior', 'P_likelihood', 'P_posterior'],
    var_name='distribution',
    value_name='probability',
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=plot_table,
    x='class',
    y='probability',
    hue='distribution',
    palette=['#4c78a8', '#f58518', '#54a24b'],
    ax=ax,
)
ax.set_title('Prior vs calibrated likelihood vs posterior')
ax.set_xlabel('Decision class')
ax.set_ylabel('Probability')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

output_row = pd.DataFrame([{
    'date': today.date().isoformat(),
    'days_to_meeting': days_to_meeting,
    'current_regime': current_regime,
    'prior_weight': prior_weight,
    'likelihood_weight': likelihood_weight,
    'p_prior_lower': prior_vector[0],
    'p_prior_same': prior_vector[1],
    'p_prior_higher': prior_vector[2],
    'p_likelihood_lower': likelihood_vector[0],
    'p_likelihood_same': likelihood_vector[1],
    'p_likelihood_higher': likelihood_vector[2],
    'p_posterior_lower': posterior_vector[0],
    'p_posterior_same': posterior_vector[1],
    'p_posterior_higher': posterior_vector[2],
}])

bayesian_out = Path('bayesian_output.csv')
output_row.to_csv(bayesian_out, index=False)
print(f'Bayesian output saved to {bayesian_out.resolve()}')
display(output_row)
"""),
]


def ensure_cell(notebook, marker_text, new_cells):
    for idx, cell in enumerate(notebook.cells):
        if marker_text in cell.source:
            notebook.cells[idx:idx + len(new_cells)] = new_cells
            return
    notebook.cells.extend(new_cells)


def patch_evaluating_notebook():
    path = ANALYSIS / "04_construct_calibrated_likelihood_layer.ipynb"
    nb = nbformat.read(path, as_version=4)
    new_cells = [
        new_markdown_cell(CALIBRATION_MD),
        new_code_cell(CALIBRATION_CODE),
    ]
    ensure_cell(nb, "## 9. Decile Calibration Curves and Export", new_cells)
    nbformat.write(nb, path)


def patch_compare_notebook():
    path = ANALYSIS / "03.2_compare_regime_feature_as_input.ipynb"
    nb = nbformat.read(path, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code" and "def augment_missing_classes" in cell.source:
            cell.source = cell.source.replace(
                "    w_synth = np.full(len(missing), 1e-9)\n",
                "    synth_weight = float(real_w.min()) * 0.1\n"
                "    w_synth = np.full(len(missing), synth_weight, dtype=float)\n",
            )
            break

    new_cells = [
        new_markdown_cell(REGIME_ACCURACY_MD),
        new_code_cell(REGIME_ACCURACY_CODE),
    ]
    ensure_cell(nb, "## 12. Regime-Specific Accuracy Export", new_cells)
    nbformat.write(nb, path)


def build_bayesian_notebook():
    path = ANALYSIS / "05_bayesian_update.ipynb"
    nb = new_notebook(
        cells=[
            new_markdown_cell(source) if cell_type == "markdown" else new_code_cell(source)
            for cell_type, source in NEW_NOTEBOOK_CELLS
        ]
    )
    nbformat.write(nb, path)


def main():
    patch_evaluating_notebook()
    patch_compare_notebook()
    build_bayesian_notebook()
    print("Updated:")
    print(f"  - {ANALYSIS / '04_construct_calibrated_likelihood_layer.ipynb'}")
    print(f"  - {ANALYSIS / '03.2_compare_regime_feature_as_input.ipynb'}")
    print(f"  - {ANALYSIS / '05_bayesian_update.ipynb'}")


if __name__ == "__main__":
    main()
