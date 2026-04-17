"""Build and load CatBoost artifacts for analysis notebooks.

The analysis notebooks should focus on interpretation and diagnostics. This
module moves CatBoost training/tuning execution into reusable artifact builders,
then exposes lightweight loaders that reconstruct the notebook-facing data
structures from saved JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize

from modeling.catboost_utils import (
    CATBOOST_AVAILABLE,
    DEFAULT_CATBOOST_PARAMS,
    build_catboost_classifier,
    fit_catboost_classifier,
    predict_catboost_label,
    predict_catboost_proba,
)

OPTUNA_AVAILABLE = False


LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}
LABEL_VALS = [-1, 0, 1]
LABEL_STRS = ["Lower", "Same", "Higher"]
ENCODED_NAMES = {0: "Lower", 1: "Same", 2: "Higher"}
CLASS_ORDER = ["Lower", "Same", "Higher"]

INITIAL_TRAIN_SIZE = 40
TUNE_START = 55
W_MEET = 6
THETA = 0.15
M = 1
FOMC_DAYS = 42
ALPHA_REG = 1.0
REGIME_ENC = {"Tightening": 1, "Plateau": 0, "Easing": -1}

# Saved CatBoost parameters from the notebook workflow.
CATBOOST_NOTEBOOK_BEST_PARAMS: dict[str, Any] = {
    "iterations": 331,
    "depth": 2,
    "learning_rate": 0.018522,
    "l2_leaf_reg": 3.272855,
    "bagging_temperature": 0.324958,
}


def artifact_root(project_root: str | Path) -> Path:
    return Path(project_root) / "artifacts" / "catboost"


def _ensure_root(project_root: str | Path) -> Path:
    root = artifact_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _arrayify_result(result: dict[str, Any]) -> dict[str, Any]:
    arr_keys = {
        "actuals_enc": int,
        "preds_enc": int,
        "actuals_orig": int,
        "preds_orig": int,
        "probas": float,
    }
    out = dict(result)
    for key, dtype in arr_keys.items():
        if key in out:
            out[key] = np.asarray(out[key], dtype=dtype)
    return out


def _study_proxy(tuning: dict[str, Any]) -> SimpleNamespace:
    trials = [SimpleNamespace(value=v) for v in tuning.get("trial_values", [])]
    return SimpleNamespace(
        trials=trials,
        best_value=float(tuning["best_value"]),
        best_params=dict(tuning["best_params"]),
    )


def augment_missing_classes(X_train: np.ndarray, y_train: np.ndarray, all_classes=(0, 1, 2)):
    present = set(y_train)
    missing = set(all_classes) - present

    class_counts = np.bincount(y_train, minlength=3)
    n_total = len(y_train)
    real_weights = np.array(
        [n_total / (3 * max(class_counts[c], 1)) for c in y_train],
        dtype=float,
    )

    if not missing:
        return X_train, y_train, real_weights

    X_mean = X_train.mean(axis=0, keepdims=True)
    X_synth = np.vstack([X_mean] * len(missing))
    y_synth = np.array(sorted(missing))
    synth_weight = float(real_weights.min()) * 0.1
    w_synth = np.full(len(missing), synth_weight, dtype=float)

    X_aug = np.vstack([X_train, X_synth])
    y_aug = np.concatenate([y_train, y_synth])
    w_aug = np.concatenate([real_weights, w_synth])
    return X_aug, y_aug, w_aug


def walk_forward_eval(
    fit_fn,
    predict_fn,
    X: np.ndarray,
    y_enc: np.ndarray,
    *,
    n_start: int = INITIAL_TRAIN_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    actuals, preds = [], []
    for t in range(n_start, len(X)):
        model = fit_fn(X[:t], y_enc[:t])
        preds.append(predict_fn(model, X[t : t + 1]))
        actuals.append(y_enc[t])
    return np.asarray(actuals, dtype=int), np.asarray(preds, dtype=int)


def walk_forward_proba(
    fit_fn,
    proba_fn,
    X: np.ndarray,
    y_enc: np.ndarray,
    *,
    n_start: int = INITIAL_TRAIN_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    actuals, probas = [], []
    for t in range(n_start, len(X)):
        model = fit_fn(X[:t], y_enc[:t])
        probas.append(proba_fn(model, X[t : t + 1])[0])
        actuals.append(y_enc[t])
    return np.asarray(actuals, dtype=int), np.vstack(probas)


def fast_wf_f1(
    fit_fn,
    predict_fn,
    X: np.ndarray,
    y_enc: np.ndarray,
    *,
    n_start: int = TUNE_START,
) -> float:
    actuals, preds = [], []
    for t in range(n_start, len(X)):
        model = fit_fn(X[:t], y_enc[:t])
        preds.append(predict_fn(model, X[t : t + 1]))
        actuals.append(y_enc[t])
    act = np.array([INV_LABEL_MAP[a] for a in actuals])
    pred = np.array([INV_LABEL_MAP[p] for p in preds])
    return float(f1_score(act, pred, labels=LABEL_VALS, average="macro", zero_division=0))


def report_metrics(name: str, actuals_enc: np.ndarray, preds_enc: np.ndarray) -> dict[str, Any]:
    act = np.array([INV_LABEL_MAP[a] for a in actuals_enc])
    pred = np.array([INV_LABEL_MAP[p] for p in preds_enc])

    acc = float(accuracy_score(act, pred))
    f1_mac = float(f1_score(act, pred, labels=LABEL_VALS, average="macro", zero_division=0))
    f1_wt = float(f1_score(act, pred, labels=LABEL_VALS, average="weighted", zero_division=0))
    return {
        "name": name,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "actuals_enc": actuals_enc,
        "preds_enc": preds_enc,
        "actuals_orig": act,
        "preds_orig": pred,
    }


def report_prob_metrics(name: str, actuals_enc: np.ndarray, probas: np.ndarray) -> dict[str, Any]:
    preds_enc = np.argmax(probas, axis=1)
    actuals_orig = np.array([INV_LABEL_MAP[a] for a in actuals_enc])
    preds_orig = np.array([INV_LABEL_MAP[p] for p in preds_enc])

    acc = float(accuracy_score(actuals_orig, preds_orig))
    f1_mac = float(
        f1_score(actuals_orig, preds_orig, labels=LABEL_VALS, average="macro", zero_division=0)
    )
    f1_wt = float(
        f1_score(actuals_orig, preds_orig, labels=LABEL_VALS, average="weighted", zero_division=0)
    )
    y_onehot = label_binarize(actuals_enc, classes=[0, 1, 2])
    logloss = float(log_loss(actuals_enc, probas, labels=[0, 1, 2]))
    brier = float(
        np.mean([brier_score_loss(y_onehot[:, k], probas[:, k]) for k in range(3)])
    )

    return {
        "name": name,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "log_loss": logloss,
        "brier_score": brier,
        "actuals_enc": actuals_enc,
        "preds_enc": preds_enc,
        "actuals_orig": actuals_orig,
        "preds_orig": preds_orig,
        "probas": probas,
    }


def _load_df_model(project_root: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(project_root) / "data" / "df_model.csv", parse_dates=["meeting_date"])


def _load_fedrate(project_root: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        Path(project_root) / "data" / "fedrate_all.csv",
        parse_dates=["observation_date"],
    )


def label_policy_regime(
    df: pd.DataFrame,
    rate_col: str = "fed_rate",
    date_col: str = "observation_date",
    W: int = W_MEET,
    theta: float = THETA,
    k: int = 1,
    epsilon_bps: float = 0.25,
    m: int = M,
    tol: float = 1e-9,
    mode: str = "realtime",
    alpha: float = ALPHA_REG,
) -> pd.DataFrame:
    assert mode in ("realtime", "analysis")
    df_out = df.copy()
    if date_col in df_out.columns:
        df_out = df_out.sort_values(date_col).reset_index(drop=True)

    rate = df_out[rate_col].copy()
    if rate.isna().sum() > 0:
        rate = rate.ffill()
    n = len(rate)

    delta_r = rate.diff()
    decision = pd.Series("Hold", index=rate.index, dtype=object)
    decision[delta_r > tol] = "Hike"
    decision[delta_r < -tol] = "Cut"
    decision[delta_r.isna()] = None

    is_hike = (decision == "Hike").fillna(False).astype(float)
    is_cut = (decision == "Cut").fillna(False).astype(float)
    H_t = is_hike.rolling(W, min_periods=1).sum()
    C_t = is_cut.rolling(W, min_periods=1).sum()
    D_t = (H_t - C_t) / W

    n_events = pd.Series(range(1, n + 1), index=rate.index, dtype=float).clip(upper=W)
    N_t = (n_events - H_t - C_t).clip(lower=0)
    denom = n_events + 3.0 * alpha
    p_tightening = (H_t + alpha) / denom
    p_easing = (C_t + alpha) / denom
    p_plateau = (N_t + alpha) / denom

    last_dir = [None] * n
    current_dir = None
    for i in range(n):
        d = decision.iloc[i]
        if d == "Hike":
            current_dir = "Hike"
        elif d == "Cut":
            current_dir = "Cut"
        last_dir[i] = current_dir

    if mode == "analysis":
        rate_vals = rate.values.astype(float)
        is_local_max = np.zeros(n, dtype=bool)
        is_local_min = np.zeros(n, dtype=bool)
        for i in range(n):
            lo, hi = max(0, i - k), min(n - 1, i + k)
            window = rate_vals[lo : hi + 1]
            if rate_vals[i] == window.max():
                is_local_max[i] = True
            if rate_vals[i] == window.min():
                is_local_min[i] = True
        most_recent_lmax = np.full(n, np.nan)
        most_recent_lmin = np.full(n, np.nan)
        last_lmax_val = np.nan
        last_lmin_val = np.nan
        for i in range(n):
            if is_local_max[i]:
                last_lmax_val = rate_vals[i]
            if is_local_min[i]:
                last_lmin_val = rate_vals[i]
            most_recent_lmax[i] = last_lmax_val
            most_recent_lmin[i] = last_lmin_val

    raw_regime = []
    for i in range(n):
        dt = D_t.iloc[i]
        if dt >= theta:
            raw_regime.append("Tightening")
        elif dt <= -theta:
            raw_regime.append("Easing")
        else:
            if mode == "realtime":
                ld = last_dir[i]
                if ld == "Hike":
                    raw_regime.append("Hold-High")
                elif ld == "Cut":
                    raw_regime.append("Hold-Low")
                else:
                    raw_regime.append("Neutral Hold / Transition")
            else:
                rv = rate_vals[i]
                lmax = most_recent_lmax[i]
                lmin = most_recent_lmin[i]
                near_max = is_local_max[i] or (not np.isnan(lmax) and rv >= lmax - epsilon_bps)
                near_min = is_local_min[i] or (not np.isnan(lmin) and rv <= lmin + epsilon_bps)
                if near_max:
                    raw_regime.append("Hold-High")
                elif near_min:
                    raw_regime.append("Hold-Low")
                else:
                    raw_regime.append("Neutral Hold / Transition")

    regime_simple = []
    prev = None
    run = 0
    for lab in raw_regime:
        if lab == prev:
            run += 1
        else:
            prev = lab
            run = 1
        if lab in ("Tightening", "Easing"):
            regime_simple.append(lab if run >= m else "Plateau")
        else:
            regime_simple.append("Plateau")

    df_out["decision"] = decision
    df_out["H_t"] = H_t
    df_out["C_t"] = C_t
    df_out["D_t"] = D_t
    df_out["p_tightening"] = p_tightening
    df_out["p_easing"] = p_easing
    df_out["p_plateau"] = p_plateau
    df_out["regime_raw"] = raw_regime
    df_out["regime_simple"] = regime_simple
    return df_out


def label_policy_regime_daily(
    df_daily: pd.DataFrame,
    rate_col: str = "fed_rate",
    date_col: str = "observation_date",
    W: int = W_MEET,
    theta: float = THETA,
    m: int = M,
    fomc_cycle_days: int = FOMC_DAYS,
    alpha: float = ALPHA_REG,
    mode: str = "realtime",
) -> pd.DataFrame:
    df_daily = df_daily.copy().sort_values(date_col).reset_index(drop=True)
    keep = [date_col, rate_col]
    df_meet = df_daily[keep].drop_duplicates().copy()
    df_lab = label_policy_regime(
        df_meet,
        rate_col=rate_col,
        date_col=date_col,
        W=W,
        theta=theta,
        m=m,
        alpha=alpha,
        mode=mode,
    )
    reg_cols = [c for c in df_lab.columns if c not in keep]
    df_daily = df_daily.merge(df_lab[[date_col] + reg_cols], on=date_col, how="left")
    regime_s = df_daily["regime_simple"].ffill()
    df_daily["regime_simple"] = regime_s
    df_daily["regime_change"] = regime_s != regime_s.shift(1)
    df_daily.loc[0, "regime_change"] = False
    return df_daily


def _study_payload(
    *,
    best_value: float,
    best_params: dict[str, Any],
    trial_values: list[float] | None = None,
    source: str = "saved_notebook_params",
) -> dict[str, Any]:
    return {
        "best_value": float(best_value),
        "best_params": dict(best_params),
        "trial_values": [float(v) for v in (trial_values or [best_value])],
        "source": source,
    }


def build_03_boosting_artifact(project_root: str | Path) -> dict[str, Any]:
    if not CATBOOST_AVAILABLE:
        raise RuntimeError("CatBoost must be installed to build artifacts.")

    df_model = _load_df_model(project_root)
    non_feature_cols = ["meeting_date", "decision", "decision_num", "prev_decision"]
    feature_cols = [c for c in df_model.columns if c not in non_feature_cols]
    X = df_model[feature_cols].values.astype(float)
    y_orig = df_model["decision_num"].values.astype(int)
    y_enc = np.array([LABEL_MAP[v] for v in y_orig])

    default_result = report_metrics(
        "CatBoost",
        *walk_forward_eval(
            lambda Xtr, ytr: fit_catboost_classifier(
                Xtr, ytr, augment_missing_classes, DEFAULT_CATBOOST_PARAMS
            ),
            predict_catboost_label,
            X,
            y_enc,
        ),
    )

    X_train_full = X[:-1]
    y_train_full = y_enc[:-1]
    X_last = X[-1:]
    cat_f = build_catboost_classifier(
        dict(DEFAULT_CATBOOST_PARAMS, auto_class_weights="Balanced"),
        thread_count=1,
    )
    cat_f.fit(X_train_full, y_train_full)
    next_meeting_proba = predict_catboost_proba(cat_f, X_last)[0]

    best_params = dict(CATBOOST_NOTEBOOK_BEST_PARAMS)

    tuned_result = report_metrics(
        "CatBoost (tuned)",
        *walk_forward_eval(
            lambda Xtr, ytr: fit_catboost_classifier(
                Xtr, ytr, augment_missing_classes, best_params
            ),
            predict_catboost_label,
            X,
            y_enc,
        ),
    )

    payload = {
        "default_result": default_result,
        "next_meeting_proba": next_meeting_proba,
        "tuning": _study_payload(
            best_value=tuned_result["f1_macro"],
            best_params=best_params,
            trial_values=[default_result["f1_macro"], tuned_result["f1_macro"]],
        ),
        "tuned_result": tuned_result,
    }
    out = _ensure_root(project_root) / "03_boosting.json"
    _write_json(out, payload)
    return payload


def _build_regime_feature_sets(project_root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_fed = _load_fedrate(project_root).sort_values("observation_date").reset_index(drop=True)
    df_model = _load_df_model(project_root)

    df_rt = label_policy_regime_daily(
        df_fed,
        rate_col="fed_rate",
        date_col="observation_date",
        W=W_MEET,
        theta=THETA,
        m=M,
        fomc_cycle_days=FOMC_DAYS,
        alpha=ALPHA_REG,
        mode="realtime",
    )
    lookup = (
        df_rt[
            [
                "observation_date",
                "regime_simple",
                "p_tightening",
                "p_easing",
                "p_plateau",
            ]
        ]
        .copy()
        .sort_values("observation_date")
        .reset_index(drop=True)
    )

    df_regime = df_model.copy().sort_values("meeting_date").reset_index(drop=True)
    meeting_dates = pd.to_datetime(df_regime["meeting_date"])
    lookup_dates = pd.to_datetime(lookup["observation_date"])

    regime_vals, p_tight_vals, p_ease_vals, p_plat_vals = [], [], [], []
    for mdate in meeting_dates:
        cutoff = mdate - pd.Timedelta(days=1)
        mask = lookup_dates <= cutoff
        if mask.any():
            row = lookup.loc[mask.index[mask][-1]]
        else:
            row = lookup.iloc[0]
        regime_vals.append(row["regime_simple"])
        p_tight_vals.append(row["p_tightening"])
        p_ease_vals.append(row["p_easing"])
        p_plat_vals.append(row["p_plateau"])

    df_regime["regime_simple"] = regime_vals
    df_regime["regime_enc"] = [REGIME_ENC[r] for r in regime_vals]
    df_regime["p_tightening"] = p_tight_vals
    df_regime["p_easing"] = p_ease_vals
    df_regime["p_plateau"] = p_plat_vals

    non_feat_cols = [
        "meeting_date",
        "decision",
        "decision_num",
        "prev_decision",
        "regime_simple",
        "p_tightening",
        "p_easing",
        "p_plateau",
    ]
    feature_cols_base = [c for c in df_model.columns if c not in non_feat_cols]
    feature_cols_regime = feature_cols_base + ["regime_enc"]
    y_orig = df_regime["decision_num"].values.astype(int)
    y_enc = np.array([LABEL_MAP[v] for v in y_orig])
    X_base = df_regime[feature_cols_base].values.astype(float)
    X_regime = df_regime[feature_cols_regime].values.astype(float)
    return X_base, X_regime, y_enc


def build_03_2_artifact(project_root: str | Path) -> dict[str, Any]:
    if not CATBOOST_AVAILABLE:
        raise RuntimeError("CatBoost must be installed to build artifacts.")

    X_base, X_regime, y_enc = _build_regime_feature_sets(project_root)
    base_params = dict(CATBOOST_NOTEBOOK_BEST_PARAMS)
    regime_params = dict(CATBOOST_NOTEBOOK_BEST_PARAMS)

    base_result = report_metrics(
        "CatBoost (base)",
        *walk_forward_eval(
            lambda Xtr, ytr: fit_catboost_classifier(
                Xtr, ytr, augment_missing_classes, base_params
            ),
            predict_catboost_label,
            X_base,
            y_enc,
        ),
    )

    regime_result = report_metrics(
        "CatBoost (+ regime)",
        *walk_forward_eval(
            lambda Xtr, ytr: fit_catboost_classifier(
                Xtr, ytr, augment_missing_classes, regime_params
            ),
            predict_catboost_label,
            X_regime,
            y_enc,
        ),
    )

    payload = {
        "best_params_base": base_params,
        "best_params_regime": regime_params,
        "tuning_base": _study_payload(
            best_value=base_result["f1_macro"],
            best_params=base_params,
        ),
        "tuning_regime": _study_payload(
            best_value=regime_result["f1_macro"],
            best_params=regime_params,
        ),
        "result_base": base_result,
        "result_regime": regime_result,
    }
    out = _ensure_root(project_root) / "03_2_compare_regime.json"
    _write_json(out, payload)
    return payload


def build_04_artifact(project_root: str | Path) -> dict[str, Any]:
    if not CATBOOST_AVAILABLE:
        raise RuntimeError("CatBoost must be installed to build artifacts.")
    best_params = dict(CATBOOST_NOTEBOOK_BEST_PARAMS)

    df_model = _load_df_model(project_root)
    non_feature_cols = ["meeting_date", "decision", "decision_num", "prev_decision"]
    feature_cols = [c for c in df_model.columns if c not in non_feature_cols]
    X = df_model[feature_cols].values.astype(float)
    y_orig = df_model["decision_num"].values.astype(int)
    y_enc = np.array([LABEL_MAP[v] for v in y_orig])

    result = report_prob_metrics(
        "CatBoost",
        *walk_forward_proba(
            lambda Xtr, ytr: fit_catboost_classifier(
                Xtr, ytr, augment_missing_classes, best_params
            ),
            predict_catboost_proba,
            X,
            y_enc,
        ),
    )
    payload = {
        "best_params": best_params,
        "final_result": result,
    }
    out = _ensure_root(project_root) / "04_calibrated_likelihood.json"
    _write_json(out, payload)
    return payload


def load_03_boosting_artifact(project_root: str | Path) -> dict[str, Any]:
    payload = _read_json(artifact_root(project_root) / "03_boosting.json")
    payload["default_result"] = _arrayify_result(payload["default_result"])
    payload["tuned_result"] = _arrayify_result(payload["tuned_result"])
    payload["next_meeting_proba"] = np.asarray(payload["next_meeting_proba"], dtype=float)
    return payload


def load_03_2_artifact(project_root: str | Path) -> dict[str, Any]:
    payload = _read_json(artifact_root(project_root) / "03_2_compare_regime.json")
    payload["result_base"] = _arrayify_result(payload["result_base"])
    payload["result_regime"] = _arrayify_result(payload["result_regime"])
    return payload


def load_04_artifact(project_root: str | Path) -> dict[str, Any]:
    payload = _read_json(artifact_root(project_root) / "04_calibrated_likelihood.json")
    payload["final_result"] = _arrayify_result(payload["final_result"])
    return payload


def catboost_artifact_available(project_root: str | Path, name: str) -> bool:
    return (artifact_root(project_root) / name).exists()


def load_03_study_proxy(project_root: str | Path) -> SimpleNamespace:
    return _study_proxy(load_03_boosting_artifact(project_root)["tuning"])
