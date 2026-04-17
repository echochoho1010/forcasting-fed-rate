from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None


LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}
LABEL_VALS = [-1, 0, 1]
CLASS_ORDER = ["Lower", "Same", "Higher"]
ENCODED_CLASS_ORDER = [0, 1, 2]
INITIAL_TRAIN_SIZE = 40

NON_FEATURE_COLS = {"meeting_date", "decision", "decision_num", "prev_decision"}
PAIR_FEATURES = ["unemployment_gap_lag1", "output_gap_pct_lag1"]
PC1_NAME = "cycle_gap_pc1"


@dataclass(frozen=True)
class FeatureSet:
    name: str
    description: str
    columns: list[str]
    use_pair_pc1: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    group: str
    feature_set: str | None
    description: str
    runner: Callable


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "data" / "df_model.csv").exists():
            return parent
    raise FileNotFoundError("Could not locate data/df_model.csv from this script.")


def load_df_model(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "data" / "df_model.csv", parse_dates=["meeting_date"])
    return df.sort_values("meeting_date").reset_index(drop=True)


def base_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if col in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(col)
    return columns


def compact_concept_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "target_rate_lag1",
        "PCE_inflation_gap_lag1",
        "PCE_gap_change_1m",
        "unemployment_gap_lag1",
        "unemployment_gap_change_1m",
        "output_gap_pct_lag1",
        "output_gap_change_1q",
        "NFCI_lag1",
        "nfci_change_1m",
        "prev_decision_num",
        "time_since_last_change",
        "consecutive_same",
    ]
    return [col for col in preferred if col in df.columns]


def build_feature_sets(df: pd.DataFrame) -> dict[str, FeatureSet]:
    full = base_feature_columns(df)
    compact = compact_concept_columns(df)

    missing_pair = [col for col in PAIR_FEATURES if col not in full]
    if missing_pair:
        raise KeyError(f"Missing PCA pair columns: {missing_pair}")

    def without(columns: list[str], drop: set[str]) -> list[str]:
        return [col for col in columns if col not in drop]

    return {
        "full": FeatureSet(
            name="full",
            description="All numeric model features from df_model.csv.",
            columns=full,
        ),
        "pair_pc1": FeatureSet(
            name="pair_pc1",
            description=(
                "Full features with unemployment_gap_lag1 and output_gap_pct_lag1 "
                "replaced by walk-forward cycle_gap_pc1."
            ),
            columns=without(full, set(PAIR_FEATURES)),
            use_pair_pc1=True,
        ),
        "hard_select_output_gap": FeatureSet(
            name="hard_select_output_gap",
            description="Full features with unemployment_gap_lag1 dropped.",
            columns=without(full, {"unemployment_gap_lag1"}),
        ),
        "hard_select_unemployment_gap": FeatureSet(
            name="hard_select_unemployment_gap",
            description="Full features with output_gap_pct_lag1 dropped.",
            columns=without(full, {"output_gap_pct_lag1"}),
        ),
        "concept_lag1_change": FeatureSet(
            name="concept_lag1_change",
            description=(
                "Compact concept set using one level and one change feature for "
                "each macro block."
            ),
            columns=compact,
        ),
        "concept_lag1_change_pc1": FeatureSet(
            name="concept_lag1_change_pc1",
            description=(
                "Compact concept set with unemployment_gap_lag1 and "
                "output_gap_pct_lag1 replaced by cycle_gap_pc1."
            ),
            columns=without(compact, set(PAIR_FEATURES)),
            use_pair_pc1=True,
        ),
    }


def oriented_pair_pca(train_pair: pd.DataFrame) -> tuple[StandardScaler, PCA]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_pair.astype(float).to_numpy())
    pca = PCA(n_components=1)
    pca.fit(x_train)

    output_idx = PAIR_FEATURES.index("output_gap_pct_lag1")
    if pca.components_[0, output_idx] < 0:
        pca.components_ *= -1
    return scaler, pca


def build_xy_for_fold(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    x_train = train[feature_set.columns].astype(float).copy()
    x_test = test[feature_set.columns].astype(float).copy()
    out_columns = list(feature_set.columns)

    if feature_set.use_pair_pc1:
        scaler, pca = oriented_pair_pca(train[PAIR_FEATURES])
        train_pc1 = pca.transform(scaler.transform(train[PAIR_FEATURES].astype(float).to_numpy()))
        test_pc1 = pca.transform(scaler.transform(test[PAIR_FEATURES].astype(float).to_numpy()))
        x_train[PC1_NAME] = train_pc1[:, 0]
        x_test[PC1_NAME] = test_pc1[:, 0]
        out_columns.append(PC1_NAME)

    return x_train.to_numpy(dtype=float), x_test.to_numpy(dtype=float), out_columns


def augment_missing_classes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    all_classes: tuple[int, ...] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    present = set(int(v) for v in y_train)
    missing = [cls for cls in all_classes if cls not in present]
    counts = np.bincount(y_train, minlength=max(all_classes) + 1)
    n_total = len(y_train)
    real_weights = np.array(
        [n_total / (len(all_classes) * max(counts[int(cls)], 1)) for cls in y_train],
        dtype=float,
    )

    if not missing:
        return x_train, y_train, real_weights

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_synth = np.vstack([x_mean] * len(missing))
    y_synth = np.array(missing, dtype=int)
    synth_weight = max(float(real_weights.min()) * 0.05, 1e-6)
    w_synth = np.full(len(missing), synth_weight, dtype=float)
    return (
        np.vstack([x_train, x_synth]),
        np.concatenate([y_train, y_synth]),
        np.concatenate([real_weights, w_synth]),
    )


def align_proba(classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    out = np.zeros(3, dtype=float)
    for idx, cls in enumerate(classes):
        out[int(cls)] = float(proba[idx])
    total = out.sum()
    if total <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    return out / total


def hgb_default() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=60,
        max_depth=4,
        learning_rate=0.03,
        min_samples_leaf=5,
        l2_regularization=1.0,
        max_leaf_nodes=16,
        max_bins=32,
        early_stopping=False,
        class_weight="balanced",
        random_state=42,
    )


def hgb_tight() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=45,
        max_depth=2,
        learning_rate=0.04,
        min_samples_leaf=12,
        l2_regularization=5.0,
        max_leaf_nodes=8,
        max_bins=32,
        early_stopping=False,
        class_weight="balanced",
        random_state=42,
    )


def hgb_stage() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=30,
        max_depth=2,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=5.0,
        max_leaf_nodes=6,
        max_bins=32,
        early_stopping=False,
        class_weight="balanced",
        random_state=42,
    )


def predict_majority(history: np.ndarray, alpha: float = 1.0) -> tuple[int, np.ndarray]:
    counts = np.bincount(history, minlength=3).astype(float)
    proba = (counts + alpha) / (counts.sum() + 3 * alpha)
    pred = int(np.argmax(counts))
    return pred, proba


def predict_last_decision(history: np.ndarray, epsilon: float = 0.01) -> tuple[int, np.ndarray]:
    pred = int(history[-1])
    proba = np.full(3, epsilon / 2, dtype=float)
    proba[pred] = 1.0 - epsilon
    return pred, proba


def fit_predict_logistic(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[int, np.ndarray]:
    x_aug, y_aug, weights = augment_missing_classes(x_train, y_train)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )
    model.fit(x_aug, y_aug, logisticregression__sample_weight=weights)
    proba = align_proba(model[-1].classes_, model.predict_proba(x_test)[0])
    return int(np.argmax(proba)), proba


def fit_predict_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    factory: Callable[[], HistGradientBoostingClassifier],
) -> tuple[int, np.ndarray]:
    x_aug, y_aug, weights = augment_missing_classes(x_train, y_train)
    model = factory()
    model.fit(x_aug, y_aug, sample_weight=weights)
    proba = align_proba(model.classes_, model.predict_proba(x_test)[0])
    return int(np.argmax(proba)), proba


def fit_predict_catboost(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[int, np.ndarray]:
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost is not installed.")

    x_aug, y_aug, weights = augment_missing_classes(x_train, y_train)
    model = CatBoostClassifier(
        iterations=100,
        depth=2,
        learning_rate=0.04,
        l2_leaf_reg=6.0,
        loss_function="MultiClass",
        random_seed=42,
        verbose=0,
        thread_count=1,
        allow_writing_files=False,
    )
    model.fit(x_aug, y_aug, sample_weight=weights)
    proba = align_proba(model.classes_, model.predict_proba(x_test)[0])
    return int(np.argmax(proba)), proba


def fit_binary_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    factory: Callable[[], HistGradientBoostingClassifier],
) -> tuple[float, np.ndarray | None]:
    unique = np.unique(y_train)
    if len(unique) == 1:
        return float(unique[0]), None

    x_aug, y_aug, weights = augment_missing_classes(x_train, y_train, all_classes=(0, 1))
    model = factory()
    model.fit(x_aug, y_aug, sample_weight=weights)
    return 0.0, align_binary_proba(model.classes_, model.predict_proba(x_test)[0])


def align_binary_proba(classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    out = np.zeros(2, dtype=float)
    for idx, cls in enumerate(classes):
        out[int(cls)] = float(proba[idx])
    total = out.sum()
    if total <= 0:
        return np.array([0.5, 0.5], dtype=float)
    return out / total


def fit_predict_two_stage_hgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    change_threshold: float = 0.4,
) -> tuple[int, np.ndarray]:
    y_change = (y_train != LABEL_MAP[0]).astype(int)
    fallback_change, change_proba = fit_binary_hgb(x_train, y_change, x_test, hgb_stage)

    if change_proba is None:
        p_change = fallback_change
    else:
        p_change = float(change_proba[1])

    changed_mask = y_train != LABEL_MAP[0]
    if changed_mask.sum() == 0:
        direction_proba = np.array([0.5, 0.5], dtype=float)
    else:
        x_dir = x_train[changed_mask]
        y_dir = (y_train[changed_mask] == LABEL_MAP[1]).astype(int)
        fallback_dir, dir_proba = fit_binary_hgb(x_dir, y_dir, x_test, hgb_stage)
        if dir_proba is None:
            direction_proba = (
                np.array([1.0, 0.0], dtype=float)
                if fallback_dir == 0
                else np.array([0.0, 1.0], dtype=float)
            )
        else:
            direction_proba = dir_proba

    proba = np.array(
        [
            p_change * direction_proba[0],
            1.0 - p_change,
            p_change * direction_proba[1],
        ],
        dtype=float,
    )
    proba = proba / proba.sum()

    if p_change >= change_threshold:
        pred = LABEL_MAP[-1] if direction_proba[0] >= direction_proba[1] else LABEL_MAP[1]
    else:
        pred = LABEL_MAP[0]
    return int(pred), proba


def run_feature_model(
    df: pd.DataFrame,
    y_enc: np.ndarray,
    feature_set: FeatureSet,
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actuals: list[int] = []
    preds: list[int] = []
    probas: list[np.ndarray] = []

    for t in range(INITIAL_TRAIN_SIZE, len(df)):
        train_idx = np.arange(t)
        test_idx = np.array([t])
        x_train, x_test, _ = build_xy_for_fold(df, feature_set, train_idx, test_idx)
        pred, proba = predictor(x_train, y_enc[:t], x_test)
        actuals.append(int(y_enc[t]))
        preds.append(pred)
        probas.append(proba)

    return np.asarray(actuals), np.asarray(preds), np.vstack(probas)


def run_history_baseline(
    y_enc: np.ndarray,
    predictor: Callable[[np.ndarray], tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actuals: list[int] = []
    preds: list[int] = []
    probas: list[np.ndarray] = []
    for t in range(INITIAL_TRAIN_SIZE, len(y_enc)):
        pred, proba = predictor(y_enc[:t])
        actuals.append(int(y_enc[t]))
        preds.append(pred)
        probas.append(proba)
    return np.asarray(actuals), np.asarray(preds), np.vstack(probas)


def metric_payload(name: str, actuals: np.ndarray, preds: np.ndarray, probas: np.ndarray) -> dict[str, object]:
    actuals_orig = np.array([INV_LABEL_MAP[int(v)] for v in actuals])
    preds_orig = np.array([INV_LABEL_MAP[int(v)] for v in preds])
    y_onehot = label_binarize(actuals, classes=ENCODED_CLASS_ORDER)
    report = classification_report(
        actuals_orig,
        preds_orig,
        labels=LABEL_VALS,
        target_names=CLASS_ORDER,
        zero_division=0,
        output_dict=True,
    )
    pred_counts = {
        CLASS_ORDER[encoded]: int((preds == encoded).sum())
        for encoded in ENCODED_CLASS_ORDER
    }
    actual_counts = {
        CLASS_ORDER[encoded]: int((actuals == encoded).sum())
        for encoded in ENCODED_CLASS_ORDER
    }
    return {
        "experiment": name,
        "accuracy": float(accuracy_score(actuals_orig, preds_orig)),
        "balanced_accuracy": float(balanced_accuracy_score(actuals_orig, preds_orig)),
        "f1_macro": float(f1_score(actuals_orig, preds_orig, labels=LABEL_VALS, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(actuals_orig, preds_orig, labels=LABEL_VALS, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(actuals, np.clip(probas, 1e-12, 1.0), labels=ENCODED_CLASS_ORDER)),
        "brier_multiclass": float(np.mean(np.sum((y_onehot - probas) ** 2, axis=1))),
        "lower_recall": float(report["Lower"]["recall"]),
        "same_recall": float(report["Same"]["recall"]),
        "higher_recall": float(report["Higher"]["recall"]),
        "lower_f1": float(report["Lower"]["f1-score"]),
        "same_f1": float(report["Same"]["f1-score"]),
        "higher_f1": float(report["Higher"]["f1-score"]),
        "pred_lower": pred_counts["Lower"],
        "pred_same": pred_counts["Same"],
        "pred_higher": pred_counts["Higher"],
        "actual_lower": actual_counts["Lower"],
        "actual_same": actual_counts["Same"],
        "actual_higher": actual_counts["Higher"],
        "n_predictions": int(len(actuals)),
    }


def prediction_rows(
    df: pd.DataFrame,
    experiment: ExperimentSpec,
    actuals: np.ndarray,
    preds: np.ndarray,
    probas: np.ndarray,
) -> list[dict[str, object]]:
    dates = df["meeting_date"].iloc[INITIAL_TRAIN_SIZE:].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "experiment": experiment.name,
                "group": experiment.group,
                "feature_set": experiment.feature_set or "",
                "meeting_date": date.date().isoformat(),
                "actual": CLASS_ORDER[int(actuals[i])],
                "predicted": CLASS_ORDER[int(preds[i])],
                "p_lower": float(probas[i, 0]),
                "p_same": float(probas[i, 1]),
                "p_higher": float(probas[i, 2]),
            }
        )
    return rows


def pca_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    scaler, pca = oriented_pair_pca(df[PAIR_FEATURES])
    x_scaled = scaler.transform(df[PAIR_FEATURES].astype(float).to_numpy())
    scores = pca.transform(x_scaled)[:, 0]
    loadings = pd.DataFrame(
        {
            "feature": PAIR_FEATURES,
            PC1_NAME: pca.components_[0],
        }
    )
    loadings.to_csv(output_dir / "pca_pair_loadings.csv", index=False)

    corr = float(df[PAIR_FEATURES[0]].corr(df[PAIR_FEATURES[1]]))
    meta = {
        "method": "two_feature_standardized_pca",
        "features": PAIR_FEATURES,
        "pc1_name": PC1_NAME,
        "orientation": "higher PC1 means higher output gap and lower unemployment gap",
        "pearson_corr": corr,
        "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores, ddof=0)),
        "leakage_note": (
            "The diagnostic loadings use the full sample for interpretation only. "
            "Model experiments fit PCA inside each walk-forward training fold."
        ),
    }
    (output_dir / "pca_pair_meta.json").write_text(json.dumps(meta, indent=2))


def build_experiments(feature_sets: dict[str, FeatureSet]) -> list[ExperimentSpec]:
    experiments = [
        ExperimentSpec(
            name="baseline_majority_expanding",
            group="baseline",
            feature_set=None,
            description="Expanding majority-class baseline with smoothed 3-class probabilities.",
            runner=lambda df, y: run_history_baseline(y, predict_majority),
        ),
        ExperimentSpec(
            name="baseline_last_decision",
            group="baseline",
            feature_set=None,
            description="Persistence baseline that repeats the previous FOMC decision.",
            runner=lambda df, y: run_history_baseline(y, predict_last_decision),
        ),
        ExperimentSpec(
            name="baseline_multinomial_logit_balanced",
            group="baseline",
            feature_set="full",
            description="Balanced multinomial logistic regression on the full feature set.",
            runner=lambda df, y: run_feature_model(df, y, feature_sets["full"], fit_predict_logistic),
        ),
        ExperimentSpec(
            name="hgb_default_full",
            group="tree",
            feature_set="full",
            description="Current-style HistGradientBoosting setup on the full feature set.",
            runner=lambda df, y: run_feature_model(
                df,
                y,
                feature_sets["full"],
                lambda xtr, ytr, xte: fit_predict_hgb(xtr, ytr, xte, hgb_default),
            ),
        ),
        ExperimentSpec(
            name="hgb_tight_full",
            group="tree",
            feature_set="full",
            description="Tighter HistGradientBoosting on the full feature set.",
            runner=lambda df, y: run_feature_model(
                df,
                y,
                feature_sets["full"],
                lambda xtr, ytr, xte: fit_predict_hgb(xtr, ytr, xte, hgb_tight),
            ),
        ),
        ExperimentSpec(
            name="hgb_tight_pair_pc1",
            group="tree",
            feature_set="pair_pc1",
            description="Tighter tree with the redundant gap pair replaced by PC1.",
            runner=lambda df, y: run_feature_model(
                df,
                y,
                feature_sets["pair_pc1"],
                lambda xtr, ytr, xte: fit_predict_hgb(xtr, ytr, xte, hgb_tight),
            ),
        ),
        ExperimentSpec(
            name="hgb_tight_hard_select_output_gap",
            group="tree",
            feature_set="hard_select_output_gap",
            description="Tighter tree that keeps output_gap_pct_lag1 and drops unemployment_gap_lag1.",
            runner=lambda df, y: run_feature_model(
                df,
                y,
                feature_sets["hard_select_output_gap"],
                lambda xtr, ytr, xte: fit_predict_hgb(xtr, ytr, xte, hgb_tight),
            ),
        ),
        ExperimentSpec(
            name="hgb_tight_hard_select_unemployment_gap",
            group="tree",
            feature_set="hard_select_unemployment_gap",
            description="Tighter tree that keeps unemployment_gap_lag1 and drops output_gap_pct_lag1.",
            runner=lambda df, y: run_feature_model(
                df,
                y,
                feature_sets["hard_select_unemployment_gap"],
                lambda xtr, ytr, xte: fit_predict_hgb(xtr, ytr, xte, hgb_tight),
            ),
        ),
        ExperimentSpec(
            name="hgb_tight_concept_lag1_change",
            group="tree",
            feature_set="concept_lag1_change",
            description="Tighter tree with compact lag-1 plus change features.",
            runner=lambda df, y: run_feature_model(
                df,
                y,
                feature_sets["concept_lag1_change"],
                lambda xtr, ytr, xte: fit_predict_hgb(xtr, ytr, xte, hgb_tight),
            ),
        ),
        ExperimentSpec(
            name="hgb_tight_concept_lag1_change_pc1",
            group="tree",
            feature_set="concept_lag1_change_pc1",
            description="Compact lag-1 plus change features with the redundant gap pair replaced by PC1.",
            runner=lambda df, y: run_feature_model(
                df,
                y,
                feature_sets["concept_lag1_change_pc1"],
                lambda xtr, ytr, xte: fit_predict_hgb(xtr, ytr, xte, hgb_tight),
            ),
        ),
        ExperimentSpec(
            name="two_stage_hgb_tight_pair_pc1",
            group="two_stage",
            feature_set="pair_pc1",
            description="Two-stage changed-vs-same then lower-vs-higher model using PC1 features.",
            runner=lambda df, y: run_feature_model(
                df,
                y,
                feature_sets["pair_pc1"],
                fit_predict_two_stage_hgb,
            ),
        ),
        ExperimentSpec(
            name="two_stage_hgb_tight_concept_pc1",
            group="two_stage",
            feature_set="concept_lag1_change_pc1",
            description="Two-stage model using compact concept features and PC1.",
            runner=lambda df, y: run_feature_model(
                df,
                y,
                feature_sets["concept_lag1_change_pc1"],
                fit_predict_two_stage_hgb,
            ),
        ),
    ]
    if CATBOOST_AVAILABLE:
        experiments.extend(
            [
                ExperimentSpec(
                    name="catboost_tight_full",
                    group="tree",
                    feature_set="full",
                    description="Small tight CatBoost check on the full feature set.",
                    runner=lambda df, y: run_feature_model(
                        df,
                        y,
                        feature_sets["full"],
                        fit_predict_catboost,
                    ),
                ),
                ExperimentSpec(
                    name="catboost_tight_pair_pc1",
                    group="tree",
                    feature_set="pair_pc1",
                    description="Small tight CatBoost check with the redundant gap pair replaced by PC1.",
                    runner=lambda df, y: run_feature_model(
                        df,
                        y,
                        feature_sets["pair_pc1"],
                        fit_predict_catboost,
                    ),
                ),
            ]
        )
    return experiments


def write_feature_sets(feature_sets: dict[str, FeatureSet], output_dir: Path) -> None:
    payload = {
        name: {
            "description": spec.description,
            "columns": spec.columns + ([PC1_NAME] if spec.use_pair_pc1 else []),
            "uses_walk_forward_pair_pc1": spec.use_pair_pc1,
        }
        for name, spec in feature_sets.items()
    }
    (output_dir / "feature_sets.json").write_text(json.dumps(payload, indent=2))


def write_summary(metrics: pd.DataFrame, experiments: list[ExperimentSpec], output_dir: Path) -> None:
    descriptions = {exp.name: exp.description for exp in experiments}
    ranked = metrics.sort_values(["f1_macro", "balanced_accuracy"], ascending=False).reset_index(drop=True)
    table_cols = [
        "experiment",
        "group",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "pred_lower",
        "pred_same",
        "pred_higher",
    ]
    table = ranked[table_cols].copy()
    for col in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]:
        table[col] = table[col].map(lambda value: f"{value:.4f}")
    markdown_rows = [
        "| " + " | ".join(table_cols) + " |",
        "| " + " | ".join(["---"] * len(table_cols)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        markdown_rows.append("| " + " | ".join(str(value) for value in row) + " |")
    lines = [
        "# PCA and Tree-Model Optimization Summary",
        "",
        "Primary metric: macro-F1 on the expanding walk-forward window.",
        "",
        "All models and baselines emit probabilities on the same `[Lower, Same, Higher]` scale.",
        "",
        "## Ranked Results",
        "",
        "\n".join(markdown_rows),
        "",
        "## Experiment Notes",
        "",
    ]
    for exp in experiments:
        row = metrics.loc[metrics["experiment"] == exp.name].iloc[0]
        lines.extend(
            [
                f"### {exp.name}",
                "",
                descriptions[exp.name],
                "",
                (
                    f"Macro-F1={row['f1_macro']:.4f}, accuracy={row['accuracy']:.4f}, "
                    f"predicted counts Lower/Same/Higher="
                    f"{int(row['pred_lower'])}/{int(row['pred_same'])}/{int(row['pred_higher'])}."
                ),
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for experiment outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_df_model(root)
    y_orig = df["decision_num"].astype(int).to_numpy()
    y_enc = np.array([LABEL_MAP[int(v)] for v in y_orig], dtype=int)

    feature_sets = build_feature_sets(df)
    experiments = build_experiments(feature_sets)

    pca_diagnostics(df, output_dir)
    write_feature_sets(feature_sets, output_dir)

    metric_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []

    for exp in experiments:
        print(f"Running {exp.name}...", flush=True)
        actuals, preds, probas = exp.runner(df, y_enc)
        row = metric_payload(exp.name, actuals, preds, probas)
        row["group"] = exp.group
        row["feature_set"] = exp.feature_set or ""
        row["description"] = exp.description
        metric_rows.append(row)
        pred_rows.extend(prediction_rows(df, exp, actuals, preds, probas))
        print(
            f"  f1_macro={row['f1_macro']:.4f} "
            f"accuracy={row['accuracy']:.4f} "
            f"pred_counts={row['pred_lower']}/{row['pred_same']}/{row['pred_higher']}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["f1_macro", "balanced_accuracy"], ascending=False
    )
    predictions = pd.DataFrame(pred_rows)

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    write_summary(metrics, experiments, output_dir)

    print()
    print(f"Wrote outputs to {output_dir}", flush=True)
    print(metrics[["experiment", "accuracy", "balanced_accuracy", "f1_macro", "pred_lower", "pred_same", "pred_higher"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
