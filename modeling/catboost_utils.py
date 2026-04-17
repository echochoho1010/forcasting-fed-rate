"""Reusable CatBoost helpers for notebook experiments.

This module keeps model-family boilerplate out of the analysis notebooks so the
notebooks can stay focused on experiment setup, diagnostics, and interpretation.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 100,
    "depth": 4,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3.0,
}


try:
    from catboost import CatBoostClassifier as _CatBoostClassifier

    CATBOOST_AVAILABLE = True
    CatBoostClassifier = _CatBoostClassifier
except ImportError:  # pragma: no cover - depends on local environment
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None


def catboost_search_space(trial: Any) -> dict[str, Any]:
    """Return the shared Optuna search space for CatBoost."""

    return {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


def build_catboost_classifier(
    params: Mapping[str, Any] | None = None,
    *,
    random_seed: int = 42,
    verbose: int = 0,
    thread_count: int = 1,
) -> Any:
    """Instantiate a CatBoost classifier with consistent runtime defaults."""

    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost is not installed.")

    merged = dict(params or {})
    merged.setdefault("random_seed", random_seed)
    merged.setdefault("verbose", verbose)
    merged.setdefault("thread_count", thread_count)
    # Keep CatBoost runtime artifacts out of the repo unless explicitly requested.
    merged.setdefault("allow_writing_files", False)
    return CatBoostClassifier(**merged)


def fit_catboost_classifier(
    X_train: Any,
    y_train: Any,
    augment_missing_classes: Callable[[Any, Any], tuple[Any, Any, Any]],
    params: Mapping[str, Any] | None = None,
    *,
    random_seed: int = 42,
    verbose: int = 0,
    thread_count: int = 1,
) -> Any:
    """Fit CatBoost after applying the notebook's class-augmentation step."""

    X_aug, y_aug, sample_weight = augment_missing_classes(X_train, y_train)
    model = build_catboost_classifier(
        params,
        random_seed=random_seed,
        verbose=verbose,
        thread_count=thread_count,
    )
    model.fit(X_aug, y_aug, sample_weight=sample_weight)
    return model


def predict_catboost_label(model: Any, X_test: Any) -> int:
    """Return the predicted encoded class label for a single test row."""

    return int(model.predict(X_test)[0])


def predict_catboost_proba(model: Any, X_test: Any) -> Any:
    """Return the `[P(Lower), P(Same), P(Higher)]` matrix for test rows."""

    return model.predict_proba(X_test)
