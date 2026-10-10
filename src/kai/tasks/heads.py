"""Per-representation cloud heads with an equal hyper-parameter search budget (PRD §7 fairness).

A head maps a decoded payload to a class. It is trained to reproduce the reference model's
decision (distillation, ADR 0004), because agreement with `f_raw` is what the certificate measures.
Every representation gets the same Optuna budget over the same two families: a standardised
logistic regression and LightGBM. Lab plane only; the head classes live in `kai.cloud.heads`.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from kai.cloud.heads import N_CLASSES, Head, LinearHead, TreeHead

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _linear(x: np.ndarray, y: np.ndarray, c: float, seed: int) -> LinearHead:
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-9)
    weight, bias = np.zeros((N_CLASSES, x.shape[1])), np.full(N_CLASSES, -1e9)
    classes = np.unique(y)
    if len(classes) == 1:  # the reference model gives one answer on all training windows
        bias[classes[0]] = 0.0
        return LinearHead(mean, scale, weight, bias)
    lr = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", random_state=seed)
    lr.fit((x - mean) / scale, y)
    if len(classes) == 2:  # binary fit: one row scoring the second class against the first
        weight[classes[1]], bias[classes[1]], bias[classes[0]] = lr.coef_[0], lr.intercept_[0], 0.0
    else:
        weight[lr.classes_], bias[lr.classes_] = lr.coef_, lr.intercept_
    return LinearHead(mean, scale, weight, bias)


def _tree(x: np.ndarray, y: np.ndarray, params: dict, seed: int, n_jobs: int) -> TreeHead:
    model = lgb.LGBMClassifier(
        **params,
        class_weight="balanced",
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=n_jobs,
        verbose=-1,
    ).fit(x, y)
    return TreeHead(model.booster_, model.classes_.astype(np.int64))


def tune_and_fit(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int,
    seed: int,
    n_jobs: int = 8,
    search_rows: int = 6000,
) -> Head:
    """Optuna search over both head families, scored by macro-F1 against `y` on bearing-grouped
    inner CV, then a refit on all rows.

    The search runs on an evenly strided subsample of at most `search_rows` windows (every bearing
    stays represented); trials, folds and rows are identical for every representation.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(np.unique(y)) == 1:
        return _linear(x, y, 1.0, seed)
    stride = max(1, len(x) // search_rows)
    xs, ys, gs = x[::stride], y[::stride], groups[::stride]
    folds = list(GroupKFold(n_splits=min(3, len(np.unique(gs)))).split(xs, ys, gs))

    def build(trial: optuna.trial.BaseTrial, xt: np.ndarray, yt: np.ndarray) -> Head:
        if trial.suggest_categorical("family", ["linear", "tree"]) == "linear":
            return _linear(xt, yt, trial.suggest_float("C", 1e-3, 10.0, log=True), seed)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 60, 240),
            "num_leaves": trial.suggest_int("num_leaves", 6, 48),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return _tree(xt, yt, params, seed, n_jobs)

    def objective(trial: optuna.Trial) -> float:
        scores = [
            f1_score(ys[va], build(trial, xs[tr], ys[tr]).predict(xs[va]), average="macro")
            for tr, va in folds
            if len(np.unique(ys[tr])) > 1
        ]
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return build(study.best_trial, x, y)
