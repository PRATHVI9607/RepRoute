"""Fit the sufficiency predictor (lab plane: scikit-learn). Never imported by the edge."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from kai.certify.predictor import F64, Layers, Predictor

HIDDEN = 32


def _constant(n_features: int, positives: int, n: int, hidden: int | None) -> Layers:
    """A representation that never (or always) disagrees on ptrain: Laplace-smoothed constant.

    Shaped like the fitted nets (`hidden` units, or none for logistic) so that the run-time
    predictor can evaluate all representations in one stacked pass.
    """
    p = (positives + 0.5) / (n + 1.0)
    bias = np.array([np.log(p / (1.0 - p))])
    if hidden is None:
        return ((np.zeros((n_features, 1)), bias),)
    return ((np.zeros((n_features, hidden)), np.zeros(hidden)), (np.zeros((hidden, 1)), bias))


def fit_predictor(
    phi: npt.NDArray[np.floating],
    d: npt.NDArray[np.bool_],
    rep_ids: tuple[str, ...],
    kind: Literal["mlp", "logistic"] = "mlp",
    seed: int = 0,
) -> Predictor:
    """Train ĝ on the `ptrain` partition only (must be disjoint from calibration and test)."""
    x = phi.astype(np.float64)
    mean: F64 = x.mean(axis=0)
    scale: F64 = np.maximum(x.std(axis=0), 1e-9)
    z = (x - mean) / scale
    nets: list[Layers] = []
    for k in range(1, len(rep_ids)):
        y = d[:, k]
        if y.all() or not y.any():
            hidden = None if kind == "logistic" else HIDDEN
            nets.append(_constant(x.shape[1], int(y.sum()), len(y), hidden))
        elif kind == "logistic":
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(z, y)
            nets.append(((lr.coef_.T.copy(), lr.intercept_.copy()),))
        else:
            mlp = MLPClassifier((HIDDEN,), alpha=1e-2, max_iter=800, random_state=seed).fit(z, y)
            nets.append(
                tuple(
                    (w.copy(), b.copy()) for w, b in zip(mlp.coefs_, mlp.intercepts_, strict=True)
                )
            )
    return Predictor(tuple(rep_ids), mean, scale, tuple(nets))
