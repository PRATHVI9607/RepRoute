"""Search for a stronger per-window sufficiency predictor ĝ_r(φ) ≈ P(d_r = 1 | φ).

Model selection on the `ptrain` partition only: 5-fold CV grouped by unit (`bearing_id`).
`calib` and `test` rows are never read past the partition mask. d_r is decision DISagreement with
f_raw, not label accuracy. Every candidate, winner or loser, is appended as one JSON line to
experiments/model_search/predictor-search.jsonl.

    uv run --no-sync python scripts/predictor_search.py [--datasets pu har] [--only mlp32_current]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # parallel over folds, not BLAS threads
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

from kai.certify.fit import fit_predictor
from kai.contracts.windows import WindowSet
from kai.reps import features as ft
from kai.tasks.pipeline import Table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "model_search" / "predictor-search.jsonl"
DATASETS = {  # name -> (KAI_WORK, dataset config, L1 fold dir)
    "pu": (ROOT, "pu", ROOT / "artifacts/folds/820e37cd09ba/s0_r0f0"),
    "speechcmd": (ROOT / "transfer/speechcmd", "speechcmd", None),
    "har": (ROOT / "transfer/har", "har", None),
    "sleepedf": (ROOT / "transfer/sleepedf", "sleepedf", None),
}
N_SPLITS, N_JOBS, EPS = 5, 5, 1e-12


# ---- extra probe features (numpy only, no matmul: edge-safe) ---------------------------------


def extra_phi(x: np.ndarray) -> np.ndarray:
    """22 extra features of one int16 window, on top of the 20 of `kai.certify.probe.probe`."""
    xf = ft.to_float(x)
    power = ft.power_spectrum(xf)
    p = power[1:] + EPS
    total = float(p.sum())
    flatness = float(np.exp(np.log(p).mean())) / (total / len(p))
    rolloff = float(np.searchsorted(np.cumsum(p), 0.85 * total)) / len(p)
    bands = ft.log_band_energies(power, ft.log_band_edges(len(power) - 1, 16))
    bands = bands - np.log10(total)  # relative band energies: level-free spectral shape
    env = ft.envelope(xf - xf.mean())
    em = float(env.sum()) / len(env)
    ec = env - em
    ev = float((ec * ec).sum()) / len(env)
    env_cv = np.sqrt(ev) / (em + EPS)
    env_kurt = float(np.square(ec * ec).sum()) / len(env) / (ev * ev + EPS)
    seg = np.square(xf[: len(xf) // 8 * 8]).reshape(8, -1).sum(1)  # energy in 8 sub-windows
    seg_cv = float(seg.std()) / (float(seg.mean()) + EPS)
    dx = np.diff(xf)
    rms = np.sqrt(float(np.square(xf).sum()) / len(xf)) + EPS
    diff_ratio = np.sqrt(float(np.square(dx).sum()) / len(dx)) / rms
    return np.concatenate([[flatness, rolloff, env_cv, env_kurt, seg_cv, diff_ratio], bands])


def _extra_chunk(x: np.ndarray) -> np.ndarray:
    return np.stack([extra_phi(w) for w in x])


def load(name: str) -> dict:
    work, ds, fold = DATASETS[name]
    fold = fold or next(iter(sorted((work / "artifacts/folds").glob("*/s0_r0f0"))))
    t = Table.load(fold / "table.npz")
    m = t.of("ptrain")  # the only rows this script ever uses
    out = {
        "phi": t.phi[m].astype(np.float64),
        "d": t.d[m],
        "groups": np.asarray(t.bearing_id[m]),
        "rep_ids": t.rep_ids,
        "bytes": np.median(t.payload_bytes[m], 0),
        "window_id": t.window_id[m],
    }
    cache_dir = Path(os.environ.get("PS_CACHE", tempfile.gettempdir()))
    cache = cache_dir / f"predictor_search_{name}_extra.npy"
    if cache.exists():
        out["extra"] = np.load(cache)
    else:
        cfg = OmegaConf.load(ROOT / "configs/dataset" / f"{ds}.yaml")
        ws = WindowSet.load(work / cfg.windows)
        order = np.argsort(ws.window_id)
        idx = order[np.searchsorted(ws.window_id, out["window_id"], sorter=order)]
        assert np.array_equal(ws.window_id[idx], out["window_id"]), "window ids do not match"
        assert np.array_equal(ws.bearing_id[idx], out["groups"]), "unit ids do not match"
        xs = ws.x[idx]
        del ws
        parts = Parallel(N_JOBS)(delayed(_extra_chunk)(c) for c in np.array_split(xs, 64))
        out["extra"] = np.concatenate(parts)
        np.save(cache, out["extra"])
    assert np.isfinite(out["extra"]).all()
    return out


# ---- models: fit(x_tr, Y_tr) -> predict(x) giving [n, cols] probabilities --------------------


def _constant(y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    rate = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    return lambda x: np.full(len(x), rate)


def per_column(make: Callable[[], object], transform: str = "z") -> Callable:
    def fit(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        tf = fit_transform(x, transform)
        z = tf(x)
        preds = []
        for k in range(y.shape[1]):
            if y[:, k].all() or not y[:, k].any():
                preds.append(_constant(y[:, k]))
            else:
                mdl = make().fit(z, y[:, k])  # type: ignore[attr-defined]
                preds.append(lambda q, mdl=mdl: mdl.predict_proba(q)[:, 1])
        return lambda q: np.stack([p(tf(q)) for p in preds], 1)

    return fit


def shared(hidden: tuple[int, ...], alpha: float, transform: str = "z") -> Callable:
    """One multi-label net: shared hidden layers, one sigmoid output per column."""

    def fit(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        tf = fit_transform(x, transform)
        live = np.flatnonzero(y.any(0) & ~y.all(0))
        mdl = MLPClassifier(hidden, alpha=alpha, max_iter=800, random_state=0)
        mdl.fit(tf(x), y[:, live].astype(int))
        consts = [_constant(y[:, k]) for k in range(y.shape[1])]

        def predict(q: np.ndarray) -> np.ndarray:
            out = np.stack([c(q) for c in consts], 1)
            out[:, live] = mdl.predict_proba(tf(q))
            return out

        return predict

    return fit


def current(kind: str) -> Callable:
    """`kai.certify.fit.fit_predictor` itself; columns are passed as fake reps after a raw one."""

    def fit(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        cols = y.shape[1]
        d = np.concatenate([np.zeros((len(y), 1), bool), y], 1)
        pred = fit_predictor(x, d, tuple(f"c{i}" for i in range(cols + 1)), kind)  # type: ignore[arg-type]
        return lambda q: pred.predict(q)[:, 1:]

    return fit


def fit_transform(x: np.ndarray, kind: str) -> Callable[[np.ndarray], np.ndarray]:
    if kind == "rankgauss":
        qt = QuantileTransformer(n_quantiles=min(1000, len(x)), output_distribution="normal")
        qt.fit(x)
        return qt.transform
    if kind == "log":  # log of strictly positive columns (heavy-tailed time stats)
        pos = x.min(0) > 0
        lo = np.where(pos, x.min(0), 0.0)

        def f(q: np.ndarray) -> np.ndarray:
            return np.where(pos, np.log(np.maximum(q, lo)), q)

        g = f
    else:
        g = lambda q: q  # noqa: E731
    gx = g(x)
    mean, scale = gx.mean(0), np.maximum(gx.std(0), 1e-9)
    return lambda q: (g(q) - mean) / scale


def mlp(hidden: tuple[int, ...], alpha: float = 1e-2) -> Callable[[], object]:
    return lambda: MLPClassifier(hidden, alpha=alpha, max_iter=800, random_state=0)


CANDIDATES: dict[str, tuple[Callable, dict, str, str]] = {
    # name -> (fit, params, features, edge note); features: base | extra | cum (targets)
    "mlp32_current": (
        current("mlp"),
        {"hidden": [32], "alpha": 1e-2},
        "base",
        "current: 8 nets 20x32+32x1, ~5.4k MAC/window, numpy",
    ),
    "logistic_current": (
        current("logistic"),
        {"C": 1.0},
        "base",
        "8 dot products of 20, ~160 MAC",
    ),
    "mlp64": (per_column(mlp((64,))), {"hidden": [64], "alpha": 1e-2}, "base", "~10.8k MAC, numpy"),
    "mlp64x32": (
        per_column(mlp((64, 32))),
        {"hidden": [64, 32], "alpha": 1e-2},
        "base",
        "~27k MAC, numpy",
    ),
    "mlp32_a1e-1": (
        per_column(mlp((32,), 1e-1)),
        {"hidden": [32], "alpha": 1e-1},
        "base",
        "as current",
    ),
    "mlp32_a1": (per_column(mlp((32,), 1.0)), {"hidden": [32], "alpha": 1.0}, "base", "as current"),
    "mlp32_log": (
        per_column(mlp((32,)), "log"),
        {"hidden": [32], "alpha": 1e-2, "transform": "log"},
        "base",
        "as current + 9 logs",
    ),
    "mlp32_rankgauss": (
        per_column(mlp((32,)), "rankgauss"),
        {"hidden": [32], "alpha": 1e-2, "transform": "quantile-normal, 1000 q"},
        "base",
        "as current + 20 interpolations in 1000-entry tables",
    ),
    "hgb": (
        per_column(
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                min_samples_leaf=40,
                random_state=0,
            )
        ),
        {"lr": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "l2": 1.0, "min_leaf": 40},
        "base",
        "needs numpy tree evaluator: 8x200 trees of 15 leaves, ~6.4k compares + branchy loops",
    ),
    "shared64": (
        shared((64,), 1e-2),
        {"hidden": [64], "alpha": 1e-2, "shared": True},
        "base",
        "one net 20x64+64x9, ~1.9k MAC (cheaper than current)",
    ),
    "shared128x64": (
        shared((128, 64), 1e-2),
        {"hidden": [128, 64], "alpha": 1e-2, "shared": True},
        "base",
        "~11k MAC, numpy",
    ),
    "mlp32_extra": (
        current("mlp"),
        {"hidden": [32], "alpha": 1e-2, "phi": "base20+extra22"},
        "extra",
        "as current on 42 features + extra_phi: 1 complex FFT/IFFT pair for envelope, ~10 sums",
    ),
    "logistic_extra": (
        current("logistic"),
        {"C": 1.0, "phi": "base20+extra22"},
        "extra",
        "42-dim dot products + extra_phi cost",
    ),
    "hgb_extra": (
        per_column(
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                min_samples_leaf=40,
                random_state=0,
            )
        ),
        {"lr": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "phi": "base20+extra22"},
        "extra",
        "tree evaluator + extra_phi cost",
    ),
    "shared64_extra": (
        shared((64,), 1e-2),
        {"hidden": [64], "alpha": 1e-2, "shared": True, "phi": "+extra22"},
        "extra",
        "one net 42x64+64x9 + extra_phi cost",
    ),
    "mlp32_cum_cheaper": (
        current("mlp"),
        {"hidden": [32], "alpha": 1e-2, "target": "max_{bytes_s<=bytes_r} d_s"},
        "cum",
        "as current (only the training targets change)",
    ),
}


# ---- grouped CV ------------------------------------------------------------------------------


def _fold(fit: Callable, x: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> tuple:
    warnings.simplefilter("ignore", ConvergenceWarning)
    return te, fit(x[tr], y[tr])(x[te])


def evaluate(data: dict, name: str) -> dict:
    fit, params, feats, note = CANDIDATES[name]
    d = data["d"][:, 1:].astype(bool)  # non-raw reps
    hard = d.any(1)
    x = data["phi"]
    if feats == "extra":
        x = np.concatenate([data["phi"], data["extra"]], 1)
    y_train = d.copy()
    if feats == "cum":  # smoothed targets: a rep fails where any rep no larger than it fails
        b = data["bytes"][1:]
        y_train = np.stack([d[:, b <= b[k]].any(1) for k in range(d.shape[1])], 1)
    y_fit = np.concatenate([y_train, hard[:, None]], 1)  # last column: the hardness target
    splits = GroupKFold(N_SPLITS).split(x, hard, data["groups"])
    t0 = time.perf_counter()
    parts = Parallel(N_JOBS)(delayed(_fold)(fit, x, y_fit, tr, te) for tr, te in splits)
    fit_s = time.perf_counter() - t0
    oof = np.full(y_fit.shape, np.nan)
    for te, p in parts:
        oof[te] = p
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    aucs, lls = {}, {}
    for k, rep in enumerate(data["rep_ids"][1:]):
        if 0 < d[:, k].mean() < 1:  # evaluated against the true d_r, never the smoothed target
            aucs[rep] = float(roc_auc_score(d[:, k], oof[:, k]))
            lls[rep] = float(log_loss(d[:, k], oof[:, k], labels=[0, 1]))
    return {
        "candidate": name,
        "params": params,
        "auc_mean": float(np.mean(list(aucs.values()))),
        "logloss_mean": float(np.mean(list(lls.values()))),
        "hardness_auc": float(roc_auc_score(hard, oof[:, -1])) if 0 < hard.mean() < 1 else None,
        "auc_per_rep": aucs,
        "fit_seconds": round(fit_s, 1),
        "edge_inference_note": note,
    }


def _self_check() -> None:
    rng = np.random.default_rng(0)
    w = (rng.standard_normal(4096) * 3000).astype(np.int16)
    f = extra_phi(w)
    assert f.shape == (22,)
    assert np.isfinite(f).all()
    assert 0.3 < f[0] <= 1.0, "white noise should have a flat spectrum"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--only", nargs="+", default=list(CANDIDATES))
    args = ap.parse_args()
    _self_check()
    for ds in args.datasets:
        data = load(ds)
        n_units = len(np.unique(data["groups"]))
        print(f"== {ds}: ptrain {len(data['phi'])} windows, {n_units} units", flush=True)
        for name in args.only:
            row = {"dataset": ds, "n_ptrain": len(data["phi"]), "n_units": n_units}
            row |= evaluate(data, name)
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"  {name:20s} auc {row['auc_mean']:.3f}  ll {row['logloss_mean']:.3f}  "
                f"hard {row['hardness_auc']:.3f}  {row['fit_seconds']:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
