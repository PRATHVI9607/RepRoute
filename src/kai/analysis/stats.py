"""Statistics plan (PRD §9.6): BCa bootstrap over folds, block bootstrap, Holm–Bonferroni."""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def bca_ci(
    values: np.ndarray, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float]:
    """BCa bootstrap CI of the mean over independent units (folds, repeats)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.ptp(values) <= 1e-12 * max(1.0, abs(values.mean())):
        return float(values.mean()), float(values.mean())  # BCa is undefined without spread
    res = stats.bootstrap((values,), np.mean, confidence_level=level, method="BCa", rng=rng)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_bca(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean of the paired difference a - b with its BCa CI. Pairs are folds."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (float(diff.mean()), *bca_ci(diff, rng, level))


def block_bootstrap_ci(
    losses: np.ndarray,
    blocks: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> tuple[float, float]:
    """CI of a mean loss when windows inside a block (bearing) are dependent: resample blocks."""
    ids = np.unique(blocks)
    sums = np.array([losses[blocks == b].sum() for b in ids], dtype=np.float64)
    counts = np.array([(blocks == b).sum() for b in ids], dtype=np.float64)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def one_sided_p(values: np.ndarray, threshold: float, greater: bool = True) -> float:
    """One-sample t-test p-value for H1: mean > threshold (or < threshold)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.ptp(values) == 0:
        better = values.mean() > threshold if greater else values.mean() < threshold
        return 0.0 if better else 1.0
    alt = "greater" if greater else "less"
    return float(stats.ttest_1samp(values, threshold, alternative=alt).pvalue)


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    """Holm–Bonferroni across hypotheses: name → (adjusted p, rejected at α)."""
    names = list(pvalues)
    reject, adjusted, _, _ = multipletests([pvalues[n] for n in names], alpha=alpha, method="holm")
    return {n: (float(p), bool(r)) for n, p, r in zip(names, adjusted, reject, strict=True)}
