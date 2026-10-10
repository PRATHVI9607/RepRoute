"""Learn-then-Test calibration (Angelopoulos et al., 2021): high-probability risk control (V1).

For a 0/1 loss the number of calibration losses at a fixed λ is Binomial(n, R(λ)), so
`p(λ) = P(Bin(n, α) ≤ observed)` is a valid p-value for H_λ: R(λ) > α. Risk is monotone in λ, so
fixed-sequence testing from the most conservative λ upward controls the family-wise error at δ:
P(R(λ̂) ≤ α) ≥ 1 - δ. The binomial model assumes i.i.d. windows (see PRD §6.4 on autocorrelation).
"""

from __future__ import annotations

import math

import numpy as np

from kai.certify.crc import F64, RAW_ONLY, Bool, threshold_for_count


def binom_cdf(k: int, n: int, p: float) -> float:
    """Exact P(Bin(n, p) ≤ k), summed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
        for i in range(k + 1)
    ]
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(v - top) for v in logs))


def max_certified_losses(n: int, alpha: float, delta: float) -> int:
    """Largest c with P(Bin(n, α) ≤ c) ≤ δ; -1 if even zero losses cannot be certified."""
    c = -1
    while c + 1 < n and binom_cdf(c + 1, n, alpha) <= delta:
        c += 1
    return c


def ltt_threshold(m: F64, alpha: float, delta: float) -> float:
    """Single-threshold LTT: the PAC counterpart of `crc_threshold`."""
    if len(m) == 0:
        return RAW_ONLY
    return threshold_for_count(np.sort(m), max_certified_losses(len(m), alpha, delta))


def ltt_per_rep(ghat: F64, d: Bool, alpha: float, delta: float) -> F64:
    """Per-representation thresholds λ_r with a Bonferroni split of α and δ over the K-1 reps.

    Each λ_r controls R_r = P(ĝ_r ≤ λ_r, d_r = 1) ≤ α/(K-1) w.p. ≥ 1 - δ/(K-1); the union bound
    then gives P(safe-set loss) ≤ α w.p. ≥ 1 - δ for any policy inside the safe set.
    """
    k = ghat.shape[1]
    lam = np.full(k, RAW_ONLY)
    for r in range(1, k):
        m_r = np.where(d[:, r], ghat[:, r], np.inf)
        lam[r] = ltt_threshold(m_r, alpha / (k - 1), delta / (k - 1))
    return lam
