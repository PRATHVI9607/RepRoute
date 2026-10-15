"""Safe sets and Conformal Risk Control calibration (PRD §6.1–§6.2).

Conventions
-----------
* Column 0 of every `[n, K]` array is the raw representation r0. It is always in the safe set and
  never disagrees with itself.
* A threshold λ selects `S_λ(x) = {r0} ∪ {r : ĝ_r(x) ≤ λ}`. `RAW_ONLY` (-1.0) is the threshold whose
  safe set is {r0} for every window, because ĝ ≥ 0.
* Every calibration below reduces to one number per calibration window: `m_i`, the smallest λ at
  which window i incurs a loss. The loss is then `1[λ ≥ m_i]`: monotone in λ, zero at `RAW_ONLY`
  and bounded by 1, which is what the Conformal Risk Control argument needs once λ is taken from
  the finite grid of floats (docs/theory/proposition1.md, step 3).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

RAW_ONLY = -1.0
ALL = 1.0


def safe_set(ghat: F64, lam: float | F64) -> Bool:
    """S_λ as a boolean mask with the same shape as `ghat`. `lam` may be per-representation."""
    mask: Bool = ghat <= lam
    mask[..., 0] = True
    return mask


def first_loss_threshold(ghat: F64, d: Bool) -> F64:
    """V0: m_i = min{ĝ_r(x_i) : d_r(x_i) = 1}, +inf if no representation disagrees.

    The safe-set loss is `L(x_i; λ) = max_{r ∈ S_λ} d_r = 1[λ ≥ m_i]`.
    """
    masked = np.where(d, ghat, np.inf)
    masked[:, 0] = np.inf
    out: F64 = masked.min(axis=1)
    return out


def safe_set_loss(ghat: F64, d: Bool, lam: float | F64) -> Bool:
    """L(x; λ) = 1 iff some member of the safe set disagrees with the raw decision."""
    out: Bool = (safe_set(ghat, lam) & d).any(axis=1)
    return out


def policy_loss_threshold(ghat: F64, d: Bool, cost: F64) -> F64:
    """V2: m_i for the greedy min-cost policy under the given per-window costs.

    Representations enter the safe set in order of increasing ĝ; the policy always holds the
    cheapest one admitted so far (raw is admitted from the start; `cost = inf` marks infeasible).
    m_i is the first ĝ at which that choice disagrees. Because the loss is then held at 1 for all
    larger λ, it is the monotone envelope of the true policy loss, hence conservative.
    """
    g = ghat.copy()
    g[:, 0] = -np.inf
    order = np.argsort(g, axis=1, kind="stable")
    g_s = np.take_along_axis(g, order, axis=1)
    c_s = np.take_along_axis(cost, order, axis=1)
    d_s = np.take_along_axis(d, order, axis=1)
    n, k = g.shape
    best = c_s[:, 0].copy()
    best_idx = order[:, 0].copy()
    chosen_disagrees = d_s[:, 0].copy()
    m = np.full(n, np.inf)
    for j in range(1, k):
        # ties in cost go to the lower menu index, exactly as the selector's argmin breaks them
        better = (c_s[:, j] < best) | ((c_s[:, j] == best) & (order[:, j] < best_idx))
        best = np.where(better, c_s[:, j], best)
        best_idx = np.where(better, order[:, j], best_idx)
        chosen_disagrees = np.where(better, d_s[:, j], chosen_disagrees)
        m = np.where(chosen_disagrees & np.isinf(m), g_s[:, j], m)
    return m


def threshold_for_count(m_sorted: F64, k: int) -> float:
    """Largest λ with at most `k` calibration losses, i.e. `#{i : m_i ≤ λ} ≤ k`."""
    n = len(m_sorted)
    if k < 0:
        return RAW_ONLY
    if k >= n or np.isinf(m_sorted[k]):
        return ALL
    # just below the (k+1)-th smallest m: ties at that value are all excluded
    return float(min(np.nextafter(m_sorted[k], -np.inf), ALL))


def crc_threshold(m: F64, alpha: float) -> float:
    """λ̂ = the largest float λ with (n·R̂_n(λ) + 1)/(n + 1) ≤ α (Angelopoulos et al., ICLR 2024).

    Guarantees E[L(x_{n+1}; λ̂)] ≤ α for exchangeable data. With fewer than ⌈1/α⌉ - 1 calibration
    windows even zero observed losses fail the bound and the result is `RAW_ONLY`.
    """
    n = len(m)
    # the 1e-9 absorbs float error in α(n + 1), e.g. 0.05 * 20; it can exceed the bound by at
    # most 1e-9 / (n + 1)
    allowed = math.floor(alpha * (n + 1) - 1 + 1e-9)
    return threshold_for_count(np.sort(m), allowed)
