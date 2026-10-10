"""Closed-form regime theory (PRD §6.6): break-even bandwidths and the Representation Phase Diagram.

With an empty queue the latency of representation r is a straight line in u = 1/B:
`L_r(u) = a_r(ρ) + 8·(S_r + 28)·u`, where `a_r` collects edge, RTT and cloud time. The fastest
representation as bandwidth varies is the lower envelope of these lines, and two representations
i (compact, slower to encode) and j (larger, faster) swap at

    B*_ij(ρ) = 8·(S_j - S_i) / Δt_ij(ρ),   Δt_ij = a_i(ρ) - a_j(ρ) > 0.

Below B* the compact representation wins on latency; above it the cheaper-to-encode one does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kai.contracts.frame import OVERHEAD_BYTES
from kai.select.cost import CostModel, State


def intercepts_ms(model: CostModel, load: float, rtt_ms: float = 0.0) -> np.ndarray:
    """a_r(ρ): everything in the latency that does not depend on bandwidth."""
    p = model.profiles
    out: np.ndarray = model.edge_time_ms(load) + rtt_ms / 2.0 + p.t_dec_ms + p.t_head_ms
    return out


def break_even_bps(model: CostModel, i: int, j: int, load: float) -> float:
    """B*_ij(ρ) for compact i versus larger j. `inf` if i is never slower (it then always wins)."""
    size = model.profiles.size_bytes
    if size[j] <= size[i]:
        raise ValueError("j must be the larger representation")
    a = intercepts_ms(model, load)
    dt_ms = a[i] - a[j]
    return float(8000.0 * (size[j] - size[i]) / dt_ms) if dt_ms > 0 else float("inf")


@dataclass(frozen=True)
class Boundary:
    low: int  # representation that wins below `bps`
    high: int  # representation that wins above
    bps: float


def latency_envelope(
    model: CostModel, load: float, allowed: np.ndarray | None = None
) -> list[Boundary]:
    """Predicted phase boundaries along bandwidth: the lower envelope of the latency lines."""
    slope = 8000.0 * (model.profiles.size_bytes + OVERHEAD_BYTES)  # ms per (1/bps)
    a = intercepts_ms(model, load)
    members = np.flatnonzero(np.ones(len(a), bool) if allowed is None else allowed)
    # sweep u = 1/B from 0 (infinite bandwidth) upward: start on the smallest intercept
    current = int(members[np.lexsort((slope[members], a[members]))[0]])
    u, out = 0.0, []
    while True:
        best_u, nxt = np.inf, -1
        for r in members:
            if slope[r] < slope[current] and a[r] > a[current]:
                cross = (a[r] - a[current]) / (slope[current] - slope[r])
                if u < cross < best_u or (cross == best_u and slope[r] < slope[nxt]):
                    best_u, nxt = cross, int(r)
        if nxt < 0:
            break
        out.append(Boundary(low=nxt, high=current, bps=1.0 / best_u))
        u, current = best_u, nxt
    return out[::-1]  # ascending bandwidth


def argmin_map(
    model: CostModel,
    bandwidths_bps: np.ndarray,
    loads: np.ndarray,
    deadline_ms: float = np.inf,
    allowed: np.ndarray | None = None,
    by: str = "latency",
    rtt_ms: float = 0.0,
) -> np.ndarray:
    """Phase diagram: `[len(loads), len(bandwidths)]` index of the best feasible representation.

    `by` is "latency" or "cost"; cells where nothing meets the deadline hold -1.
    """
    out = np.full((len(loads), len(bandwidths_bps)), -1)
    for i, rho in enumerate(loads):
        for j, b in enumerate(bandwidths_bps):
            s = State(float(b), rtt_ms, 0.0, float(rho), deadline_ms)
            ok = model.latency_ms(s) <= deadline_ms
            if allowed is not None:
                ok &= allowed
            if ok.any():
                score = model.latency_ms(s) if by == "latency" else model.cost(s)
                out[i, j] = int(np.where(ok, score, np.inf).argmin())
    return out


def crossings(bandwidths_bps: np.ndarray, latency_ms: np.ndarray) -> list[Boundary]:
    """Empirical boundaries from measured latency curves `[n_bandwidths, K]` (nan = not measured).

    A boundary is placed where the argmin changes, by log-linear interpolation of the two curves.
    """
    seen = ~np.isnan(latency_ms).all(axis=1)  # bandwidths where nothing was measured are skipped
    bandwidths_bps, latency_ms = bandwidths_bps[seen], latency_ms[seen]
    best = np.nanargmin(latency_ms, axis=1) if len(latency_ms) else np.empty(0, int)
    out = []
    for g in range(len(bandwidths_bps) - 1):
        lo, hi = int(best[g]), int(best[g + 1])
        if lo == hi:
            continue
        gap0 = latency_ms[g, lo] - latency_ms[g, hi]  # ≤ 0: lo wins at g
        gap1 = latency_ms[g + 1, lo] - latency_ms[g + 1, hi]  # ≥ 0: hi wins at g+1
        frac = -gap0 / (gap1 - gap0) if gap1 != gap0 else 0.5
        logb = np.log(bandwidths_bps[g]) + frac * np.log(bandwidths_bps[g + 1] / bandwidths_bps[g])
        out.append(Boundary(low=lo, high=hi, bps=float(np.exp(logb))))
    return out
