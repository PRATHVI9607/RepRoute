"""Baselines B4–B11 (PRD §9.1). B9–B11 are conceptual re-implementations of prior art, adapted to
the representation-selection setting; none of them is the original authors' code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from kai.certify.ltt import binom_cdf
from kai.select.base import Decision, Mission, Truth, Window, ladder, send_anyway
from kai.select.cost import Bool, CostModel, OverheadKind, State


def marginally_admissible(d_calib: Bool, alpha: float) -> Bool:
    """Representations whose *marginal* disagreement rate passes the CRC-style bound."""
    n = len(d_calib)
    out: Bool = (d_calib.sum(axis=0) + 1) / (n + 1) <= alpha
    out[0] = True
    return out


def ltt_admissible(d_calib: Bool, alpha: float, delta: float) -> Bool:
    """Representations certified at level α by a Bonferroni-corrected binomial test (for B11)."""
    n, k = d_calib.shape
    out: Bool = np.array(
        [binom_cdf(int(d_calib[:, r].sum()), n, alpha) <= delta / (k - 1) for r in range(k)]
    )
    out[0] = True
    return out


def best_single(admissible: Bool, plan: CostModel, nominal: State) -> int:
    """B4: the cheapest admissible representation under the scenario's nominal state."""
    feasible = admissible & plan.feasible(nominal)
    pool = feasible if feasible.any() else admissible
    return int(np.where(pool, plan.cost(nominal), np.inf).argmin())


@dataclass
class ContextBlind:
    """B5: uses the per-window safe set but ignores bandwidth, queue and load (smallest wins)."""

    plan: CostModel
    name: str = "b5_context_blind"
    overhead: ClassVar[OverheadKind] = "full"  # it needs the per-window safe set

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        rep = int(np.where(win.safe, self.plan.profiles.size_bytes, np.inf).argmin())
        return send_anyway(rep, win.safe)


@dataclass
class ProfileTable:
    """B8: the obvious engineer's lookup table — regime bin → representation, no predictor.

    Built offline for decade bandwidth bins and the given load levels, assuming an empty queue.
    """

    plan: CostModel
    admissible: Bool
    deadline_ms: float
    loads: tuple[float, ...] = (0.0, 0.5, 0.8)
    name: str = "b8_profile_table"
    overhead: ClassVar[OverheadKind] = "select"
    _edges: np.ndarray = field(init=False)
    _table: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._edges = np.logspace(4, 8, 9)  # 10 kbps … 100 Mbps, two bins per decade
        centres = np.sqrt(self._edges[:-1] * self._edges[1:])
        self._table = np.array(
            [
                [
                    best_single(
                        self.admissible, self.plan, State(float(b), 0.0, 0.0, rho, self.deadline_ms)
                    )
                    for rho in self.loads
                ]
                for b in centres
            ]
        )

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        i = int(
            np.clip(np.searchsorted(self._edges, state.bandwidth_bps) - 1, 0, len(self._table) - 1)
        )
        j = int(np.abs(np.asarray(self.loads) - state.load).argmin())
        return send_anyway(int(self._table[i, j]), win.safe)


@dataclass
class AdaptiveThreshold:
    """B9, after Zhang et al. 2023: drift-driven adaptive data reduction inside one codec family.

    Sends the most compressed reconstruction-oriented representation while the signal is steady and
    steps up to higher fidelity when the probe's RMS moves away from its running level. The
    threshold adapts to the running spread. There is no notion of task sufficiency.
    """

    ladder_reps: tuple[int, ...]  # most compressed → least compressed
    beta: float = 0.05
    k_sigma: float = 2.0
    name: str = "b9_adaptive_threshold"
    overhead: ClassVar[OverheadKind] = "probe"  # reads the probe's RMS, runs no predictor
    _mean: float = field(default=0.0, init=False)
    _var: float = field(default=1.0, init=False)
    _started: bool = field(default=False, init=False)

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        level = float(np.log(win.phi[0] + 1e-12))  # log RMS
        if not self._started:
            self._mean, self._var, self._started = level, 0.25, True
        z = abs(level - self._mean) / np.sqrt(self._var)
        self._mean += self.beta * (level - self._mean)
        self._var += self.beta * ((level - self._mean) ** 2 - self._var)
        step = min(int(z / self.k_sigma), len(self.ladder_reps) - 1)
        return send_anyway(self.ladder_reps[step], win.safe)


@dataclass
class TwoModeSwitch:
    """B10, after Qu et al. 2026: switch between a semantic and a traditional mode on bandwidth.

    The switching point is the latency break-even bandwidth of the two modes (PRD §6.6).
    """

    plan: CostModel
    semantic: int
    traditional: int
    name: str = "b10_two_mode"
    overhead: ClassVar[OverheadKind] = "select"

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        p = self.plan.profiles
        dt = (
            self.plan.edge_time_ms(state.load)[self.semantic]
            + p.t_dec_ms[self.semantic]
            + p.t_head_ms[self.semantic]
            - self.plan.edge_time_ms(state.load)[self.traditional]
            - p.t_dec_ms[self.traditional]
            - p.t_head_ms[self.traditional]
        )
        ds_bits = (p.size_bytes[self.traditional] - p.size_bytes[self.semantic]) * 8.0
        b_star = ds_bits / (dt / 1000.0) if dt > 0 else np.inf
        rep = self.semantic if state.bandwidth_bps < b_star else self.traditional
        return send_anyway(rep, win.safe)


@dataclass
class ConformalModelSelection:
    """B11, in the style of Kalør & Ohtsuki 2025: state-aware choice among configurations whose
    *marginal* risk is certified offline. No per-window sufficiency predictor."""

    plan: CostModel
    admissible: Bool
    mission: Mission = "fidelity_first"
    name: str = "b11_conformal_model_selection"
    overhead: ClassVar[OverheadKind] = "select"

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        d = ladder(self.admissible, self.plan.feasible(state), self.plan.cost(state), self.mission)
        # "certified" is reserved for Kai's per-window certificate; report membership honestly
        return d if d.rep < 0 else send_anyway(d.rep, win.safe)
