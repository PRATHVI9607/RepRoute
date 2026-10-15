"""Core selectors: S0 static, S1 greedy-safe, S2 MPC-safe and the oracle (PRD §6.5)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import ClassVar

import numpy as np

from kai.contracts.frame import OVERHEAD_BYTES
from kai.select.base import Decision, Mission, Truth, Window, ladder, send_anyway
from kai.select.cost import Bool, CostModel, OverheadKind, State


@dataclass
class Static:
    """S0 / B1–B4: one fixed representation, whatever the window or the link looks like."""

    rep: int
    name: str = "s0_static"
    overhead: ClassVar[OverheadKind] = "none"  # no probe, no predictor, nothing to decide

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        return send_anyway(self.rep, win.safe)


@dataclass
class GreedySafe:
    """S1: argmin cost over S_λ̂(x) ∩ Feas(z), with the degradation ladder. The core runtime.

    `plan` is the cost model the planner believes in; ablations pass a deliberately wrong one
    (no encode cost, no queue awareness) while the simulator keeps charging the true costs.
    """

    plan: CostModel
    mission: Mission = "fidelity_first"
    queue_aware: bool = True
    name: str = "s1_greedy_safe"
    overhead: ClassVar[OverheadKind] = "full"

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        s = state if self.queue_aware else replace(state, backlog_bytes=0.0)
        cost, feasible = self.plan.evaluate(s)
        return ladder(win.safe, feasible, cost, self.mission)


@dataclass
class CascadeSafe:
    """S1 with a value-of-information gate on the per-window work (ADR 0011).

    Every window first sees the *free* part of its safe set: the representations certified for
    every window, which needs no probe. The probe and predictor run only when some other
    representation, even after paying for them, could be cheaper than the best free one that is
    feasible. Both paths choose inside the certified set, so the any-policy bound holds.
    """

    plan: CostModel
    mission: Mission = "fidelity_first"
    name: str = "s1_cascade"
    overhead: ClassVar[OverheadKind] = "full"  # what a probed window costs; free ones pay "select"

    def __post_init__(self) -> None:
        self._free_plan = self.plan.for_policy("select")
        self._full_plan = self.plan.for_policy("full")

    def select_without_scores(self, free: Bool, state: State) -> Decision | None:
        """A decision from the free set, or None when probing could pay for itself."""
        c_free, f_free = self._free_plan.evaluate(state)
        c_full, f_full = self._full_plan.evaluate(state)
        best_free = np.where(free & f_free, c_free, np.inf)
        best_other = np.where(~free & f_full, c_full, np.inf).min()
        if np.isfinite(best_free).any() and best_free.min() <= best_other:
            return Decision(int(best_free.argmin()), "certified")
        if not np.isfinite(best_other):  # nothing the probe could add fits: no need to run it
            return ladder(free, f_free, c_free, self.mission)
        return None

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        cost, feasible = self._full_plan.evaluate(state)
        return ladder(win.safe, feasible, cost, self.mission)


@dataclass
class GreedyStatic(GreedySafe):
    """S1 on a safe set that is the same for every window (constant scores).

    The choice still follows the link state, but nothing is computed from the window: the edge
    runs no probe and no predictor, only the selector.
    """

    name: str = "s1_greedy_static"
    overhead: ClassVar[OverheadKind] = "select"


@dataclass
class Oracle:
    """S*: knows d_r and the true link state. An upper bound, never deployable."""

    plan: CostModel
    mission: Mission = "fidelity_first"
    name: str = "oracle"
    overhead: ClassVar[OverheadKind] = "select"  # it is told d_r: no probe or predictor to pay for

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        if truth is None:
            raise ValueError("the oracle needs the ground truth")
        cost, feasible = self.plan.evaluate(truth.state)
        return ladder(~truth.d, feasible, cost, self.mission)


class MpcSafe:
    """S2: receding-horizon control over the next `horizon` windows (ABR-style MPC).

    Assumes the current safe set and bandwidth persist, and that one window arrives every
    `period_ms`. Enumerates every sequence over the safe set, propagates the uplink backlog
    `Q ← max(0, Q + S + 28 - B·T)` and applies the first action of the cheapest sequence.
    """

    name = "s2_mpc_safe"
    overhead: ClassVar[OverheadKind] = "full"

    def __init__(
        self,
        plan: CostModel,
        period_ms: float,
        horizon: int = 3,
        mission: Mission = "fidelity_first",
    ) -> None:
        self.plan, self.period_ms, self.horizon, self.mission = plan, period_ms, horizon, mission
        self._wire = plan.profiles.size_bytes + OVERHEAD_BYTES
        self._seqs: dict[tuple[int, ...], np.ndarray] = {}

    def _sequences(self, members: tuple[int, ...]) -> np.ndarray:
        if members not in self._seqs:
            self._seqs[members] = np.array(list(product(members, repeat=self.horizon)))
        return self._seqs[members]

    def select(self, win: Window, state: State, truth: Truth | None = None) -> Decision:
        cost, feasible = self.plan.evaluate(state)
        now = ladder(win.safe, feasible, cost, self.mission)
        members = tuple(np.flatnonzero(win.safe & feasible).tolist())
        if now.flag != "certified" or len(members) == 1:
            return now  # nothing to plan over: the ladder already decided
        seqs = self._sequences(members)
        drained = state.bandwidth_bps * self.period_ms / 8000.0  # bytes the link sends per period
        edge = self.plan.edge_time_ms(state.load)
        fixed = state.rtt_ms / 2.0 + self.plan.profiles.t_dec_ms + self.plan.profiles.t_head_ms
        pr = self.plan.prices
        base = pr.p_byte * self._wire + pr.p_j * self.plan.energy_j(state) + pr.p_cpu_ms * edge
        backlog = np.full(len(seqs), state.backlog_bytes)
        total = np.zeros(len(seqs))
        for step in range(self.horizon):
            r = seqs[:, step]
            latency = edge[r] + (backlog + self._wire[r]) * 8000.0 / state.bandwidth_bps + fixed[r]
            total += base[r] + pr.p_late_ms * np.maximum(0.0, latency - state.deadline_ms)
            backlog = np.maximum(0.0, backlog + self._wire[r] - drained)
        return Decision(int(seqs[total.argmin(), 0]), "certified")
