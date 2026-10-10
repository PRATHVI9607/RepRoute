"""Latency, energy and cost model (PRD §5.5–§5.6). numpy only — runs on the edge.

All arrays are over the menu: element k belongs to representation `rep_ids[k]`, element 0 is raw.
Units: bytes, milliseconds, joules, bits per second.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from kai.contracts.frame import OVERHEAD_BYTES
from kai.contracts.schemas import EnergyModel, Prices, RepProfile

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Profiles:
    rep_ids: tuple[str, ...]
    size_bytes: F64  # planning size: q95 for variable-size codecs, exact for fixed ones
    t_enc_ms: F64  # encoder time at edge load ρ = 0
    load_exponent: F64  # t_enc(ρ) = t_enc_ms / (1 - ρ)**load_exponent
    t_dec_ms: F64
    t_head_ms: F64

    @staticmethod
    def from_list(profiles: list[RepProfile]) -> Profiles:
        def col(name: str) -> F64:
            return np.array([getattr(p, name) for p in profiles], dtype=np.float64)

        return Profiles(
            tuple(p.rep_id for p in profiles),
            col("size_q95_bytes"),
            col("t_enc_ms"),
            col("load_exponent"),
            col("t_dec_ms"),
            col("t_head_ms"),
        )

    def without_encode_cost(self) -> Profiles:
        """Ablation H4: a planner that believes encoding is free."""
        return replace(self, t_enc_ms=np.zeros_like(self.t_enc_ms))


# What a policy runs for every window: nothing, a table lookup, the probe, or probe + predictor.
OverheadKind = Literal["none", "select", "probe", "full"]
_OVERHEAD_PARTS: dict[str, tuple[int, int, int]] = {
    "none": (0, 0, 0),
    "select": (0, 0, 1),
    "probe": (1, 0, 1),
    "full": (1, 1, 1),
}


@dataclass(frozen=True)
class Overheads:
    """Work done for every window regardless of the choice (select-then-encode, PRD §5.2)."""

    t_probe_ms: float = 0.0
    t_pred_ms: float = 0.0
    t_sel_ms: float = 0.0
    load_exponent: float = (
        1.0  # like the encoders: 1.0 is processor sharing, fitted by the profiler
    )

    @property
    def total_ms(self) -> float:
        return self.t_probe_ms + self.t_pred_ms + self.t_sel_ms

    def only(self, kind: OverheadKind) -> Overheads:
        """The part of the overhead a policy really incurs (a static policy runs no probe)."""
        probe, pred, sel = _OVERHEAD_PARTS[kind]
        return Overheads(
            self.t_probe_ms * probe, self.t_pred_ms * pred, self.t_sel_ms * sel, self.load_exponent
        )


@dataclass(frozen=True)
class State:
    """System state z at decision time. Anything here may depend on everything except d_r."""

    bandwidth_bps: float
    rtt_ms: float
    backlog_bytes: float
    load: float  # edge CPU load ρ ∈ [0, 1) from other tenants
    deadline_ms: float  # remaining budget for this window
    energy_budget_j: float = np.inf
    storage_free_bytes: float = np.inf


@dataclass(frozen=True)
class CostModel:
    profiles: Profiles
    overheads: Overheads
    energy: EnergyModel
    prices: Prices

    def for_policy(self, kind: OverheadKind) -> CostModel:
        """This model with only the per-window overhead that a policy of `kind` pays."""
        return replace(self, overheads=self.overheads.only(kind))

    def edge_time_ms(self, load: float) -> F64:
        """Edge CPU time. Probe, predictor and selector share the CPU, so they slow down too."""
        p = self.profiles
        ov = self.overheads
        out: F64 = (
            ov.total_ms / (1.0 - load) ** ov.load_exponent
            + p.t_enc_ms / (1.0 - load) ** p.load_exponent
        )
        return out

    def latency_ms(self, s: State) -> F64:
        """L_r = t_edge,r(ρ) + (Q + S_r + 28)/B + RTT/2 + t_dec,r + t_head,r."""
        p = self.profiles
        tx_ms = (s.backlog_bytes + p.size_bytes + OVERHEAD_BYTES) * 8000.0 / s.bandwidth_bps
        out: F64 = self.edge_time_ms(s.load) + tx_ms + s.rtt_ms / 2.0 + p.t_dec_ms + p.t_head_ms
        return out

    def energy_j(self, s: State) -> F64:
        """Modelled edge energy: active CPU power × CPU time + radio energy per transmitted bit."""
        bits = (self.profiles.size_bytes + OVERHEAD_BYTES) * 8.0
        out: F64 = (
            self.energy.p_cpu_w * self.edge_time_ms(s.load) / 1000.0 + self.energy.e_bit_j * bits
        )
        return out

    def cost(self, s: State) -> F64:
        """cost(r | z) = p_byte·(S+28) + p_J·E + p_cpu·t_edge + p_late·max(0, L - D)."""
        pr = self.prices
        out: F64 = (
            pr.p_byte * (self.profiles.size_bytes + OVERHEAD_BYTES)
            + pr.p_j * self.energy_j(s)
            + pr.p_cpu_ms * self.edge_time_ms(s.load)
            + pr.p_late_ms * np.maximum(0.0, self.latency_ms(s) - s.deadline_ms)
        )
        return out

    def evaluate(self, s: State) -> tuple[F64, Bool]:
        """(cost, feasible) with the shared terms computed once: the selector's hot path."""
        p, pr = self.profiles, self.prices
        wire = p.size_bytes + OVERHEAD_BYTES
        edge = self.edge_time_ms(s.load)
        latency = (
            edge
            + (s.backlog_bytes + wire) * 8000.0 / s.bandwidth_bps
            + s.rtt_ms / 2.0
            + p.t_dec_ms
            + p.t_head_ms
        )
        energy = self.energy.p_cpu_w * edge / 1000.0 + self.energy.e_bit_j * wire * 8.0
        cost: F64 = (
            pr.p_byte * wire
            + pr.p_j * energy
            + pr.p_cpu_ms * edge
            + pr.p_late_ms * np.maximum(0.0, latency - s.deadline_ms)
        )
        feasible: Bool = (
            (latency <= s.deadline_ms)
            & (energy <= s.energy_budget_j)
            & (wire <= s.storage_free_bytes)
        )
        return cost, feasible

    def feasible(self, s: State) -> Bool:
        """Feas(z) = {r : L_r ≤ D, E_r ≤ E_budget, S_r + 28 ≤ storage_free}."""
        out: Bool = (
            (self.latency_ms(s) <= s.deadline_ms)
            & (self.energy_j(s) <= s.energy_budget_j)
            & (self.profiles.size_bytes + OVERHEAD_BYTES <= s.storage_free_bytes)
        )
        return out
