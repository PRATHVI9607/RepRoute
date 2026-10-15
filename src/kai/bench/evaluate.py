"""Scenario cells, the policy zoo (Kai, baselines B1–B11, ablations) and per-run metrics."""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from kai.bench.calibrate import (
    Scored,
    calibrate,
    cascade_certificate,
    score,
    static_safe_set,
    threshold,
)
from kai.bench.config import ROOT
from kai.certify import crc
from kai.certify.aci import AdaptiveLambda, ipw_risk
from kai.certify.fit import FIT_VERSION
from kai.certify.predictor import Predictor
from kai.certify.runtime import Certifier
from kai.contracts.frame import OVERHEAD_BYTES
from kai.contracts.schemas import FLAGS, EnergyModel, LambdaTable, Partition, Prices, RepProfile
from kai.link import traces as tr
from kai.link.des import Result, Scenario, Workload, simulate
from kai.select import baselines as bl
from kai.select.base import Mission, Selector
from kai.select.cost import CostModel, Overheads, Profiles, State
from kai.select.policies import CascadeSafe, GreedySafe, GreedyStatic, MpcSafe, Oracle, Static
from kai.tasks.pipeline import Table

COMPRESSION = ("r0_raw", "r1_lossless", "r2a_wavelet", "r2b_wavelet")
MEASURED_PREFIX = "ucc4g_"  # links named ucc4g_<mobility class> replay measured logs
MEASURED_DIR = ROOT / "data/raw/traces/ucc4g"  # one set of traces for every dataset
LEARNED = ("r0_raw", "r5_ae", "r6_bottleneck", "r7_vq")


@dataclass(frozen=True)
class Cell:
    """One point of the scenario grid (PRD §9.2)."""

    link: str  # "static" or a trace name from kai.link.traces
    bandwidth_bps: float  # the static rate, or the rate the trace is rescaled to (its median)
    deadline_ms: float
    load: float
    n_sensors: int
    alpha: float
    energy_budget_j: float = np.inf
    storage_bytes: float = np.inf
    rtt_ms: float = 20.0
    p_audit: float = 0.0

    @property
    def bursty(self) -> bool:
        return self.link != "static"


def make_trace(cell: Cell, seed: int) -> tr.Trace:
    """The cell's link. Every cell starts at its own point of the trace; the α levels of one
    cell share it, so the policies and certificates they compare see the same link."""
    if cell.link == "static":
        return tr.static(cell.bandwidth_bps)
    key = repr((cell.link, cell.bandwidth_bps, cell.deadline_ms, cell.load, cell.n_sensors))
    rng = np.random.default_rng([seed, 11, zlib.crc32(key.encode())])
    if cell.link == "markov_bursty":
        trace = tr.markov_bursty(rng, cell.bandwidth_bps, cell.bandwidth_bps / 20.0)
        return tr.Trace(
            trace.name, trace.step_s, np.roll(trace.bps, -int(rng.integers(len(trace.bps))))
        )
    if cell.link.startswith(MEASURED_PREFIX):
        # measured 4G throughput logs of one mobility class; the seed picks the log
        files = sorted(MEASURED_DIR.glob(f"{cell.link.removeprefix(MEASURED_PREFIX)}__*.csv"))
        if not files:
            raise FileNotFoundError(
                f"no measured traces for {cell.link}: run `python scripts/download_traces.py`"
            )
        trace = tr.from_csv(files[int(rng.integers(len(files)))], step_s=1.0)
    else:
        trace = tr.cellular(rng, cell.link)
    # rescale to the sensor-data regime: keep the shape, move the median to `bandwidth_bps`
    bps = trace.bps * cell.bandwidth_bps / float(np.median(trace.bps))
    return tr.Trace(trace.name, trace.step_s, np.roll(bps, -int(rng.integers(len(bps)))))


@dataclass(frozen=True)
class FoldCtx:
    """Everything one fold contributes to an evaluation: scores, calibration data, cost model."""

    scored: Scored
    model: CostModel
    lambdas: dict[float, LambdaTable]  # per α; V2 is added per cell
    window_period_ms: float
    static_safe: dict[float, np.ndarray]  # per α: the window-independent safe set (V0, constant ĝ)
    cascade: dict[float, tuple[np.ndarray, float]]  # per α: (prior, λ) of the cascade (ADR 0011)

    @property
    def rep_ids(self) -> tuple[str, ...]:
        return self.scored.table.rep_ids

    def workload(self, n: int, partition: Partition = "test") -> Workload:
        t = self.scored.table
        idx = spread(np.flatnonzero(t.of(partition)), n)
        return Workload(self.scored.ghat[idx], t.phi[idx], t.d[idx], t.payload_bytes[idx])


def spread(idx: np.ndarray, n: int) -> np.ndarray:
    """`n` indices evenly spaced over `idx`, in order (repeating if `idx` is shorter).

    Windows are stored bearing by bearing, so the first `n` would cover one bearing only.
    """
    out: np.ndarray = idx[np.linspace(0, len(idx), n, endpoint=False).astype(int)]
    return out


def load_fold(
    fold_dir: Path,
    profiles: list[RepProfile],
    overheads: Overheads,
    prices: Prices,
    energy: EnergyModel,
    alphas: tuple[float, ...],
    delta: float,
    window_period_ms: float,
    predictor_kind: Literal["mlp", "logistic"] = "mlp",
    seed: int = 0,
) -> FoldCtx:
    scored = scored_fold(fold_dir, predictor_kind, seed)
    by_id = {p.rep_id: p for p in profiles}
    model = CostModel(
        Profiles.from_list([by_id[r] for r in scored.table.rep_ids]), overheads, energy, prices
    )
    ghat_cal, d_cal = scored.part("calib")
    lambdas = {a: calibrate(ghat_cal, d_cal, scored.table.rep_ids, (a,), delta) for a in alphas}
    d_ptrain = scored.table.d[scored.table.of("ptrain")]
    static = {a: static_safe_set(d_ptrain, d_cal, a) for a in alphas}
    cascade = {a: cascade_certificate(ghat_cal, d_cal, d_ptrain, a) for a in alphas}
    return FoldCtx(scored, model, lambdas, window_period_ms, static, cascade)


def scored_fold(fold_dir: Path, kind: Literal["mlp", "logistic"] = "mlp", seed: int = 0) -> Scored:
    """Score a fold, fitting the sufficiency predictor once and keeping it next to the fold.

    Fitting takes about a minute and every worker process of a grid run needs it. The fit is
    deterministic, so a cached predictor is identical to a fresh one; the file is written
    atomically because several workers may reach a new fold at the same time.
    """
    table = Table.load(fold_dir / "table.npz")
    # keyed on the table itself, the seed and the fitting code: a fold rebuilt in place must
    # never meet a predictor fitted on other bearings or by other code (ADR 0007)
    key = f"{FIT_VERSION}|{seed}|".encode() + (fold_dir / "table.npz").read_bytes()
    tag = f"{kind}_{zlib.crc32(key):08x}"
    cached = fold_dir / f"predictor_{tag}.npz"
    if cached.exists():
        predictor = Predictor.load(cached)
        return Scored(table, predictor, predictor.predict(table.phi))
    scored = score(table, kind, seed)
    tmp = fold_dir / f".predictor_{tag}.{os.getpid()}.npz"
    scored.predictor.save(tmp)
    tmp.replace(cached)
    return scored


def nominal_state(cell: Cell) -> State:
    return State(
        cell.bandwidth_bps,
        cell.rtt_ms,
        0.0,
        cell.load,
        cell.deadline_ms,
        cell.energy_budget_j,
        cell.storage_bytes,
    )


def v2_threshold(ctx: FoldCtx, cell: Cell) -> float:
    """Policy-specific CRC for greedy-safe under the cell's nominal state (V2)."""
    ghat_cal, d_cal = ctx.scored.part("calib")
    s = nominal_state(cell)
    cost = np.where(ctx.model.feasible(s), ctx.model.cost(s), np.inf)
    m = crc.policy_loss_threshold(ghat_cal, d_cal, np.broadcast_to(cost, ghat_cal.shape))
    return crc.crc_threshold(m, cell.alpha)


def policies(
    ctx: FoldCtx, cell: Cell, mission: Mission, which: str = "main"
) -> dict[str, tuple[Selector, Certifier]]:
    """name → (selector, certifier). `which`: "main" (E3) or "ablation" (E5)."""
    model, ids, alpha = ctx.model, ctx.rep_ids, cell.alpha
    table = ctx.lambdas[alpha]
    lam0 = threshold(table, "v0", alpha, ids)
    cert0 = Certifier(lam0)
    _, d_cal = ctx.scored.part("calib")
    period = ctx.window_period_ms / cell.n_sensors
    nominal = nominal_state(cell)

    def idx(rep_id: str) -> int:
        return ids.index(rep_id)

    def mask(members: tuple[str, ...]) -> np.ndarray:
        return np.array([r in members for r in ids])

    # each baseline plans with the overhead it really pays; the simulator charges the same
    lookup, static = model.for_policy("select"), model.for_policy("none")
    if which == "main":
        marginal = bl.marginally_admissible(d_cal, alpha)
        return {
            "b1_raw": (Static(0, "b1_raw"), cert0),
            "b2_compress": (Static(idx("r2a_wavelet"), "b2_compress"), cert0),
            "b3_features": (Static(idx("r3_spectral"), "b3_features"), cert0),
            "b4_best_single": (
                Static(bl.best_single(marginal, static, nominal), "b4_best_single"),
                cert0,
            ),
            "b5_context_blind": (bl.ContextBlind(model), cert0),
            "b8_profile_table": (bl.ProfileTable(lookup, marginal, cell.deadline_ms), cert0),
            "b9_adaptive_threshold": (
                bl.AdaptiveThreshold((idx("r2b_wavelet"), idx("r2a_wavelet"), idx("r1_lossless"))),
                cert0,
            ),
            "b10_two_mode": (
                bl.TwoModeSwitch(lookup, idx("r6_bottleneck"), idx("r1_lossless")),
                cert0,
            ),
            "b11_cms": (
                bl.ConformalModelSelection(lookup, bl.ltt_admissible(d_cal, alpha, 0.1), mission),
                cert0,
            ),
            "b13_adaedge": (
                bl.AdaEdgeBandit(lookup, idx("r1_lossless"), np.random.default_rng(13)),
                cert0,
            ),
            "kai_s1_v0": (GreedySafe(model, mission), cert0),
            "kai_s1_v1": (
                GreedySafe(model, mission),
                Certifier(threshold(table, "v1", alpha, ids)),
            ),
            "kai_s1_v1_per_rep": (
                GreedySafe(model, mission),
                Certifier(threshold(table, "v1_per_rep", alpha, ids)),
            ),
            # V0 on min(ĝ, prior): probe only when it can pay for itself (ADR 0011)
            "kai_s1_cascade": (
                CascadeSafe(model, mission),
                Certifier(ctx.cascade[alpha][1], prior=ctx.cascade[alpha][0]),
            ),
            # the same certificate with constant scores: one safe set, no probe, no predictor
            "kai_s1_v0_static": (
                GreedyStatic(lookup, mission),
                Certifier(np.where(ctx.static_safe[alpha], crc.ALL, crc.RAW_ONLY)),
            ),
            "kai_s1_v2": (GreedySafe(model, mission), Certifier(v2_threshold(ctx, cell))),
            "kai_s2_v0": (MpcSafe(model, period, mission=mission), cert0),
            "kai_s1_v0_live": (GreedySafe(model, "liveness_first"), cert0),
            "oracle": (Oracle(lookup, mission), cert0),
        }
    no_enc = replace(model, profiles=model.profiles.without_encode_cost())
    learned_free = tuple(r for r in ids if r not in LEARNED[1:])
    calib_m = crc.first_loss_threshold(*ctx.scored.part("calib"))
    return {
        "kai_s1_v0": (GreedySafe(model, mission), cert0),
        "no_encode_cost": (GreedySafe(no_enc, mission), cert0),
        "no_queue_awareness": (GreedySafe(model, mission, queue_aware=False), cert0),
        "no_certificate": (GreedySafe(model, mission), Certifier(alpha)),  # raw ĝ ≤ α, uncalibrated
        "no_predictor": (
            bl.ProfileTable(lookup, bl.marginally_admissible(d_cal, alpha), cell.deadline_ms),
            cert0,
        ),
        "compression_only": (GreedySafe(model, mission), Certifier(lam0, menu=mask(COMPRESSION))),
        "learned_only": (GreedySafe(model, mission), Certifier(lam0, menu=mask(LEARNED))),
        "no_learned": (GreedySafe(model, mission), Certifier(lam0, menu=mask(learned_free))),
        "audit_aci": (
            GreedySafe(model, mission),
            Certifier(lam0, aci=AdaptiveLambda(calib_m, alpha)),
        ),
        "oracle": (Oracle(lookup, mission), cert0),
    }


def metrics(
    res: Result, cell: Cell, model: CostModel, late_cap_factor: float = 10.0
) -> dict[str, float]:
    """Per-run metrics (PRD §9.3). Costs are realised: actual bytes, CPU time and lateness."""
    n = len(res.rep)
    sent = res.rep >= 0
    certified = res.flag == FLAGS.index("certified")
    cap = late_cap_factor * cell.deadline_ms
    lateness = np.where(
        np.isnan(res.latency_ms), cap, np.clip(res.latency_ms - cell.deadline_ms, 0.0, cap)
    )
    energy = (
        model.energy.p_cpu_w * res.edge_ms / 1000.0 + model.energy.e_bit_j * res.wire_bytes * 8.0
    )
    pr = model.prices
    cost = (
        pr.p_byte * res.wire_bytes
        + pr.p_j * energy
        + pr.p_cpu_ms * res.edge_ms
        + pr.p_late_ms * lateness
    )
    delivered = res.latency_ms[~np.isnan(res.latency_ms)]
    out = {
        "cost": float(cost.mean()),
        "bytes": float(res.wire_bytes.mean()),
        "energy_j": float(energy.mean()),
        "edge_ms": float(res.edge_ms.mean()),
        "slo_violation": float(res.late.mean()),
        "risk_certified": float(res.disagreed[certified & sent].mean())
        if (certified & sent).any()
        else 0.0,
        # what Proposition 1 bounds: disagreeing *and* certified, per offered window. The rate
        # among certified decisions above divides by P(certified) and is not bounded by α.
        "risk_certified_joint": float((res.disagreed & certified & sent).sum() / n),
        "risk_sent": float(res.disagreed[sent].mean()) if sent.any() else 0.0,
        "risk_all": float(res.disagreed.sum() / n),  # undelivered windows cannot disagree
        "probe_rate": float(res.probed.mean()),  # < 1 only for a cascade (ADR 0011)
        "switching_rate": float(np.mean(np.diff(res.rep[sent]) != 0)) if sent.sum() > 1 else 0.0,
        "audit_bytes": float(
            res.audited.sum() * (model.profiles.size_bytes[0] + OVERHEAD_BYTES) / n
        ),
    }
    if cell.p_audit:  # what the cloud can estimate online from the audited decisions alone
        audited_losses = int((res.audited & res.disagreed).sum())
        out["risk_audit_estimate"] = ipw_risk(audited_losses, n, cell.p_audit)
    for q in (50, 95, 99):
        out[f"latency_p{q}_ms"] = (
            float(np.percentile(delivered, q)) if len(delivered) else float("nan")
        )
    for flag in FLAGS:
        out[f"rate_{flag}"] = res.flag_rate(flag)
    for k, rep_id in enumerate(model.profiles.rep_ids):
        out[f"share/{rep_id}"] = float(np.mean(res.rep == k))
    return out


def run_cell(
    ctx: FoldCtx,
    cell: Cell,
    mission: Mission,
    which: str,
    n_windows: int,
    seed: int,
    only: tuple[str, ...] = (),
    late_cap_factor: float = 10.0,
) -> list[dict[str, float | str]]:
    """Simulate every policy of the zoo on one cell. Returns one metrics row per policy."""
    work = ctx.workload(n_windows)
    scenario = Scenario(
        make_trace(cell, seed), cell.deadline_ms, cell.load, cell.n_sensors, ctx.window_period_ms,
        cell.rtt_ms, cell.energy_budget_j, cell.storage_bytes, cell.p_audit, seed,
    )  # fmt: skip
    rows: list[dict[str, float | str]] = []
    for name, (selector, certifier) in policies(ctx, cell, mission, which).items():
        if only and name not in only:
            continue
        # a learning baseline needs audits to learn from; it pays for their raw copies
        wanted = max(scenario.p_audit, getattr(selector, "audit_rate", 0.0))
        res = simulate(replace(scenario, p_audit=wanted), work, selector, certifier, ctx.model)
        audited_cell = replace(cell, p_audit=wanted)
        rows.append({"policy": name, **metrics(res, audited_cell, ctx.model, late_cap_factor)})
    return rows
