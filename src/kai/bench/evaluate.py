"""Scenario cells, the policy zoo (Kai, baselines B1–B11, ablations) and per-run metrics."""

from __future__ import annotations

import json
import os
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from kai.certify.fit import FIT_VERSION, fit_predictor
from kai.certify.predictor import Predictor
from kai.certify.runtime import Certifier
from kai.cloud.heads import TreeHead, load_head
from kai.contracts.frame import OVERHEAD_BYTES
from kai.contracts.schemas import FLAGS, EnergyModel, LambdaTable, Partition, Prices, RepProfile
from kai.contracts.windows import WindowSet
from kai.link import traces as tr
from kai.link.des import Result, Scenario, Workload, simulate
from kai.reps.registry import WAVEFORM_REPS, load_menu
from kai.select import baselines as bl
from kai.select.base import Mission, Selector
from kai.select.cost import CostModel, Overheads, Profiles, State
from kai.select.policies import CascadeSafe, GreedySafe, GreedyStatic, MpcSafe, Oracle, Static
from kai.tasks.pipeline import Table, latent_features

MEASURED_PREFIX = "ucc4g_"  # links named ucc4g_<mobility class> replay measured logs
MEASURED_DIR = ROOT / "data/raw/traces/ucc4g"  # one set of traces for every dataset
LEARNED = ("r0_raw", "r5_ae", "r6_bottleneck", "r7_vq")
# ADR 0025 C: policies scored by ĝ_z and charged the r6 encode on every window
LATENT_POLICIES = ("kai_s1_v0_latent", "kai_s1_v0_abc")
# ADR 0026: scored by the head committee's ĝ_D; the last one also sends the committee's decisions
COMMITTEE_POLICIES = ("kai_s1_v0_committee", "kai_s1_v0_late_committee")
DEC_POLICY = "kai_s1_v0_late_committee_dec"
EDGE_POLICY = "b14_edge_inference"
AGREEMENT_MIN = 0.999  # share of windows where a committee head's argmax must match the table
SEL_POLICIES = ("kai_s1_v0_zsel", "kai_s1_v0_ab_zsel")  # C′: per-rep choice of ĝ or ĝ_z


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
    group: dict[float, float]  # per α: V0 with one calibration sample per unit (ADR 0020)
    m_cal: np.ndarray  # per calibration window: the first-loss threshold (CRC, ACI ladder)
    # ADR 0025 C (exploratory); None / empty when the fold has no r6 or no windows path was given
    ghat_z: np.ndarray | None = None
    lam_z: dict[float, float] = field(default_factory=dict)  # per α: V0 on ĝ_z
    model_z: CostModel | None = None
    # ADR 0025 C′: per representation, the score the fold's ptrain CV chose (ĝ or ĝ_z)
    ghat_sel: np.ndarray | None = None
    use_z: np.ndarray | None = None
    lam_sel: dict[float, float] = field(default_factory=dict)
    model_sel: CostModel | None = None
    # ADR 0026 (exploratory): head-committee score, its cost models, decision payloads, edge f_raw
    ghat_d: np.ndarray | None = None
    lam_d: dict[float, float] = field(default_factory=dict)
    d_dec: np.ndarray | None = None  # d with each member's column = what the edge sends (its own ŷ)
    lam_dec: dict[float, float] = field(default_factory=dict)  # V0 on ĝ_D and d_dec
    model_d: CostModel | None = None
    model_dec: CostModel | None = None
    committee: tuple[int, ...] = ()
    model_edge: CostModel | None = None
    sizes: dict[str, int] = field(default_factory=dict)  # model_bytes of the fold

    @property
    def latent_ghat_cal(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.ghat_z is not None
        m = self.scored.table.of("calib")
        return self.ghat_z[m], self.scored.table.d[m]

    @property
    def rep_ids(self) -> tuple[str, ...]:
        return self.scored.table.rep_ids

    def workload(
        self,
        n: int,
        partition: Partition = "test",
        scores: Literal["plain", "latent", "sel", "committee", "dec", "edge"] = "plain",
    ) -> Workload:
        t = self.scored.table
        idx = spread(np.flatnonzero(t.of(partition)), n)
        ghat = {
            "plain": self.scored.ghat, "latent": self.ghat_z, "sel": self.ghat_sel,
            "committee": self.ghat_d, "dec": self.ghat_d, "edge": self.scored.ghat,
        }[scores]  # fmt: skip
        assert ghat is not None, "no latent scores for this fold"
        payload, d = t.payload_bytes[idx].copy(), t.d[idx]
        if scores == "dec":
            assert self.d_dec is not None
            payload[:, list(self.committee)] = 1  # the decision, one byte
            d = self.d_dec[idx]
        if scores == "edge":
            payload[:, 0] = 1  # f_raw ran on the edge: its decision is sent
        return Workload(ghat[idx], t.phi[idx], d, payload)


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
    windows_path: Path | None = None,
    edge_slowdown: float = 1.0,
) -> FoldCtx:
    """`edge_slowdown` must be the one the profiles were loaded with: head times that run on the
    edge (b14's f_raw, the committee's heads) are scaled by it too (ADR 0027)."""
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
    m_cal = crc.first_loss_threshold(ghat_cal, d_cal)
    units = scored.table.bearing_id[scored.table.of("calib")]
    group = {a: crc.group_crc_threshold(m_cal, units, a) for a in alphas}
    latent = latent_scores(fold_dir, scored.table, windows_path, predictor_kind, seed)
    ghat_z = use_z = ghat_sel = model_z = model_sel = None
    lam_z: dict[float, float] = {}
    lam_sel: dict[float, float] = {}
    if latent is not None:
        ghat_z, use_z = latent
        cal = scored.table.of("calib")
        m_z = crc.first_loss_threshold(ghat_z[cal], d_cal)
        lam_z = {a: crc.crc_threshold(m_z, a) for a in alphas}
        # the edge runs the r6 encoder on every window to get z, so that time joins the per-window
        # overhead and r6's own encode is free (the latent is already computed)
        k6 = scored.table.rep_ids.index("r6_bottleneck")
        t_enc = model.profiles.t_enc_ms.copy()
        t_z = float(t_enc[k6])
        t_enc[k6] = 0.0
        model_z = replace(
            model,
            profiles=replace(model.profiles, t_enc_ms=t_enc),
            overheads=replace(model.overheads, t_probe_ms=model.overheads.t_probe_ms + t_z),
        )
        # C′: the chosen score per representation; the r6 encode is paid only if some rep uses z
        ghat_sel = np.where(use_z[None, :], ghat_z, scored.ghat)
        m_sel = crc.first_loss_threshold(ghat_sel[cal], d_cal)
        lam_sel = {a: crc.crc_threshold(m_sel, a) for a in alphas}
        model_sel = model_z if use_z.any() else model
    ghat_d, committee, d_dec = committee_scores(
        fold_dir, scored.table, windows_path, predictor_kind, seed
    )
    lam_d: dict[float, float] = {}
    lam_dec: dict[float, float] = {}
    model_d = model_dec = None
    if ghat_d is not None:
        assert d_dec is not None
        cal = scored.table.of("calib")
        m_d = crc.first_loss_threshold(ghat_d[cal], d_cal)
        lam_d = {a: crc.crc_threshold(m_d, a) for a in alphas}
        m_dec = crc.first_loss_threshold(ghat_d[cal], d_dec[cal])
        lam_dec = {a: crc.crc_threshold(m_dec, a) for a in alphas}
        model_d, model_dec = committee_models(model, committee, edge_slowdown)
    model_edge = edge_model(model, edge_slowdown)
    return FoldCtx(
        scored, model, lambdas, window_period_ms, static, cascade, group, m_cal,
        ghat_z, lam_z, model_z, ghat_sel, use_z, lam_sel, model_sel,
        ghat_d, lam_d, d_dec, lam_dec, model_d, model_dec, committee, model_edge,
        model_bytes(fold_dir, predictor_kind),
    )  # fmt: skip


def committee_models(
    model: CostModel, committee: tuple[int, ...], edge_slowdown: float = 1.0
) -> tuple[CostModel, CostModel]:
    """(model_d, model_dec) of ADR 0026. The edge encodes, decodes and runs every member's head on
    every window (probe work), so a member's own encode is free when it is sent. Head times are the
    stored per-rep t_head_ms (this machine's profile), the same measurement b14 uses for f_raw.
    With decision payloads the edge already holds the cloud's answer: 1 byte, no cloud work.
    The heads run on the edge, so their time is scaled by `edge_slowdown` (encodes already are)."""
    c, p = list(committee), model.profiles
    t_enc = p.t_enc_ms.copy()
    t_enc[c] = 0.0
    probe = model.overheads.t_probe_ms + float((p.t_enc_ms + edge_slowdown * p.t_head_ms)[c].sum())
    model_d = replace(
        model,
        profiles=replace(p, t_enc_ms=t_enc),
        overheads=replace(model.overheads, t_probe_ms=probe),
    )
    size, t_dec, t_head = (a.copy() for a in (p.size_bytes, p.t_dec_ms, p.t_head_ms))
    size[c], t_dec[c], t_head[c] = 1.0, 0.0, 0.0
    return model_d, replace(
        model_d,
        profiles=replace(model_d.profiles, size_bytes=size, t_dec_ms=t_dec, t_head_ms=t_head),
    )


def edge_model(model: CostModel, edge_slowdown: float = 1.0) -> CostModel:
    """b14: f_raw on the edge, one byte sent, d = 0. Its time is raw's stored t_head_ms: the same
    profile measurement as the committee's head times (timing parity, ADR 0026), scaled by
    `edge_slowdown` like every other edge time (ADR 0027)."""
    p = model.profiles
    size, t_enc, t_dec, t_head = (
        a.copy() for a in (p.size_bytes, p.t_enc_ms, p.t_dec_ms, p.t_head_ms)
    )
    size[0], t_enc[0], t_dec[0], t_head[0] = 1.0, edge_slowdown * p.t_head_ms[0], 0.0, 0.0
    return replace(
        model,
        profiles=replace(p, size_bytes=size, t_enc_ms=t_enc, t_dec_ms=t_dec, t_head_ms=t_head),
    )


def choose_latent(
    phi: np.ndarray,
    z: np.ndarray,
    d: np.ndarray,
    groups: np.ndarray,
    ptrain: np.ndarray,
    rep_ids: tuple[str, ...],
    kind: Literal["mlp", "logistic"] = "mlp",
    seed: int = 0,
    n_splits: int = 3,
) -> np.ndarray:
    """C′ (ADR 0025 addendum): per representation, does [φ, z] beat φ alone?

    Unit-grouped CV on the `ptrain` rows only: use_z[k] iff the mean held-out AUC against d_k is
    strictly higher with z. Ties, one-class held-out parts and raw (k = 0) go to φ.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    k_all = len(rep_ids)
    phi, z, d, groups = phi[ptrain], z[ptrain], d[ptrain], groups[ptrain]
    both = np.hstack([phi, z])
    auc: dict[bool, list[list[float]]] = {
        False: [[] for _ in range(k_all)],
        True: [[] for _ in range(k_all)],
    }
    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        for tr, te in GroupKFold(min(n_splits, n_groups)).split(phi, groups=groups):
            for with_z, x in ((False, phi), (True, both)):
                g = fit_predictor(x[tr], d[tr], rep_ids, kind, seed).predict(x[te])
                for k in range(1, k_all):
                    if 0 < d[te, k].sum() < len(te):
                        auc[with_z][k].append(float(roc_auc_score(d[te, k], g[:, k])))
    # the two fits see the same splits, so a rep has the same number of valid splits in both
    return np.array(
        [
            bool(auc[True][k]) and np.mean(auc[True][k]) > np.mean(auc[False][k])
            for k in range(k_all)
        ]
    )


def latent_scores(
    fold_dir: Path,
    table: Table,
    windows_path: Path | None,
    kind: Literal["mlp", "logistic"] = "mlp",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """(ĝ_z, use_z): the predictor on [φ, z] with z the fold's own r6 latent (ADR 0025 C), and
    the per-representation choice between ĝ and ĝ_z (C′).

    Everything is fitted or chosen on `ptrain` only and cached next to the fold. None when the
    fold has no r6 or no windows path was given.
    """
    if windows_path is None or "r6_bottleneck" not in table.rep_ids:
        return None
    menu = fold_dir / "menu"
    key = f"{FIT_VERSION}|{seed}|z|".encode() + b"".join(
        p.read_bytes()
        for p in (fold_dir / "table.npz", menu / "r6_bottleneck.onnx", menu / "r6_bottleneck.npy")
        if p.exists()
    )
    tag = f"{kind}_{zlib.crc32(key):08x}"
    cached = fold_dir / f"predictor_z_{tag}.npz"
    # the scores themselves are cached too: every grid worker loads every fold, and recomputing z
    # means reading all windows and running the r6 encoder over them
    cached_ghat = fold_dir / f"ghat_z_{tag}.npy"
    cached_use = fold_dir / f"use_z_{tag}.npy"
    if cached.exists() and cached_ghat.exists() and cached_use.exists():
        return np.load(cached_ghat), np.load(cached_use)
    ws = WindowSet.load(windows_path)
    if not np.array_equal(ws.window_id, table.window_id):
        raise ValueError(f"{windows_path}: window ids differ from {fold_dir / 'table.npz'}")
    rep = next(r for r in load_menu(menu) if r.rep_id == "r6_bottleneck")
    z = latent_features(rep, ws.x)  # type: ignore[arg-type]  # r6 is an OnnxLatent
    feats = np.hstack([table.phi, z])
    ptrain = table.of("ptrain")
    if cached.exists():
        predictor = Predictor.load(cached)
    else:
        predictor = fit_predictor(feats[ptrain], table.d[ptrain], table.rep_ids, kind, seed)
        _atomic(fold_dir, f"predictor_z_{tag}.npz", predictor.save)
    ghat_z = predictor.predict(feats)
    use_z = choose_latent(
        table.phi, z, table.d, table.bearing_id, ptrain, table.rep_ids, kind, seed
    )
    _atomic(fold_dir, f"ghat_z_{tag}.npy", lambda p: _save_npy(p, ghat_z))
    _atomic(fold_dir, f"use_z_{tag}.npy", lambda p: _save_npy(p, use_z))
    return ghat_z, use_z


def committee_features(
    fold_dir: Path, table: Table, windows_path: Path
) -> tuple[np.ndarray, tuple[int, ...]]:
    """ADR 0026 D: `[p_r for r in committee] ⊕ votes` for every window, and the committee's
    columns in the table. The committee is every menu rep whose cloud head takes features; a
    member's p_r is its head's probabilities on `decode_for_head(encode(x))`, as the cloud does.

    Raises if a member's argmax disagrees with the stored ŷ_r on more than 0.1 % of windows.
    """
    from kai.reps.latent import OnnxLatent, OnnxVQ

    ws = WindowSet.load(windows_path)
    if not np.array_equal(ws.window_id, table.window_id):
        raise ValueError(f"{windows_path}: window ids differ from {fold_dir / 'table.npz'}")
    n_classes = int(max(table.label.max(), table.yhat.max())) + 1
    reps = [r for r in load_menu(fold_dir / "menu") if r.head_input == "features"]
    if not reps:
        return np.zeros((len(table.window_id), 0), np.float32), ()
    probas, rates = [], {}
    for rep in reps:
        k = table.rep_ids.index(rep.rep_id)
        if isinstance(rep, OnnxLatent | OnnxVQ):  # batched, through the same path as the wire
            feats = latent_features(rep, ws.x)
        else:
            feats = np.stack([rep.decode_for_head(rep.encode(x)) for x in ws.x])
        head = load_head(fold_dir / "heads" / rep.rep_id)
        p = head.proba(feats, n_classes) if isinstance(head, TreeHead) else head.proba(feats)
        p = np.pad(p, ((0, 0), (0, n_classes - p.shape[1])))  # a head may lack the top classes
        rates[rep.rep_id] = float((p.argmax(axis=1) == table.yhat[:, k]).mean())
        probas.append(p)
    if any(r < AGREEMENT_MIN for r in rates.values()):
        raise ValueError(f"committee heads disagree with the table's ŷ (rates: {rates})")
    votes = sum(np.eye(n_classes)[p.argmax(axis=1)] for p in probas)
    feats_all = np.hstack([*probas, votes]).astype(np.float32)
    return feats_all, tuple(table.rep_ids.index(r.rep_id) for r in reps)


def _fold_key(fold_dir: Path, seed: int, tag: bytes) -> bytes:
    """FIT_VERSION, seed, the table and every stored model file: the committee's cache key."""
    menu, heads = fold_dir / "menu", fold_dir / "heads"
    files = [
        fold_dir / "table.npz", menu / "reps.json", *sorted(menu.glob("*.onnx")),
        *sorted(menu.glob("*.npy")), *sorted(heads.glob("*")),
    ]  # fmt: skip
    blobs = b"".join(p.read_bytes() for p in files if p.is_file())
    return f"{FIT_VERSION}|{seed}|".encode() + tag + blobs


def committee_scores(
    fold_dir: Path,
    table: Table,
    windows_path: Path | None,
    kind: Literal["mlp", "logistic"] = "mlp",
    seed: int = 0,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray] | tuple[None, tuple[int, ...], None]:
    """(ĝ_D, committee columns, d_dec): the predictor on the committee features, fitted on `ptrain`
    only, and the loss of the decision payload (`table.d` with each member's column replaced by
    1[member's own argmax ≠ f_raw's decision]: what the edge actually sends).

    Features, predictor, ĝ_D, d_dec and the committee are cached next to the fold. (None, (), None)
    when no windows path was given or the fold has no committee.
    """
    if windows_path is None:
        return None, (), None
    tag = f"{kind}_{zlib.crc32(_fold_key(fold_dir, seed, b'D|')):08x}"
    names = {
        k: f"{k}_{tag}.{e}"
        for k, e in (("feats", "npy"), ("d_dec", "npy"), ("idx", "npy"), ("ghat_d", "npy"))
    }
    if all((fold_dir / v).exists() for v in names.values()):
        return (
            np.load(fold_dir / names["ghat_d"]),
            tuple(np.load(fold_dir / names["idx"]).tolist()),
            np.load(fold_dir / names["d_dec"]),
        )
    feats, committee = committee_features(fold_dir, table, windows_path)
    if not committee:
        return None, (), None
    ptrain = table.of("ptrain")
    predictor = fit_predictor(feats[ptrain], table.d[ptrain], table.rep_ids, kind, seed)
    _atomic(fold_dir, f"predictor_d_{tag}.npz", predictor.save)
    ghat_d = predictor.predict(feats)
    n_cls = feats.shape[1] // (len(committee) + 1)
    d_dec = table.d.copy()
    for j, k in enumerate(committee):
        sent = feats[:, j * n_cls : (j + 1) * n_cls].argmax(axis=1)
        d_dec[:, k] = sent != table.yhat[:, 0]
    _atomic(fold_dir, names["d_dec"], lambda p: _save_npy(p, d_dec))
    _atomic(fold_dir, names["feats"], lambda p: _save_npy(p, feats))
    _atomic(fold_dir, names["idx"], lambda p: _save_npy(p, np.array(committee)))
    _atomic(fold_dir, names["ghat_d"], lambda p: _save_npy(p, ghat_d))  # last: marks it complete
    return ghat_d, committee, d_dec


def model_bytes(fold_dir: Path, kind: str = "mlp") -> dict[str, int]:
    """Size in bytes of every stored edge model: `f_raw`, `enc/<rep>`, `head/<rep>`, `predictor`,
    `predictor_d` (the newest cached file of each; absent ones are left out)."""
    out: dict[str, int] = {}
    if (fold_dir / "f_raw.onnx").exists():
        out["f_raw"] = (fold_dir / "f_raw.onnx").stat().st_size
    menu = fold_dir / "menu"
    index = json.loads((menu / "reps.json").read_text()) if (menu / "reps.json").exists() else []
    for entry in index:
        rid = entry["rep_id"]
        out[f"enc/{rid}"] = sum(p.stat().st_size for p in menu.glob(f"{rid}.*"))
        if heads := list((fold_dir / "heads").glob(f"{rid}.*")):
            out[f"head/{rid}"] = sum(p.stat().st_size for p in heads)
    for name, pattern in (
        ("predictor", f"predictor_{kind}_*.npz"),
        ("predictor_d", f"predictor_d_{kind}_*.npz"),
    ):
        if files := sorted(fold_dir.glob(pattern), key=lambda p: p.stat().st_mtime):
            out[name] = files[-1].stat().st_size
    return out


def required_bytes(ctx: FoldCtx, name: str, selector: Selector) -> int:
    """Edge model memory a policy needs (ADR 0026): b14 f_raw; the committee policies the
    committee's encoders and heads plus ĝ_D; a static policy its encoder; other Kai policies and
    baselines the whole menu's encoders (and the predictor for Kai). The oracle needs nothing."""
    sz, ids = ctx.sizes, ctx.rep_ids
    if name == EDGE_POLICY:
        return sz.get("f_raw", 0)
    if name in (*COMMITTEE_POLICIES, DEC_POLICY):
        members = [ids[k] for k in ctx.committee]
        return sum(sz.get(f"enc/{r}", 0) + sz.get(f"head/{r}", 0) for r in members) + sz.get(
            "predictor_d", 0
        )
    if name == "oracle":
        return 0
    if isinstance(selector, Static):
        return sz.get(f"enc/{ids[selector.rep]}", 0)
    menu = sum(sz.get(f"enc/{r}", 0) for r in ids)
    return menu + (sz.get("predictor", 0) if name.startswith("kai_") else 0)


def _save_npy(path: Path, a: np.ndarray) -> None:
    with path.open("wb") as f:  # a file object: np.save would append ".npy" to the tmp name
        np.save(f, a)


def _atomic(directory: Path, name: str, write: Callable[[Path], None]) -> None:
    """Several grid workers may reach a new fold at once: write to a private name, then rename."""
    tmp = directory / f".{os.getpid()}.{name}"
    write(tmp)
    tmp.replace(directory / name)


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
) -> dict[str, Callable[[], tuple[Selector, Certifier]]]:
    """name → builder of (selector, certifier). `which`: "main" (E3) or "ablation" (E5).

    Builders are lazy: a run restricted to a few policies never fits V2 or the best single.
    """
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
    latent: dict[str, Callable[[], tuple[Selector, Certifier]]] = {}
    if ctx.ghat_z is not None and ctx.model_z is not None and which == "main":
        model_z, lam_z = ctx.model_z, ctx.lam_z[alpha]
        latent = {
            "kai_s1_v0_latent": lambda: (GreedySafe(model_z, mission), Certifier(lam_z)),
            "kai_s1_v0_abc": lambda: (
                GreedySafe(model_z, "certified_late", defer_rep=idx("r1_lossless")),
                Certifier(lam_z),
            ),
        }
    if ctx.ghat_sel is not None and ctx.model_sel is not None and which == "main":
        model_sel, lam_sel = ctx.model_sel, ctx.lam_sel[alpha]
        latent |= {
            "kai_s1_v0_zsel": lambda: (GreedySafe(model_sel, mission), Certifier(lam_sel)),
            "kai_s1_v0_ab_zsel": lambda: (
                GreedySafe(model_sel, "certified_late", defer_rep=idx("r1_lossless")),
                Certifier(lam_sel),
            ),
        }
    if ctx.ghat_d is not None and ctx.model_d is not None and which == "main":
        model_d, lam_d = ctx.model_d, ctx.lam_d[alpha]
        model_dec = ctx.model_dec or model_d
        defer = idx("r1_lossless")
        latent |= {
            "kai_s1_v0_committee": lambda: (GreedySafe(model_d, mission), Certifier(lam_d)),
            "kai_s1_v0_late_committee": lambda: (
                GreedySafe(model_d, "certified_late", defer_rep=defer),
                Certifier(lam_d),
            ),
            DEC_POLICY: lambda: (
                GreedySafe(model_dec, "certified_late", defer_rep=defer),
                Certifier(ctx.lam_dec[alpha]),
            ),
        }
    if which == "main":
        marginal = bl.marginally_admissible(d_cal, alpha)
        return {
            "b1_raw": lambda: (Static(0, "b1_raw"), cert0),
            "b2_compress": lambda: (Static(idx("r2a_wavelet"), "b2_compress"), cert0),
            "b3_features": lambda: (Static(idx("r3_spectral"), "b3_features"), cert0),
            "b4_best_single": lambda: (
                Static(bl.best_single(marginal, static, nominal), "b4_best_single"),
                cert0,
            ),
            "b5_context_blind": lambda: (bl.ContextBlind(model), cert0),
            "b8_profile_table": lambda: (
                bl.ProfileTable(lookup, marginal, cell.deadline_ms),
                cert0,
            ),
            "b9_adaptive_threshold": lambda: (
                bl.AdaptiveThreshold((idx("r2b_wavelet"), idx("r2a_wavelet"), idx("r1_lossless"))),
                cert0,
            ),
            "b10_two_mode": lambda: (
                bl.TwoModeSwitch(lookup, idx("r6_bottleneck"), idx("r1_lossless")),
                cert0,
            ),
            "b11_cms": lambda: (
                bl.ConformalModelSelection(lookup, bl.ltt_admissible(d_cal, alpha, 0.1), mission),
                cert0,
            ),
            EDGE_POLICY: lambda: (Static(0, EDGE_POLICY), cert0),
            "b13_adaedge": lambda: (
                bl.AdaEdgeBandit(lookup, idx("r1_lossless"), np.random.default_rng(13)),
                cert0,
            ),
            "kai_s1_v0": lambda: (GreedySafe(model, mission), cert0),
            "kai_s1_v1": lambda: (
                GreedySafe(model, mission),
                Certifier(threshold(table, "v1", alpha, ids)),
            ),
            "kai_s1_v1_per_rep": lambda: (
                GreedySafe(model, mission),
                Certifier(threshold(table, "v1_per_rep", alpha, ids)),
            ),
            # V0 on min(ĝ, prior): probe only when it can pay for itself (ADR 0011)
            "kai_s1_cascade": lambda: (
                CascadeSafe(model, mission),
                Certifier(ctx.cascade[alpha][1], prior=ctx.cascade[alpha][0]),
            ),
            # the same certificate with constant scores: one safe set, no probe, no predictor
            "kai_s1_v0_static": lambda: (
                GreedyStatic(lookup, mission),
                Certifier(np.where(ctx.static_safe[alpha], crc.ALL, crc.RAW_ONLY)),
            ),
            "kai_s1_v2": lambda: (GreedySafe(model, mission), Certifier(v2_threshold(ctx, cell))),
            "kai_s1_v0_group": lambda: (GreedySafe(model, mission), Certifier(ctx.group[alpha])),
            "kai_s2_v0": lambda: (MpcSafe(model, period, mission=mission), cert0),
            "kai_s1_v0_live": lambda: (GreedySafe(model, "liveness_first"), cert0),
            # shortcut: simulator only (ADR 0025); the edge runtime needs this before E7 or a confirmatory run  # noqa: E501
            "kai_s1_v0_late": lambda: (GreedySafe(model, "certified_late"), cert0),
            "kai_s1_v0_lldefer": lambda: (
                GreedySafe(model, mission, defer_rep=idx("r1_lossless")),
                cert0,
            ),
            **latent,
            "oracle": lambda: (Oracle(lookup, mission), cert0),
        }
    no_enc = replace(model, profiles=model.profiles.without_encode_cost())
    learned_free = tuple(r for r in ids if r not in LEARNED[1:])
    return {
        "kai_s1_v0": lambda: (GreedySafe(model, mission), cert0),
        "no_encode_cost": lambda: (GreedySafe(no_enc, mission), cert0),
        "no_queue_awareness": lambda: (GreedySafe(model, mission, queue_aware=False), cert0),
        "no_certificate": lambda: (
            GreedySafe(model, mission),
            Certifier(alpha),
        ),  # raw ĝ ≤ α, uncalibrated
        "no_predictor": lambda: (
            bl.ProfileTable(lookup, bl.marginally_admissible(d_cal, alpha), cell.deadline_ms),
            cert0,
        ),
        "compression_only": lambda: (
            GreedySafe(model, mission),
            Certifier(lam0, menu=mask(WAVEFORM_REPS)),
        ),
        "learned_only": lambda: (GreedySafe(model, mission), Certifier(lam0, menu=mask(LEARNED))),
        "no_learned": lambda: (
            GreedySafe(model, mission),
            Certifier(lam0, menu=mask(learned_free)),
        ),
        "audit_aci": lambda: (
            GreedySafe(model, mission),
            Certifier(lam0, aci=AdaptiveLambda(ctx.m_cal, alpha)),
        ),
        "oracle": lambda: (Oracle(lookup, mission), cert0),
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
        # nothing certified: the rate is undefined (NaN, skipped by means), never a perfect 0
        "risk_certified": float(res.disagreed[certified & sent].mean())
        if (certified & sent).any()
        else float("nan"),
        # what Proposition 1 bounds: disagreeing *and* certified, per offered window. The rate
        # among certified decisions above divides by P(certified) and is not bounded by α.
        "risk_certified_joint": float((res.disagreed & certified & sent).sum() / n),
        "rate_certified_on_time": float((certified & sent & ~res.late).mean()),
        "risk_sent": float(res.disagreed[sent].mean()) if sent.any() else 0.0,
        "risk_all": float(res.disagreed.sum() / n),  # undelivered windows cannot disagree
        "probe_rate": float(res.probed.mean()),  # < 1 only for a cascade (ADR 0011)
        "switching_rate": float(np.mean(np.diff(res.rep[sent]) != 0)) if sent.sum() > 1 else 0.0,
        "audit_bytes": float(
            res.audited.sum() * (model.profiles.size_bytes[0] + OVERHEAD_BYTES) / n
        ),
    }
    # audit copies still queued (or dropped) at the end never fed back to the edge
    out["rate_audit_unfed"] = (
        float((res.audited & ~res.audit_fed).sum() / res.audited.sum())
        if res.audited.any()
        else 0.0
    )
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
    aci_p_audit: float = 0.0,
) -> list[dict[str, float | str]]:
    """Simulate every policy of the zoo on one cell. Returns one metrics row per policy.

    `aci_p_audit` is the audit rate of a policy whose certifier runs ACI: it learns from audits.
    Every row carries `required_bytes`, the stored model files the policy needs on the edge.
    """
    # policy name → (workload, truth model); the latent ones see ĝ_z and pay the r6 encode on
    # every window, the zsel ones see the chosen scores and pay it only if some rep uses z
    plain = (ctx.workload(n_windows), ctx.model)
    special: dict[str, tuple] = {}
    if ctx.ghat_z is not None:
        special |= dict.fromkeys(
            LATENT_POLICIES, (ctx.workload(n_windows, scores="latent"), ctx.model_z)
        )
    if ctx.ghat_sel is not None:
        special |= dict.fromkeys(
            SEL_POLICIES, (ctx.workload(n_windows, scores="sel"), ctx.model_sel)
        )
    if ctx.model_edge is not None and which == "main":
        special[EDGE_POLICY] = (ctx.workload(n_windows, scores="edge"), ctx.model_edge)
    if ctx.ghat_d is not None:
        special |= dict.fromkeys(
            COMMITTEE_POLICIES, (ctx.workload(n_windows, scores="committee"), ctx.model_d)
        )
        special[DEC_POLICY] = (ctx.workload(n_windows, scores="dec"), ctx.model_dec)
    scenario = Scenario(
        make_trace(cell, seed), cell.deadline_ms, load=cell.load, n_sensors=cell.n_sensors,
        window_period_ms=ctx.window_period_ms, rtt_ms=cell.rtt_ms,
        energy_budget_j=cell.energy_budget_j, storage_bytes=cell.storage_bytes,
        p_audit=cell.p_audit, seed=seed,
    )  # fmt: skip  # keywords: a positional seed once landed in probe_s (ADR 0023)
    rows: list[dict[str, float | str]] = []
    for name, build in policies(ctx, cell, mission, which).items():
        if only and name not in only:
            continue
        selector, certifier = build()
        # a learning baseline (or ACI) needs audits to learn from; it pays for their raw copies
        aci_rate = aci_p_audit if certifier.aci is not None else 0.0
        wanted = max(scenario.p_audit, getattr(selector, "audit_rate", 0.0), aci_rate)
        w, truth = special.get(name, plain)
        res = simulate(replace(scenario, p_audit=wanted), w, selector, certifier, truth)
        audited_cell = replace(cell, p_audit=wanted)
        rows.append(
            {
                "policy": name,
                **metrics(res, audited_cell, truth, late_cap_factor),
                "required_bytes": float(required_bytes(ctx, name, selector)),
            }
        )
    return rows
