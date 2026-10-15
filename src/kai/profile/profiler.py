"""Profiler harness (PRD E1): payload size, encode / decode / head time, and time versus edge load.

Timing protocol (PRD §9.6): `perf_counter_ns`, warm-up, ≥ 30 repetitions, median + MAD, one thread.
Edge load is produced by `m` always-busy co-tenant processes pinned to the same core as the
measurement: the scheduler's fair share then leaves the encoder 1/(m+1) of the core, i.e.
ρ = m/(m+1) — 0, 0.5 and 0.8 for m = 0, 1, 4.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Callable

import numpy as np

from kai.certify.predictor import Predictor
from kai.certify.probe import probe
from kai.cloud.heads import HeadZoo
from kai.contracts.schemas import RepProfile
from kai.reps.base import BaseRep
from kai.select.cost import Overheads

LOAD_TENANTS = {0.0: 0, 0.5: 1, 0.8: 4}


def time_ms(
    fn: Callable[[int], object],
    n_items: int,
    repeats: int = 30,
    warmup: int = 5,
    period_ms: float = 0.0,
) -> np.ndarray:
    """Wall time in ms of `fn(i)` for `repeats` items after `warmup` untimed calls.

    With `period_ms > 0` one call starts per period, like an edge that handles one window per
    period: the process sleeps for whatever is left of the period after the call. That matters:
    after an idle gap the caches are cold and the CPU has clocked down, and the same call takes
    several times longer than in a tight loop. A call longer than the period leaves no idle time,
    which is the tight loop again — a node with many sensors is never idle (ADR 0008).
    """
    for i in range(warmup):
        fn(i % n_items)
    out = np.empty(repeats)
    spent_ns = 0.0
    for j in range(repeats):
        _rest(period_ms, spent_ns)
        t0 = time.perf_counter_ns()
        fn(j % n_items)
        spent_ns = float(time.perf_counter_ns() - t0)
        out[j] = spent_ns / 1e6
    return out


def _rest(period_ms: float, spent_ns: float) -> None:
    """Sleep for what is left of the period after work that took `spent_ns`."""
    if period_ms * 1e6 > spent_ns:
        time.sleep((period_ms * 1e6 - spent_ns) / 1e9)


def summarise(samples: np.ndarray) -> tuple[float, float, float]:
    """(median, MAD, q95)."""
    med = float(np.median(samples))
    return med, float(np.median(np.abs(samples - med))), float(np.quantile(samples, 0.95))


def _burn(core: int) -> None:
    os.sched_setaffinity(0, {core})
    while True:
        pass


class co_tenants:
    """Pin this process to one core and start `m` busy processes on the same core."""

    def __init__(self, m: int, core: int | None = None) -> None:
        self.m = m
        self.core = max(os.sched_getaffinity(0)) if core is None else core
        self._procs: list[mp.process.BaseProcess] = []
        self._saved: set[int] = set()

    def __enter__(self) -> co_tenants:
        self._saved = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {self.core})
        ctx = mp.get_context("spawn")  # the parent holds ONNX Runtime threads: never fork it
        self._procs = [
            ctx.Process(target=_burn, args=(self.core,), daemon=True) for _ in range(self.m)
        ]
        for p in self._procs:
            p.start()
        if self.m:
            time.sleep(1.0)  # let the tenants start and the scheduler settle into fair shares
        return self

    def __exit__(self, *exc: object) -> None:
        for p in self._procs:
            p.terminate()
            p.join()
        os.sched_setaffinity(0, self._saved)


def fit_load_exponent(loads: list[float], mean_ms: list[float]) -> float:
    """γ in t(ρ) = t(0) / (1 - ρ)**γ by least squares through the origin in log space."""
    x = np.array([-np.log1p(-rho) for rho in loads[1:]])
    y = np.log(np.array(mean_ms[1:]) / mean_ms[0])
    return float(max(0.0, x @ y / (x @ x)))  # a negative fit is timing noise on a trivial encoder


def profile_menu(
    reps: list[BaseRep],
    zoo: HeadZoo,
    windows: np.ndarray,
    repeats: int = 30,
    with_load: bool = True,
    period_ms: float = 0.0,
) -> tuple[list[RepProfile], dict[str, dict[float, dict[str, float]]]]:
    """Measure every representation. Returns profiles and the raw load curves (median / mean ms)."""
    n = len(windows)
    profiles, curves = [], {}
    for rep in reps:
        payloads = [rep.encode(w) for w in windows]
        decoded = [rep.decode_for_head(p) for p in payloads]

        def encode(i: int, rep: BaseRep = rep) -> object:
            return rep.encode(windows[i])

        def decode(i: int, rep: BaseRep = rep, payloads: list[bytes] = payloads) -> object:
            return rep.decode_for_head(payloads[i])

        def head_of(i: int, rep: BaseRep = rep, decoded: list[np.ndarray] = decoded) -> object:
            return zoo.head(rep.rep_id, decoded[i])

        with co_tenants(0):
            enc = time_ms(encode, n, repeats, period_ms=period_ms)
            dec = time_ms(decode, n, repeats, period_ms=period_ms)
            head = time_ms(head_of, n, repeats, period_ms=period_ms)
        curve = {0.0: {"median_ms": float(np.median(enc)), "mean_ms": float(enc.mean())}}
        if with_load:
            for rho, m in LOAD_TENANTS.items():
                if m:
                    with co_tenants(m):
                        # more samples under load: preemption makes single calls bimodal
                        t = time_ms(encode, n, 4 * repeats, period_ms=period_ms)
                    curve[rho] = {"median_ms": float(np.median(t)), "mean_ms": float(t.mean())}
        curves[rep.rep_id] = curve
        loads = sorted(curve)
        gamma = fit_load_exponent(loads, [curve[r]["mean_ms"] for r in loads]) if with_load else 1.0
        med, mad, q95 = summarise(enc)
        size = rep.size_model()
        profiles.append(
            RepProfile(
                rep_id=rep.rep_id,
                size_mean_bytes=size.mean_bytes,
                size_q95_bytes=size.q95_bytes,
                t_enc_ms=med,
                t_enc_mad_ms=mad,
                t_enc_q95_ms=q95,
                t_dec_ms=summarise(dec)[0],
                t_head_ms=summarise(head)[0],
                load_exponent=gamma,
            )
        )
    return profiles, curves


def profile_pipeline(
    reps: list[BaseRep],
    windows: np.ndarray,
    predictor: Predictor,
    select: Callable[[np.ndarray], object],
    period_ms: float,
    repeats: int = 30,
    warmup: int = 5,
) -> tuple[list[tuple[float, float, float]], Overheads]:
    """Timings inside the whole per-window pipeline, one window per `period_ms`.

    A busy node does not run its stages one at a time with idle gaps in between: every window goes
    probe → predictor → selector → encoder back to back, and the next window follows `period_ms`
    after the previous one started. Each stage is timed where it runs. Returns, per representation,
    the encoder's (median, MAD, q95) in ms, and the overhead stages as the median over the
    pipelines of all representations (load exponent left at 0: fitted at one sensor).
    """
    n = len(windows)
    encoders: list[tuple[float, float, float]] = []
    stages = np.empty((len(reps), 3))
    for r, rep in enumerate(reps):
        times = np.empty((warmup + repeats, 4))
        spent_ns = 0.0
        for j in range(warmup + repeats):
            _rest(period_ms, spent_ns)
            window = windows[j % n]
            t0 = time.perf_counter_ns()
            phi = probe(window)
            t1 = time.perf_counter_ns()
            ghat = predictor.predict(phi)
            t2 = time.perf_counter_ns()
            select(ghat)
            t3 = time.perf_counter_ns()
            rep.encode(window)
            t4 = time.perf_counter_ns()
            spent_ns = float(t4 - t0)
            times[j] = (t1 - t0, t2 - t1, t3 - t2, t4 - t3)
        times = times[warmup:] / 1e6
        encoders.append(summarise(times[:, 3]))
        stages[r] = np.median(times[:, :3], axis=0)
    t_probe, t_pred, t_sel = np.median(stages, axis=0)
    return encoders, Overheads(float(t_probe), float(t_pred), float(t_sel), 0.0)


def measure_overheads(
    windows: np.ndarray,
    predictor: Predictor,
    select: Callable[[np.ndarray], object],
    repeats: int = 30,
    period_ms: float = 0.0,
    with_load: bool = True,
) -> Overheads:
    """Per-window cost of the probe, the sufficiency predictor and the selector itself, and how
    the three together slow down under co-tenant load."""
    n = len(windows)
    phis = [probe(w) for w in windows]
    ghats = [predictor.predict(p) for p in phis]

    def timed(fn: Callable[[int], object], k: int = repeats) -> np.ndarray:
        return time_ms(fn, n, k, period_ms=period_ms)

    def whole(i: int) -> object:
        return select(predictor.predict(probe(windows[i])))

    with co_tenants(0):
        t_probe = summarise(timed(lambda i: probe(windows[i])))[0]
        t_pred = summarise(timed(lambda i: predictor.predict(phis[i])))[0]
        t_sel = summarise(timed(lambda i: select(ghats[i])))[0]
        means = [float(timed(whole).mean())]
    gamma = 1.0
    if with_load:
        loads = sorted(LOAD_TENANTS)
        for rho in loads[1:]:
            with co_tenants(LOAD_TENANTS[rho]):
                means.append(float(timed(whole, 4 * repeats).mean()))
        gamma = fit_load_exponent(loads, means)
    return Overheads(t_probe, t_pred, t_sel, gamma)
