"""Timing harness: duty-cycled measurement, load-exponent fit, co-tenant pinning."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from kai.analysis import phase
from kai.contracts.schemas import RepProfile
from kai.profile import profiler as pf


def test_time_ms_measures_only_the_call_and_can_idle_between_calls() -> None:
    calls = []

    def work(i: int) -> None:
        calls.append(i)
        time.sleep(0.002)

    t0 = time.perf_counter()
    samples = pf.time_ms(work, n_items=3, repeats=6, warmup=2, period_ms=10.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert calls == [0, 1, 0, 1, 2, 0, 1, 2]  # warm-up calls first, then the items cyclically
    assert samples.shape == (6,)
    assert 1.5 < np.median(samples) < 8.0  # the 10 ms idle gap is not part of the measurement
    assert elapsed_ms > 6 * 10.0  # ...but it did happen, once per timed call
    med, mad, q95 = pf.summarise(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert (med, mad) == (3.0, 1.0)
    assert q95 > 4.0


def test_load_exponent_fit() -> None:
    loads = [0.0, 0.5, 0.8]
    for gamma in (0.0, 0.5, 1.0, 1.4):
        means = [2.0 / (1 - rho) ** gamma for rho in loads]
        assert pf.fit_load_exponent(loads, means) == pytest.approx(gamma, abs=1e-9)
    # an encoder that got *faster* under load (a clocked-up core) is clamped, not extrapolated
    assert pf.fit_load_exponent(loads, [2.0, 1.2, 1.5]) == 0.0


def test_co_tenants_pin_and_restore_affinity() -> None:
    before = os.sched_getaffinity(0)
    with pf.co_tenants(1) as ctx:
        assert os.sched_getaffinity(0) == {ctx.core}
        assert len(ctx._procs) == 1
        assert ctx._procs[0].is_alive()
    assert os.sched_getaffinity(0) == before
    assert not ctx._procs[0].is_alive()


def test_crossings_skip_bandwidths_where_nothing_was_measured() -> None:
    grid = np.array([1e5, 1e6, 1e7, 1e8])
    lat = np.array([[np.nan, np.nan], [10.0, 20.0], [10.0, 9.0], [10.0, 5.0]])
    found = phase.crossings(grid, lat)
    assert [(b.low, b.high) for b in found] == [(0, 1)]
    assert 1e6 < found[0].bps < 1e7
    assert phase.crossings(grid, np.full((4, 2), np.nan)) == []


def test_one_call_starts_per_period_and_a_long_call_leaves_no_idle_time() -> None:
    def work(_: int) -> None:
        time.sleep(0.005)

    t0 = time.perf_counter()
    pf.time_ms(work, 1, repeats=6, warmup=0, period_ms=20.0)
    paced = time.perf_counter() - t0
    assert 0.10 < paced < 0.16  # the first call waits a full period, then one start every 20 ms
    t0 = time.perf_counter()
    pf.time_ms(work, 1, repeats=6, warmup=0, period_ms=2.0)
    assert time.perf_counter() - t0 < 0.06  # 5 ms of work in a 2 ms period: a tight loop


def test_profiles_are_looked_up_by_the_duty_cycle_of_the_node(tmp_path) -> None:
    import json

    from kai.bench.experiments import load_profiles

    def entry(t_enc: float) -> dict:
        prof = RepProfile(
            rep_id="r0_raw", size_mean_bytes=8, size_q95_bytes=8, t_enc_ms=t_enc, t_enc_mad_ms=0.0,
            t_enc_q95_ms=t_enc, t_dec_ms=0.1, t_head_ms=0.1, load_exponent=0.0,
        )  # fmt: skip
        return {"profiles": [prof.model_dump()], "overheads": {"t_probe_ms": t_enc}}

    path = tmp_path / "local.json"
    path.write_text(json.dumps({**entry(4.0), "busy": {"8": entry(2.0), "32": entry(1.0)}}))
    assert [load_profiles(path, n_sensors=n)[0][0].t_enc_ms for n in (1, 4, 8, 16, 32, 64)] == [
        4.0, 4.0, 2.0, 2.0, 1.0, 1.0,
    ]  # fmt: skip
    assert load_profiles(path, edge_slowdown=8.0, n_sensors=32)[1].t_probe_ms == 8.0
    path.write_text(json.dumps(entry(4.0)))  # a profile written before duty cycles were measured
    assert load_profiles(path, n_sensors=32)[0][0].t_enc_ms == 4.0


def test_pipeline_profile_times_each_stage_where_it_runs() -> None:
    from kai.certify.fit import fit_predictor
    from kai.certify.probe import probe
    from kai.reps.registry import classical_menu

    rng = np.random.default_rng(0)
    windows = rng.integers(-2000, 2000, size=(8, 1024)).astype(np.int16)
    phi = np.stack([probe(w) for w in windows])
    d = np.zeros((8, 3), bool)
    d[::2, 2] = True
    predictor = fit_predictor(phi, d, ("r0_raw", "a", "b"), kind="logistic")
    reps = classical_menu(1024)[:2]  # raw and lossless
    calls = []
    encoders, overheads = pf.profile_pipeline(
        reps,
        windows,
        predictor,
        calls.append,
        period_ms=1.0,
        repeats=6,
        warmup=2,
    )
    assert len(encoders) == len(reps)
    assert len(calls) == len(reps) * 8  # the selector ran once per window of every pipeline
    assert all(0 < med <= q95 for med, _, q95 in encoders)
    assert encoders[0][0] < encoders[1][0]  # raw is a copy, lossless compresses
    assert overheads.t_probe_ms > 0
    assert overheads.t_pred_ms > 0
    assert overheads.total_ms < 20
