"""Timing harness: duty-cycled measurement, load-exponent fit, co-tenant pinning."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from kai.analysis import phase
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
