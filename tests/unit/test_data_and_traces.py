"""Paderborn loader pieces (without the archives) and trace generators."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from scipy.io import savemat

from kai.data import pu
from kai.link import traces as tr


def test_bearing_catalogue_matches_the_published_dataset() -> None:
    assert len(pu.BEARINGS) == 32
    origins = Counter(m.origin for m in pu.BEARINGS.values())
    assert origins == {"healthy": 6, "artificial": 12, "real": 14}
    labelled = [m.label for m in pu.BEARINGS.values() if m.label is not None]
    assert Counter(labelled) == {0: 6, 1: 11, 2: 12}  # healthy / inner race / outer race
    assert all(pu.BEARINGS[b].label is None for b in ("KB23", "KB24", "KB27"))  # combined damage


def test_vibration_channel_is_found_by_name_and_quantised(
    tmp_path, rng: np.random.Generator
) -> None:
    signal = rng.normal(0, 0.5, 10_000)
    channels = np.array(
        [("force", np.zeros(8)), ("vibration_1", signal), ("speed", np.ones(8))],
        dtype=[("Name", "O"), ("Data", "O")],
    )
    path = tmp_path / "N15_M07_F10_K001_1.mat"
    savemat(path, {"N15_M07_F10_K001_1": {"Y": channels, "X": np.zeros(1)}})
    loaded = pu.load_vibration(path)
    assert np.allclose(loaded, signal)

    samples, clipped = pu.quantise(loaded, full_scale=16.0)
    assert samples.dtype == np.int16
    assert clipped == 0
    assert np.abs(samples / 32767 * 16.0 - signal).max() <= 16.0 / 32767  # half an LSB each way
    _, clipped = pu.quantise(loaded, full_scale=0.5)
    assert clipped > 0  # a too-small full scale is reported, never silent

    windows = pu.segment(samples, 4096)
    assert windows.shape == (2, 4096)  # the 1808-sample tail is dropped; windows do not overlap
    assert np.array_equal(windows.ravel(), samples[:8192])

    savemat(tmp_path / "bad.mat", {"rec": {"Y": channels[[0, 2]], "X": np.zeros(1)}})
    with pytest.raises(ValueError, match="vibration_1"):
        pu.load_vibration(tmp_path / "bad.mat")


def test_trace_integration_is_exact_and_cyclic() -> None:
    trace = tr.Trace("t", 1.0, np.array([8000.0, 0.0, 16000.0]))  # 1 kB/s, outage, 2 kB/s
    assert trace.finish_time(0.0, 500) == pytest.approx(0.5)
    assert trace.finish_time(0.5, 1000) == pytest.approx(2.25)  # 0.5 s, wait out the outage, 0.25 s
    # from 2.5 s: 1 kB at 2 kB/s, the cycle wraps, 1 kB at 1 kB/s, outage, last 0.5 kB at 2 kB/s
    assert trace.finish_time(2.5, 2500) == pytest.approx(5.25)
    assert trace.rate(4.2) == 0.0
    with pytest.raises(ValueError, match="non-zero"):
        tr.Trace("dead", 1.0, np.zeros(3))
    # a start time that float division places one segment short must not stall
    assert tr.Trace("s", 0.1, np.full(3, 1e6)).finish_time(0.30000000000000004, 125) > 0.3


def test_generators_are_seeded_and_plausible(tmp_path) -> None:
    for kind, (median, *_rest) in tr.CELLULAR.items():
        a = tr.cellular(np.random.default_rng(1), kind)
        b = tr.cellular(np.random.default_rng(1), kind)
        assert np.array_equal(a.bps, b.bps)
        assert (a.bps > 0).all()
        assert 0.4 * median < np.median(a.bps) < 2.5 * median
    bursty = tr.markov_bursty(np.random.default_rng(2), 1e6, 5e4)
    assert set(np.unique(bursty.bps)) == {5e4, 1e6}
    assert 0.6 < np.mean(bursty.bps == 1e6) < 0.95  # stationary share of the good state is 0.8
    path = tmp_path / "trace.csv"
    path.write_text("1000\n2000\n4000\n")
    loaded = tr.from_csv(path, step_s=0.5, scale=1000.0)
    assert loaded.bps.tolist() == [1e6, 2e6, 4e6]
    assert loaded.name == "trace"


def test_harmonic_mean_estimator_is_conservative() -> None:
    est = tr.HarmonicMeanEstimator(1e6, k=3)
    for sample in (1e6, 1e6, 1e4):
        est.add(sample)
    assert est.bps < 3.1e4  # one slow sample dominates
    assert est.last_bps == 1e4
    for _ in range(3):
        est.add(2e6)
    assert est.bps == pytest.approx(2e6)  # old samples leave the window


def test_measured_traces_are_picked_by_class_and_rescaled_to_the_cell_rate(
    tmp_path, monkeypatch, rng: np.random.Generator
) -> None:
    from kai.bench import evaluate as ev

    folder = tmp_path / "ucc4g"
    folder.mkdir()
    monkeypatch.setattr(ev, "MEASURED_DIR", folder)
    for i in range(3):
        np.savetxt(folder / f"car__log{i}.csv", rng.uniform(1e6, 9e6, 300), fmt="%.0f")
    cell = ev.Cell("ucc4g_car", 256e3, 250.0, 0.0, 1, 0.05)
    trace = ev.make_trace(cell, seed=0)
    assert trace.step_s == 1.0
    assert np.median(trace.bps) == pytest.approx(256e3)  # the shape is kept, the median moved
    assert trace.name.startswith("car__")
    assert ev.make_trace(cell, seed=0).name == trace.name  # the seed picks the log
    with pytest.raises(FileNotFoundError, match="download_traces"):
        ev.make_trace(ev.Cell("ucc4g_train", 256e3, 250.0, 0.0, 1, 0.05), seed=0)


def test_bandwidth_estimate_survives_a_trace_that_starts_in_an_outage() -> None:
    from kai.link.traces import HarmonicMeanEstimator

    est = HarmonicMeanEstimator(0.0)
    assert 0 < est.bps <= 1.0  # tiny but defined: the selector must still be able to plan
    est.add(1e6)
    assert est.bps > 1.0


def test_workload_samples_every_bearing_and_cells_start_at_their_own_trace_point() -> None:
    from kai.bench import evaluate as ev

    idx = np.arange(1000)  # stored bearing by bearing: 0–249 is bearing 0, …
    picked = ev.spread(idx, 100)
    assert len(picked) == 100
    assert np.all(np.diff(picked) > 0)  # in order
    assert len(np.unique(picked // 250)) == 4  # every bearing, not the first one only
    assert len(ev.spread(np.arange(10), 25)) == 25  # a short partition repeats
    a = ev.Cell("markov_bursty", 1e6, 250.0, 0.0, 1, 0.05)
    trace = ev.make_trace(a, seed=0)
    assert np.array_equal(trace.bps, ev.make_trace(a, seed=0).bps)  # deterministic
    same_cell_other_alpha = ev.Cell("markov_bursty", 1e6, 250.0, 0.0, 1, 0.10)
    assert np.array_equal(trace.bps, ev.make_trace(same_cell_other_alpha, seed=0).bps)
    other_cell = ev.Cell("markov_bursty", 1e6, 250.0, 0.0, 32, 0.05)
    assert not np.array_equal(trace.bps, ev.make_trace(other_cell, seed=0).bps)
