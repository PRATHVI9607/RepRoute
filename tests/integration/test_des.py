"""Simulator laws (PRD §12.2 #8, #9): conservation, Little's law, determinism, ladder behaviour."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kai.certify.aci import AdaptiveLambda
from kai.certify.crc import RAW_ONLY
from kai.certify.drift import DriftMonitor
from kai.certify.runtime import Certifier
from kai.contracts.schemas import FLAGS
from kai.link import traces as tr
from kai.link.des import Scenario, simulate
from kai.select.policies import GreedySafe, MpcSafe, Oracle, Static
from toy import REP_IDS, toy_model, toy_workload

MODEL = toy_model()
WORK = toy_workload(np.random.default_rng(3), 600)
CERT = Certifier(lam=0.05)


def outage_trace() -> tr.Trace:
    """1 Mbps, then 3 s at 1 kbps, then 1 Mbps again."""
    bps = np.full(80, 1e6)
    bps[10:40] = 1e3
    return tr.Trace("outage", 0.1, bps)


def test_single_window_latency_matches_the_formula_by_hand() -> None:
    work = toy_workload(np.random.default_rng(0), 1)
    res = simulate(
        Scenario(tr.static(1e6), deadline_ms=100, rtt_ms=20), work, Static(0), CERT, MODEL
    )
    # a static policy runs no probe and no predictor: it is charged the encoder alone
    expected = (
        MODEL.for_policy("none").edge_time_ms(0.0)[0] + 2076 * 8 / 1e6 * 1000 + 10 + 0.01 + 1.0
    )
    assert res.latency_ms[0] == pytest.approx(expected)
    assert not res.late[0]
    assert res.wire_bytes[0] == 2076
    assert res.flag_rate("certified") == 1.0


@settings(max_examples=25)
@given(
    bps=st.sampled_from([2e4, 2e5, 2e6]),
    sensors=st.sampled_from([1, 8]),
    storage=st.sampled_from([np.inf, 20_000.0, 3000.0]),
    mission=st.sampled_from(["fidelity_first", "liveness_first"]),
    p_audit=st.sampled_from([0.0, 0.2]),
    bursty=st.booleans(),
    load=st.sampled_from([0.0, 0.8]),
)
def test_every_byte_is_accounted_for(bps, sensors, storage, mission, p_audit, bursty, load) -> None:
    trace = tr.markov_bursty(np.random.default_rng(1), bps, bps / 50) if bursty else tr.static(bps)
    sc = Scenario(trace, 100, load, sensors, storage_bytes=storage, p_audit=p_audit)
    res = simulate(sc, WORK, GreedySafe(MODEL, mission), Certifier(lam=0.05), MODEL)
    assert abs(res.conservation_error) < 1e-6
    assert res.offered_bytes == pytest.approx(res.wire_bytes.sum())
    assert sum(res.flag_rate(f) for f in FLAGS) == pytest.approx(1.0)
    sent = res.rep >= 0
    assert (res.flag[~sent] >= FLAGS.index("deferred")).all()
    assert not res.disagreed[~sent].any()
    assert np.isfinite(res.latency_ms[~res.late]).all()
    assert (res.latency_ms[~res.late] <= 100).all()


def test_littles_law_holds_on_a_stable_queue() -> None:
    work = toy_workload(np.random.default_rng(5), 4000)
    # 8 sensors of raw = 8 × 2076 B per 64 ms ≈ 2.08 Mbps offered on a bursty ~2.3 Mbps link
    trace = tr.markov_bursty(np.random.default_rng(2), 2.5e6, 3e5, 0.02, 0.2, step_s=0.05)
    res = simulate(Scenario(trace, 1e6, n_sensors=8), work, Static(0), CERT, MODEL)
    assert res.backlog_bytes == 0  # drained: every frame's wait was recorded
    mean_in_queue = res.queue_area / res.duration_s
    arrival_rate = len(res.queue_wait_s) / res.duration_s
    assert mean_in_queue > 0.1  # the queue is genuinely exercised
    assert mean_in_queue == pytest.approx(arrival_rate * np.mean(res.queue_wait_s), rel=1e-6)


def test_same_seed_and_config_give_identical_results() -> None:
    def run():
        trace = tr.markov_bursty(np.random.default_rng(4), 5e5, 2e4)
        sc = Scenario(trace, 100, 0.5, 8, p_audit=0.1, seed=11)
        cert = Certifier(lam=0.05, aci=AdaptiveLambda(np.random.default_rng(0).random(500), 0.05))
        return simulate(sc, WORK, MpcSafe(MODEL, 8.0), cert, MODEL)

    a, b = run(), run()
    for name in (
        "rep",
        "flag",
        "latency_ms",
        "late",
        "edge_ms",
        "wire_bytes",
        "disagreed",
        "audited",
    ):
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
    assert a.offered_bytes == b.offered_bytes


def test_outage_defers_then_forwards_raw_and_small_buffers_drop() -> None:
    sc = Scenario(outage_trace(), deadline_ms=100, n_sensors=1)
    res = simulate(sc, toy_workload(np.random.default_rng(3), 110), GreedySafe(MODEL), CERT, MODEL)
    deferred = res.flag == FLAGS.index("deferred")
    assert deferred.sum() > 10  # nothing certified could meet the deadline during the outage
    assert res.late[deferred].all()  # store-and-forward arrives after the deadline by definition
    assert np.isfinite(res.latency_ms[deferred]).all()  # ...but it does arrive, as raw
    assert not res.disagreed[deferred].any()
    assert res.flag_rate("dropped") == 0
    assert abs(res.conservation_error) < 1e-6

    # a link too slow for anything: every window is deferred, and the buffer holds only two of them
    tiny = Scenario(tr.static(1e3), deadline_ms=100, storage_bytes=5000.0)
    res2 = simulate(
        tiny, toy_workload(np.random.default_rng(3), 110), GreedySafe(MODEL), CERT, MODEL
    )
    assert res2.flag_rate("dropped") > 0.9  # the oldest cached window is dropped for each new one
    assert res2.dropped_bytes > 100 * 2076
    assert abs(res2.conservation_error) < 1e-6

    live = simulate(
        sc,
        toy_workload(np.random.default_rng(3), 110),
        GreedySafe(MODEL, "liveness_first"),
        CERT,
        MODEL,
    )
    assert live.flag_rate("deferred") == 0
    assert live.flag_rate("uncertified") + live.flag_rate("dropped") > 0.1


def test_kai_keeps_the_slo_where_raw_always_collapses() -> None:
    work = toy_workload(np.random.default_rng(6), 3000)
    sc = Scenario(tr.static(1e6), deadline_ms=100, n_sensors=8)  # raw would need 2.08 Mbps
    raw = simulate(sc, work, Static(0), Certifier(lam=0.05), MODEL)
    kai = simulate(sc, work, GreedySafe(MODEL, "liveness_first"), Certifier(lam=0.05), MODEL)
    oracle = simulate(sc, work, Oracle(MODEL, "liveness_first"), Certifier(lam=0.05), MODEL)
    assert raw.late.mean() > 0.9
    assert kai.late.mean() < 0.1
    assert oracle.late.mean() <= kai.late.mean() + 0.02
    assert not oracle.disagreed[oracle.flag == 0].any()  # the oracle never disagrees
    certified = kai.flag == FLAGS.index("certified")
    assert kai.disagreed[certified].mean() < 0.08  # λ = 0.05 on scores with σ = 0.03 noise


def test_audit_ships_raw_copies_and_feeds_aci() -> None:
    work = toy_workload(np.random.default_rng(8), 6000)
    aci = AdaptiveLambda(np.random.default_rng(0).random(2000), alpha=0.05, gamma=0.05)
    sc = Scenario(tr.static(5e6), deadline_ms=200, p_audit=0.2, seed=3)
    res = simulate(sc, work, GreedySafe(MODEL), Certifier(lam=0.0, aci=aci), MODEL)
    non_raw = res.rep > 0
    assert non_raw.sum() > 500
    assert res.audited[non_raw].mean() == pytest.approx(0.2, abs=0.04)
    assert not res.audited[~non_raw].any()  # auditing a raw decision would be pointless
    sizes = MODEL.profiles.size_bytes
    assert np.allclose(res.wire_bytes[res.audited], sizes[res.rep[res.audited]] + 28 + 2076)
    assert aci.alpha_t != 0.05  # the certifier received feedback


def test_drift_monitor_forces_raw_only() -> None:
    rng = np.random.default_rng(9)
    work = toy_workload(rng, 1200)
    work.phi[600:] += 3.0  # the probe distribution shifts half-way through
    cert = Certifier(lam=1.0, drift=DriftMonitor(work.phi[:600].copy(), window=100, stride=20))
    res = simulate(Scenario(tr.static(5e6), deadline_ms=500), work, GreedySafe(MODEL), cert, MODEL)
    assert cert.safe_mode
    assert (res.rep[:500] != 0).mean() > 0.9  # before the shift cheap representations are used
    assert (res.rep[-300:] == 0).all()  # after detection only raw is certified
    assert REP_IDS[0] == "r0_raw"


def test_store_and_forward_does_not_starve_live_traffic() -> None:
    """Cached raw windows wait for slack; only a periodic probe goes out on a saturated link."""
    work = toy_workload(np.random.default_rng(12), 2000)
    # 256 kbit/s with a window every 8 ms: a raw frame needs 65 ms of link, eight arrival gaps
    sc = Scenario(tr.static(256e3), deadline_ms=100, n_sensors=8)
    res = simulate(sc, work, GreedySafe(MODEL), Certifier(lam=0.02), MODEL)
    certified = (res.flag == FLAGS.index("certified")) & (res.rep > 0)
    deferred = res.flag == FLAGS.index("deferred")
    assert certified.mean() > 0.2
    assert deferred.mean() > 0.2
    assert res.late[certified].mean() < 0.05  # live frames are not stuck behind forwarded raw
    forwarded = np.isfinite(res.latency_ms[deferred]).sum()
    assert forwarded <= res.duration_s / sc.probe_s + 2  # at most the periodic probes
    assert abs(res.conservation_error) < 1e-6


def test_a_frame_nearly_off_the_wire_does_not_block_the_next_window() -> None:
    """Regression: counting the whole in-flight frame as backlog deferred every second window.

    Raw needs 55 ms of a 300 kbit/s link and windows arrive every 64 ms, so the link keeps up.
    """
    work = toy_workload(np.random.default_rng(3), 400)
    sc = Scenario(tr.static(300e3), deadline_ms=100, n_sensors=1)
    res = simulate(sc, work, GreedySafe(MODEL), Certifier(lam=RAW_ONLY), MODEL)
    assert res.flag_rate("certified") == 1.0
    assert res.late.mean() == 0.0
