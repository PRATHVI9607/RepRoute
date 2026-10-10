"""Policy containment (PRD §12.2 #3) and selector behaviour."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from kai.select import baselines as bl
from kai.select.base import NO_REP, Decision, Truth, Window, ladder
from kai.select.cost import State
from kai.select.policies import GreedySafe, MpcSafe, Oracle, Static
from toy import PRICES, REP_IDS, toy_model

K = len(REP_IDS)
MODEL = toy_model()

states = st.builds(
    State,
    bandwidth_bps=st.floats(2e3, 5e7),
    rtt_ms=st.floats(0, 200),
    backlog_bytes=st.floats(0, 5e4),
    load=st.sampled_from([0.0, 0.5, 0.8, 0.95]),
    deadline_ms=st.floats(-50, 2000),
    energy_budget_j=st.sampled_from([np.inf, 0.01, 1e-4]),
    storage_free_bytes=st.sampled_from([np.inf, 4096.0, 100.0]),
)


@st.composite
def windows(draw: st.DrawFn) -> tuple[Window, Truth]:
    safe = np.array([True, *draw(st.lists(st.booleans(), min_size=K - 1, max_size=K - 1))])
    ghat = np.array([0.0, *draw(st.lists(st.floats(0, 1), min_size=K - 1, max_size=K - 1))])
    d = np.array([False, *draw(st.lists(st.booleans(), min_size=K - 1, max_size=K - 1))])
    phi = np.array([draw(st.floats(1e-4, 1.0)), 0.0, 0.0, 0.0])
    return Window(safe, ghat, phi), Truth(d, draw(states))


def all_selectors() -> list:
    admissible = np.array([True, True, True, False, False])
    return [
        GreedySafe(MODEL),
        GreedySafe(MODEL, mission="liveness_first"),
        GreedySafe(
            replace(MODEL, profiles=MODEL.profiles.without_encode_cost()), queue_aware=False
        ),
        MpcSafe(MODEL, period_ms=64.0),
        MpcSafe(MODEL, period_ms=2.0, horizon=4, mission="liveness_first"),
        Static(0),
        Static(4),
        bl.ContextBlind(MODEL),
        bl.ProfileTable(MODEL, admissible, deadline_ms=100.0),
        bl.AdaptiveThreshold((2, 1, 0)),
        bl.TwoModeSwitch(MODEL, semantic=4, traditional=1),
        bl.ConformalModelSelection(MODEL, admissible),
    ]


@given(windows(), states)
def test_certified_decisions_are_always_inside_the_safe_set(
    wt: tuple[Window, Truth], state: State
) -> None:
    win, truth = wt
    for sel in all_selectors():
        dec = sel.select(win, state, truth)
        assert isinstance(dec, Decision)
        assert (dec.rep == NO_REP) == (dec.flag in ("deferred", "dropped"))
        if dec.flag == "certified":
            assert win.safe[dec.rep], sel.name
        if dec.rep != NO_REP:
            assert 0 <= dec.rep < K


@given(windows(), states)
def test_kai_selectors_respect_feasibility_and_the_ladder(
    wt: tuple[Window, Truth], state: State
) -> None:
    win, _ = wt
    feasible, cost = MODEL.feasible(state), MODEL.cost(state)
    fast_cost, fast_feasible = MODEL.evaluate(state)  # the selector's single-pass hot path
    assert np.allclose(fast_cost, cost, rtol=1e-12, atol=0, equal_nan=True)
    assert np.array_equal(fast_feasible, feasible)
    for sel in (
        GreedySafe(MODEL),
        MpcSafe(MODEL, 64.0),
        GreedySafe(MODEL, mission="liveness_first"),
    ):
        dec = sel.select(win, state)
        if dec.flag == "certified":
            assert feasible[dec.rep]
        if dec.flag == "uncertified":
            assert sel.mission == "liveness_first"
            assert feasible[dec.rep]
            assert not win.safe[dec.rep]
            assert not (win.safe & feasible).any()
        if dec.flag == "deferred":
            assert sel.mission == "fidelity_first"
            assert not (win.safe & feasible).any()
        if dec.flag == "dropped":
            assert not feasible.any()
    greedy = GreedySafe(MODEL).select(win, state)
    if greedy.flag == "certified":  # greedy is the exact argmin over S ∩ Feas
        pool = np.flatnonzero(win.safe & feasible)
        assert cost[greedy.rep] == cost[pool].min()


@given(windows())
def test_oracle_never_picks_a_disagreeing_representation(wt: tuple[Window, Truth]) -> None:
    win, truth = wt
    dec = Oracle(MODEL).select(win, truth.state, truth)
    if dec.rep != NO_REP and dec.flag == "certified":
        assert not truth.d[dec.rep]
        assert MODEL.feasible(truth.state)[dec.rep]
    with pytest.raises(ValueError, match="ground truth"):
        Oracle(MODEL).select(win, truth.state)


def test_each_policy_is_charged_only_the_work_it_does() -> None:
    kinds = {type(s).__name__: s.overhead for s in all_selectors()}
    assert kinds == {
        "GreedySafe": "full", "MpcSafe": "full", "ContextBlind": "full",  # need the safe set
        "AdaptiveThreshold": "probe",  # reads the probe, runs no predictor
        "ProfileTable": "select", "TwoModeSwitch": "select", "ConformalModelSelection": "select",
        "Static": "none",
    }  # fmt: skip
    assert Oracle(MODEL).overhead == "select"
    ov = MODEL.overheads
    assert MODEL.for_policy("none").overheads.total_ms == 0
    assert MODEL.for_policy("select").overheads.total_ms == ov.t_sel_ms
    assert MODEL.for_policy("probe").overheads.total_ms == ov.t_probe_ms + ov.t_sel_ms
    assert MODEL.for_policy("full").overheads == ov
    # the simulator charges accordingly: same window, same representation, different edge time
    full, none = MODEL.edge_time_ms(0.5), MODEL.for_policy("none").edge_time_ms(0.5)
    assert np.allclose(full - none, ov.total_ms / 0.5)


def test_ladder_steps_in_order() -> None:
    cost = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    safe = np.array([True, True, False, False, False])
    none = np.zeros(K, bool)
    assert ladder(safe, np.ones(K, bool), cost, "fidelity_first") == Decision(1, "certified")
    only_cheap = np.array([False, False, False, True, True])
    assert ladder(safe, only_cheap, cost, "fidelity_first") == Decision(NO_REP, "deferred")
    assert ladder(safe, only_cheap, cost, "liveness_first") == Decision(4, "uncertified")
    assert ladder(safe, none, cost, "liveness_first") == Decision(NO_REP, "dropped")


def test_cost_model_matches_the_prd_formula_by_hand() -> None:
    s = State(bandwidth_bps=1e6, rtt_ms=20, backlog_bytes=1000, load=0.5, deadline_ms=30)
    k = 2  # wavelet: 600 B, enc 1.0, dec 0.6, head 1.0
    edge = (0.05 + 0.02 + 0.01) / 0.5 + 1.0 / 0.5
    latency = edge + (1000 + 600 + 28) * 8 / 1e6 * 1000 + 10 + 0.6 + 1.0
    assert MODEL.edge_time_ms(0.5)[k] == pytest.approx(edge)
    assert MODEL.latency_ms(s)[k] == pytest.approx(latency)
    energy = 2.0 * edge / 1000 + 1e-7 * 628 * 8
    assert MODEL.energy_j(s)[k] == pytest.approx(energy)
    expected = 1.0 * 628 + 0.0 * energy + 5.0 * edge + 1000.0 * max(0.0, latency - 30)
    assert MODEL.cost(s)[k] == pytest.approx(expected)
    assert MODEL.feasible(s)[k] == (latency <= 30)
    assert not MODEL.feasible(replace(s, storage_free_bytes=100.0))[k]
    assert not MODEL.feasible(replace(s, deadline_ms=1e6, energy_budget_j=1e-9)).any()


def test_higher_load_shifts_the_choice_away_from_heavy_encoders() -> None:
    win = Window(np.ones(K, bool), np.zeros(K), np.zeros(4))
    cheap_cpu = PRICES.model_copy(update={"p_cpu_ms": 0.0})
    sel = GreedySafe(toy_model(cheap_cpu))
    idle = sel.select(win, State(2e5, 20, 0, 0.0, 100))
    busy = sel.select(win, State(2e5, 20, 0, 0.99, 100))
    assert REP_IDS[idle.rep] == "latent"  # smallest payload wins when encoding is fast
    assert REP_IDS[busy.rep] == "features"  # the 3 ms encoder now takes 300 ms: far too late


def test_mpc_equals_greedy_without_queue_pressure_and_backs_off_with_it() -> None:
    prices = PRICES.model_copy(
        update={"p_byte": 0.0, "p_cpu_ms": 1.0}
    )  # CPU time and lateness only
    model = toy_model(prices)
    win = Window(np.ones(K, bool), np.zeros(K), np.zeros(4))
    roomy = State(bandwidth_bps=5e7, rtt_ms=10, backlog_bytes=0, load=0.0, deadline_ms=200)
    assert MpcSafe(model, 64.0).select(win, roomy) == GreedySafe(model).select(win, roomy)
    # 300 kbps and a window every 4 ms: the link drains 150 B per period, so raw's 2 kB stays queued
    # and pushes a later window past the deadline. Greedy cannot see that; MPC can.
    tight = State(bandwidth_bps=3e5, rtt_ms=10, backlog_bytes=0, load=0.0, deadline_ms=61.47)
    greedy = GreedySafe(model).select(win, tight)
    mpc = MpcSafe(model, period_ms=4.0, horizon=4).select(win, tight)
    assert REP_IDS[greedy.rep] == "r0_raw"  # myopically the cheapest feasible choice
    assert REP_IDS[mpc.rep] == "features"


def test_offline_baseline_construction() -> None:
    rng = np.random.default_rng(0)
    d = rng.random((2000, K)) < np.array([0.0, 0.0, 0.02, 0.2, 0.4])
    adm = bl.marginally_admissible(d, 0.05)
    assert adm.tolist() == [True, True, True, False, False]
    assert bl.ltt_admissible(d, 0.05, 0.1).tolist() == [True, True, True, False, False]
    assert not bl.ltt_admissible(d[:30], 0.05, 0.1)[1:].any()  # too little data to certify anything
    nominal = State(1e6, 20, 0, 0.0, 100)
    assert REP_IDS[bl.best_single(adm, MODEL, nominal)] == "wavelet"
    # nothing admissible is feasible → still returns an admissible representation
    assert adm[bl.best_single(adm, MODEL, replace(nominal, deadline_ms=0.0))]
    two = bl.TwoModeSwitch(MODEL, semantic=4, traditional=1)
    win = Window(np.ones(K, bool), np.zeros(K), np.zeros(4))
    assert two.select(win, replace(nominal, bandwidth_bps=1e5)).rep == 4
    assert two.select(win, replace(nominal, bandwidth_bps=1e8)).rep == 1


def test_adaptive_threshold_steps_up_on_a_level_change() -> None:
    sel = bl.AdaptiveThreshold((2, 1, 0))
    safe = np.ones(K, bool)

    def at(rms: float) -> int:
        return sel.select(
            Window(safe, np.zeros(K), np.array([rms, 0, 0, 0])), State(1e6, 0, 0, 0, 100)
        ).rep

    assert {at(0.1) for _ in range(50)} == {2}  # steady signal → most compressed
    assert at(5.0) == 0  # sudden jump → highest fidelity
