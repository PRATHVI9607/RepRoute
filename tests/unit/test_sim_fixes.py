"""ADR 0028 §1 fixes 1-3, 5, 6 (simulator, ladder inputs, metrics)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kai.bench import evaluate as ev
from kai.certify.runtime import Certifier
from kai.link import traces as tr
from kai.link.des import Scenario, simulate
from kai.select.base import Window
from kai.select.cost import State
from kai.select.policies import GreedySafe, Static
from toy import toy_model, toy_workload

MODEL = toy_model()
CERT = Certifier(lam=0.05)


def test_certified_late_defers_when_the_link_cannot_deliver() -> None:
    # 1: a dead link (estimate frozen high, but backlog huge) cannot meet cap·D
    s = State(1e6, 20.0, backlog_bytes=5e6, load=0.0, deadline_ms=100.0)
    assert not MODEL.sendable(s).any()
    assert MODEL.sendable(replace(s, backlog_bytes=0.0)).any()
    win = Window(np.ones(5, bool), np.zeros(5), np.zeros(4))
    assert GreedySafe(MODEL, "certified_late").select(win, s).flag == "deferred"


def test_raw_deferral_charges_the_raw_encode_and_io() -> None:
    bps = np.full(40, 1e3)  # dead from the start: everything defers
    sc = Scenario(tr.Trace("dead", 0.1, bps), deadline_ms=100, n_sensors=1)
    work = toy_workload(np.random.default_rng(1), 20)
    res = simulate(sc, work, GreedySafe(MODEL), CERT, MODEL)
    deferred = res.rep < 0
    assert deferred.any()
    full = MODEL.for_policy("full").edge_time_ms(0.0)[0]
    assert res.edge_ms[deferred] == pytest.approx(full)


def test_bytes_queued_at_the_end_are_charged() -> None:
    sc = Scenario(tr.static(1e4), deadline_ms=100, n_sensors=8)  # far too slow: a long queue
    work = toy_workload(np.random.default_rng(2), 60)
    res = simulate(sc, work, Static(0), CERT, MODEL)
    assert res.backlog_bytes > 0
    assert res.wire_bytes.sum() == pytest.approx(res.delivered_bytes + res.backlog_bytes)


def test_raw_agree_feedback_waits_for_delivery() -> None:
    fed: list[bool] = []

    class Spy(Certifier):
        def feedback(self, disagreed: bool) -> None:
            fed.append(disagreed)

    sc = Scenario(tr.static(1e5), deadline_ms=100, n_sensors=8, p_audit=1.0)
    work = toy_workload(np.random.default_rng(4), 30)
    res = simulate(sc, work, Static(0), Spy(lam=0.05), MODEL)
    delivered = int(res.delivered_bytes / 2076)
    assert 0 < len(fed) <= delivered  # only frames that arrived feed back (was: all 30)
    assert len(fed) < 30


def test_metrics_cost_terms_sum_to_cost_and_caps_are_ordered() -> None:
    cell = ev.Cell("static", 1e6, 100.0, 0.0, 8, 0.1)
    sc = Scenario(tr.static(1e5), deadline_ms=100, n_sensors=8)
    res = simulate(sc, toy_workload(np.random.default_rng(5), 80), Static(0), CERT, MODEL)
    m = ev.metrics(res, cell, MODEL, window_period_ms=64.0)
    parts = ("cost_bytes", "cost_energy_cpu", "cost_energy_link", "cost_cpu", "cost_late")
    assert sum(m[k] for k in parts) == m["cost"]
    assert m["cost_cap5"] <= m["cost"] <= m["cost_cap20"]
    assert m["edge_util"] == pytest.approx(res.edge_ms.mean() / (64.0 / 8))


def test_certified_late_cap_uses_the_nominal_deadline() -> None:
    d = 100.0
    win = Window(np.ones(5, bool), np.zeros(5), np.zeros(4))
    sel = GreedySafe(MODEL, "certified_late", deadline_nominal_ms=d)
    # waited 2·D already (remaining budget -2·D); the link adds ~0.03·D: sendable, certified
    s = State(1e6, 20.0, 0.0, 0.0, deadline_ms=-2 * d)
    assert sel.select(win, s).flag == "certified"
    # waited 2·D and a backlog that adds more than 8·D: over the 10·D cap, so it defers
    s = replace(s, backlog_bytes=1e6 * 8 * d / 1e3 / 8 * 9)
    assert sel.select(win, s).flag == "deferred"
