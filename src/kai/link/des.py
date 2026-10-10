"""Discrete-event simulator of the edge → uplink → cloud path (PRD §5.5), built on SimPy.

Two FIFO stages: the edge CPU (probe + predictor + selector + exactly one encoder per window) and
the uplink (time-varying rate from a trace). Windows the ladder defers are cached and forwarded
when the link is otherwise idle. Every byte is accounted for:
`offered = delivered + backlog + dropped` (checked by `Result.conservation_error`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
import simpy

from kai.certify.aci import AuditSampler
from kai.certify.runtime import Certifier
from kai.contracts.frame import OVERHEAD_BYTES
from kai.contracts.schemas import FLAGS, Flag
from kai.link.traces import HarmonicMeanEstimator, Trace
from kai.select.base import NO_REP, Selector, Truth, Window
from kai.select.cost import CostModel, State


@dataclass(frozen=True)
class Scenario:
    trace: Trace
    deadline_ms: float
    load: float = 0.0  # edge CPU load ρ from other tenants
    n_sensors: int = 1
    window_period_ms: float = 64.0  # one window per sensor every N / fs
    rtt_ms: float = 20.0
    energy_budget_j: float = np.inf
    storage_bytes: float = np.inf  # shared by the uplink queue and the store-and-forward buffer
    p_audit: float = 0.0
    probe_s: float = 1.0  # an idle link forwards one cached window this often to re-measure itself
    seed: int = 0


@dataclass(frozen=True)
class Workload:
    """Per-window inputs in arrival order (rows of the Disagreement Table, test partition)."""

    ghat: np.ndarray  # [n, K]
    phi: np.ndarray  # [n, P]
    d: np.ndarray  # [n, K] bool — never shown to a deployable selector
    payload_bytes: np.ndarray  # [n, K] actual encoded sizes

    def __len__(self) -> int:
        return len(self.ghat)


@dataclass
class Result:
    rep: np.ndarray  # [n] chosen representation, NO_REP if none was encoded
    flag: np.ndarray  # [n] index into FLAGS
    latency_ms: np.ndarray  # [n] arrival → cloud decision; nan if never delivered
    late: np.ndarray  # [n] bool: not delivered within the deadline (includes deferred / dropped)
    edge_ms: np.ndarray  # [n] edge CPU time actually spent
    wire_bytes: np.ndarray  # [n] bytes put on the link for this window (payload + frame + audit)
    disagreed: np.ndarray  # [n] bool: the transmitted decision differs from the raw decision
    audited: np.ndarray  # [n] bool
    offered_bytes: float = 0.0
    delivered_bytes: float = 0.0
    dropped_bytes: float = 0.0
    backlog_bytes: float = 0.0
    queue_wait_s: list[float] = field(default_factory=list)  # per frame: time waiting for the link
    queue_area: float = 0.0  # ∫ (frames waiting) dt
    duration_s: float = 0.0

    @property
    def conservation_error(self) -> float:
        return self.offered_bytes - self.delivered_bytes - self.dropped_bytes - self.backlog_bytes

    def flag_rate(self, name: Flag) -> float:
        return float(np.mean(self.flag == FLAGS.index(name)))


@dataclass
class _Frame:
    window: int
    nbytes: float
    live: bool  # False for audit copies and store-and-forward traffic
    enqueued_s: float = 0.0


def simulate(  # noqa: PLR0915 - three coupled processes sharing queue state: kept in one scope
    scenario: Scenario,
    work: Workload,
    selector: Selector,
    certifier: Certifier,
    truth_model: CostModel,
) -> Result:
    """Run every window of `work` through the edge and the link. `truth_model` charges real costs;
    the selector may plan with a different (ablated) model of its own."""
    n = len(work)
    env = simpy.Environment()
    est = HarmonicMeanEstimator(scenario.trace.rate(0.0))
    audit = AuditSampler(scenario.p_audit, np.random.default_rng([scenario.seed, 7]))
    res = Result(
        rep=np.full(n, NO_REP),
        flag=np.zeros(n, np.int8),
        latency_ms=np.full(n, np.nan),
        late=np.ones(n, bool),
        edge_ms=np.zeros(n),
        wire_bytes=np.zeros(n),
        disagreed=np.zeros(n, bool),
        audited=np.zeros(n, bool),
    )
    arrived = np.zeros(n)
    cpu_queue: simpy.Store = simpy.Store(env)
    live: deque[_Frame] = deque()
    deferred: deque[_Frame] = deque()
    wake = env.event()
    backlog = 0.0  # bytes queued or in flight on the uplink
    deferred_bytes = 0.0
    in_flight = 0.0  # bytes of the frame currently on the wire
    forwarding = 0.0  # ... of which store-and-forward traffic (not part of `backlog`)
    last_tx_end = 0.0
    tx_start = 0.0
    gap_s = scenario.window_period_ms / scenario.n_sensors / 1000.0
    waiting = 0  # frames queued behind the one in flight
    last_change = 0.0
    # every policy is charged only the per-window work it really does (a static one runs no probe)
    actual = truth_model.for_policy(selector.overhead)
    overhead_ms = (
        actual.overheads.total_ms / (1.0 - scenario.load) ** actual.overheads.load_exponent
    )
    edge_ms = actual.edge_time_ms(scenario.load)
    cloud_ms = truth_model.profiles.t_dec_ms + truth_model.profiles.t_head_ms
    raw_wire = float(truth_model.profiles.size_bytes[0]) + OVERHEAD_BYTES

    def track_queue(delta: int) -> None:
        nonlocal waiting, last_change
        res.queue_area += waiting * (env.now - last_change)
        waiting, last_change = waiting + delta, env.now

    def enqueue(queue: deque[_Frame], frame: _Frame) -> None:
        nonlocal wake
        frame.enqueued_s = env.now
        queue.append(frame)
        res.offered_bytes += frame.nbytes
        res.wire_bytes[frame.window] += frame.nbytes
        track_queue(+1)
        if not wake.triggered:
            wake.succeed()

    def link_recovered() -> bool:
        """Forward cached raw only if it fits between two arrivals, or as a periodic probe."""
        fits = raw_wire * 8.0 / est.last_bps <= gap_s
        return fits or env.now - last_tx_end >= scenario.probe_s

    def backlog_seen() -> float:
        """Bytes still to leave the edge, as the edge can know them: what is queued, plus the part
        of the frame on the wire that the estimated rate has not carried away yet."""
        progress = min(in_flight, est.bps * (env.now - tx_start) / 8.0)
        return backlog + forwarding - progress

    def source():
        for i in range(n):
            arrived[i] = env.now
            yield cpu_queue.put(i)
            yield env.timeout(gap_s)

    def cpu():
        nonlocal backlog, deferred_bytes
        for _ in range(n):
            i = yield cpu_queue.get()
            state = State(
                bandwidth_bps=est.bps,
                rtt_ms=scenario.rtt_ms,
                backlog_bytes=backlog_seen(),
                load=scenario.load,
                deadline_ms=scenario.deadline_ms - (env.now - arrived[i]) * 1000.0,
                energy_budget_j=scenario.energy_budget_j,
                storage_free_bytes=scenario.storage_bytes - backlog - deferred_bytes,
            )
            ghat = work.ghat[i]
            win = Window(certifier.safe_set(ghat, work.phi[i]), ghat, work.phi[i])
            truth = Truth(work.d[i], replace(state, bandwidth_bps=scenario.trace.rate(env.now)))
            dec = selector.select(win, state, truth)
            res.flag[i] = FLAGS.index(dec.flag)
            res.rep[i] = dec.rep
            res.edge_ms[i] = edge_ms[dec.rep] if dec.rep != NO_REP else overhead_ms
            yield env.timeout(res.edge_ms[i] / 1000.0)
            if dec.rep != NO_REP:
                res.disagreed[i] = bool(work.d[i, dec.rep])
                nbytes = float(work.payload_bytes[i, dec.rep]) + OVERHEAD_BYTES
                backlog += nbytes
                enqueue(live, _Frame(i, nbytes, live=True))
                if dec.rep != 0 and audit.should_audit():  # also ship raw so the cloud can compare
                    res.audited[i] = True
                    certifier.feedback(bool(work.d[i, dec.rep]))
                    backlog += raw_wire
                    enqueue(live, _Frame(i, raw_wire, live=False))
            elif dec.flag == "deferred":
                while deferred and deferred_bytes + raw_wire > scenario.storage_bytes - backlog:
                    old = deferred.popleft()  # buffer full: drop the oldest cached window
                    deferred_bytes -= old.nbytes
                    res.dropped_bytes += old.nbytes
                    res.flag[old.window] = FLAGS.index("dropped")
                    track_queue(-1)
                if deferred_bytes + raw_wire <= scenario.storage_bytes - backlog:
                    deferred_bytes += raw_wire
                    enqueue(deferred, _Frame(i, raw_wire, live=False))
                else:
                    res.flag[i] = FLAGS.index("dropped")

    def link():
        nonlocal backlog, deferred_bytes, wake, in_flight, forwarding, last_tx_end, tx_start
        while True:
            if not live and not (deferred and link_recovered()):
                wake = env.event()
                # with cached windows waiting, wake by the next probe time even if nothing arrives
                yield (wake | env.timeout(scenario.probe_s)) if deferred else wake
                continue
            from_live = bool(live)
            frame = live.popleft() if from_live else deferred.popleft()
            track_queue(-1)
            res.queue_wait_s.append(env.now - frame.enqueued_s)
            start, in_flight = env.now, frame.nbytes
            tx_start = start
            forwarding = 0.0 if from_live else frame.nbytes
            yield env.timeout(scenario.trace.finish_time(start, frame.nbytes) - start)
            in_flight = forwarding = 0.0
            last_tx_end = env.now
            if from_live:
                backlog -= frame.nbytes
            else:
                deferred_bytes -= frame.nbytes
            res.delivered_bytes += frame.nbytes
            est.add(frame.nbytes * 8.0 / max(env.now - start, 1e-9))
            i = frame.window
            if frame.live or res.flag[i] == FLAGS.index("deferred"):
                rep = res.rep[i] if frame.live else 0  # store-and-forward delivers raw
                done = env.now + (scenario.rtt_ms / 2.0 + cloud_ms[rep]) / 1000.0
                res.latency_ms[i] = (done - arrived[i]) * 1000.0
                res.late[i] = res.latency_ms[i] > scenario.deadline_ms

    env.process(source())
    done = env.process(cpu())
    env.process(link())
    env.run(until=done)
    env.run(until=env.now + scenario.deadline_ms / 1000.0)  # grace: let in-deadline frames land
    track_queue(0)
    res.backlog_bytes = in_flight + sum(f.nbytes for f in live) + sum(f.nbytes for f in deferred)
    res.duration_s = float(env.now)
    return res
