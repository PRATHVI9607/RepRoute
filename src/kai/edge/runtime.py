"""Edge runtime loop (PRD §5.1): probe → predict → safe set → select → encode ONE → send.

numpy + onnxruntime only. Latency is measured on the edge clock: a window's end-to-end latency is
`ack received - window arrival - RTT/2` (the reply's return trip is not part of the uplink path).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np
import psutil

from kai.certify.aci import AdaptiveLambda, AuditSampler
from kai.certify.drift import DriftMonitor
from kai.certify.probe import probe
from kai.certify.runtime import Certifier
from kai.contracts import frame as fr
from kai.contracts.representation import REP_CODES
from kai.contracts.schemas import EnergyModel, Prices, Variant
from kai.edge.bundle import Bundle
from kai.link.framing import PING_CODE, read_reply
from kai.link.traces import HarmonicMeanEstimator
from kai.select.base import NO_REP, Decision, Mission, Selector, Window
from kai.select.cost import CostModel, Profiles, State
from kai.select.policies import CascadeSafe, GreedySafe, MpcSafe

SELECTOR_NAMES = ("s1_greedy_safe", "s2_mpc_safe", "s1_cascade")
SPIN_S = 2e-3  # the last stretch of a pacing wait yields instead of trusting a ~1 ms timer
logger = logging.getLogger(__name__)


def _now_s() -> float:
    return time.perf_counter_ns() / 1e9


async def _wait_until(t_s: float) -> None:
    """Sleep until `t_s` (perf-counter seconds) without blocking the event loop.

    asyncio timers fire up to ~1 ms late, so the bulk is a timer and the final `SPIN_S` is a
    yielding spin: the reply reader keeps running and the wait stays sub-millisecond precise.
    """
    while (left_s := t_s - _now_s()) > 0:
        await asyncio.sleep(left_s - SPIN_S if left_s > SPIN_S else 0)


def _validate_selector(selector: str) -> None:
    """Fail closed: a typo must not silently run a different policy."""
    if selector not in SELECTOR_NAMES:
        expected = ", ".join(repr(name) for name in SELECTOR_NAMES)
        raise ValueError(f"unknown selector {selector!r}; expected one of {expected}")


@dataclass(frozen=True)
class EdgeConfig:
    alpha: float = 0.05
    variant: Variant = "v0"
    selector: str = "s1_greedy_safe"  # one of SELECTOR_NAMES
    mission: Mission = "fidelity_first"
    deadline_ms: float = 250.0
    late_cap_factor: float = 10.0  # certified_late sends only if deliverable within cap·D
    period_ms: float = 64.0
    load: float | None = None  # None → measure with psutil
    p_audit: float = 0.0
    use_aci: bool = False
    use_drift: bool = False
    shape_bps: float | None = None  # app-level pacing for loopback runs without tc/netem
    initial_bps: float = 1e6
    defer_buffer_windows: int = 256  # store-and-forward capacity (ladder step 2)
    probe_s: float = 1.0  # an idle link forwards one cached window this often to re-measure itself
    seed: int = 0

    def __post_init__(self) -> None:
        _validate_selector(self.selector)
        if self.selector == "s1_cascade" and (self.use_aci or self.use_drift):
            raise ValueError("the cascade certificate supports neither ACI nor the drift monitor")


MIN_SAMPLE_BYTES = 1000  # smaller frames cannot measure throughput (see `Edge._acks`)


class NeedsBundleError(RuntimeError):
    """The cloud holds no head for a contract hash this edge is sending."""


def build_certifier(bundle: Bundle, cfg: EdgeConfig) -> Certifier:
    _validate_selector(cfg.selector)
    drift = DriftMonitor(bundle.phi_ref) if cfg.use_drift else None
    if cfg.selector == "s1_cascade":
        if bundle.cascade is None:
            raise ValueError("this bundle predates the cascade (ADR 0011): rebuild it")
        if cfg.alpha not in bundle.cascade:
            raise ValueError(
                f"the bundle's cascade is calibrated for α in {sorted(bundle.cascade)}"
            )
        # ACI re-reads thresholds calibrated on the plain scores; drift needs the probe on every
        # window, which the cascade skips
        if cfg.use_aci or cfg.use_drift:
            raise ValueError("the cascade certificate supports neither ACI nor the drift monitor")
        prior, cascade_lam = bundle.cascade[cfg.alpha]
        return Certifier(cascade_lam, prior=prior)
    entry = bundle.lambdas.get(cfg.variant, cfg.alpha)
    lam = (
        np.array([entry.lam_per_rep[r] for r in bundle.manifest.rep_ids])
        if entry.lam_per_rep is not None
        else entry.lam
    )
    assert lam is not None
    aci = AdaptiveLambda(bundle.calib_m, cfg.alpha) if cfg.use_aci else None
    return Certifier(lam, aci, drift)


def build_selector(
    bundle: Bundle, cfg: EdgeConfig, prices: Prices, energy: EnergyModel
) -> Selector:
    _validate_selector(cfg.selector)
    plan = CostModel(Profiles.from_list(bundle.profiles), bundle.overheads, energy, prices)
    if cfg.selector == "s1_greedy_safe":
        return GreedySafe(
            plan, cfg.mission, deadline_nominal_ms=cfg.deadline_ms,
            late_cap_factor=cfg.late_cap_factor,
        )  # fmt: skip
    if cfg.selector == "s2_mpc_safe":
        return MpcSafe(plan, cfg.period_ms, mission=cfg.mission)
    if cfg.selector == "s1_cascade":
        return CascadeSafe(plan, cfg.mission)
    raise ValueError(f"selector {cfg.selector!r} has no implementation")


class Edge:
    def __init__(
        self,
        bundle: Bundle,
        cfg: EdgeConfig,
        selector: Selector,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.bundle, self.cfg, self.selector = bundle, cfg, selector
        self.reader, self.writer = reader, writer
        self.certifier = build_certifier(bundle, cfg)
        self.audit = AuditSampler(cfg.p_audit, np.random.default_rng([cfg.seed, 7]))
        self.est = HarmonicMeanEstimator(cfg.initial_bps)
        self.rtt_ms = 0.0
        self.backlog = 0.0  # bytes sent and not yet acknowledged
        self._in_flight: deque[tuple[int, int, float, int]] = (
            deque()
        )  # (window, rep_code, t_send, bytes)
        self._last_ack = _now_s()  # no ack yet: the probe clock starts at construction
        self._pace_free_at = 0.0
        # shaped frames wait for the pacer in a sender task, not in the edge loop (ADR 0028 §13)
        self._tx: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue()
        self._deferred: deque[tuple[int, np.ndarray]] = deque()
        self.records: dict[int, dict] = {}
        self.needs_bundle = False
        self._free = np.zeros(0, dtype=bool)  # the cascade's free set of the current window

    # --- transport --------------------------------------------------------------------------
    async def _send(self, window_id: int, rep_code: int, chash: bytes, payload: bytes) -> float:
        """Queue (shaped) or write (unshaped) one frame; returns its token-bucket delay.

        With `shape_bps` the frame's slot on the emulated uplink is reserved here, but the wait
        for it happens in `_sender`, so edge compute and transmission overlap as in kai.link.des.
        The returned delay is informational (`pace_ms`); the caller does not wait for it.
        The frame is stamped when its serialization starts, not when it is released: goodput
        samples and `_backlog_seen` measure from that moment (ADR 0013, addendum). `_in_flight`
        and `backlog` count it from the enqueue, in the order the sender writes.
        """
        buf = fr.pack(fr.Frame(rep_code, chash, window_id, time.time_ns(), payload))
        pace_s = 0.0
        t_start = _now_s()
        if self.cfg.shape_bps:  # token-bucket pacing: emulate a rate-limited uplink in-process
            now = t_start
            t_start = max(now, self._pace_free_at)
            self._pace_free_at = t_start + len(buf) * 8.0 / self.cfg.shape_bps
            pace_s = max(0.0, self._pace_free_at - now)
            self._tx.put_nowait((self._pace_free_at, buf))
        else:
            self.writer.write(buf)
        self.backlog += len(buf)
        self._in_flight.append((window_id, rep_code, t_start, len(buf)))
        return pace_s

    async def _sender(self) -> None:
        """Release queued frames when their serialization time has elapsed (`_wait_until`, not a
        plain asyncio timer: its ~1 ms tick would add an offset the DES does not model)."""
        while True:
            release_s, buf = await self._tx.get()
            await _wait_until(release_s)
            self.writer.write(buf)
            self._tx.task_done()

    async def measure_rtt(self, samples: int = 5) -> float:
        rtts = []
        for _ in range(samples + 1):  # the first ping warms the connection and is discarded
            t0 = time.perf_counter_ns()
            self.writer.write(fr.pack(fr.Frame(PING_CODE, b"\0\0\0\0", 0, 0, b"")))
            await self.writer.drain()
            await read_reply(self.reader)
            rtts.append((time.perf_counter_ns() - t0) / 1e6)
        self.rtt_ms = float(min(rtts[1:]))
        return self.rtt_ms

    async def _acks(self) -> None:
        """Consume cloud replies: latency, goodput samples, audit feedback."""
        while True:
            try:
                reply = await read_reply(self.reader)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                return
            now = _now_s()
            window_id, rep_code, t_send, nbytes = self._in_flight.popleft()
            self.backlog -= nbytes
            # Goodput sample. A frame sent into an idle link is acknowledged one round trip after
            # its last byte left, so the round trip is removed; a queued frame is acknowledged one
            # transmission time after its predecessor. Tiny frames finish in less than the timing
            # noise and would only measure the delay, so they do not produce a sample.
            if t_send >= self._last_ack:
                # the ack also waited for the cloud's decode + head: not link time (ADR 0019)
                elapsed = now - t_send - self.rtt_ms / 1000.0 - reply.cloud_ms / 1000.0
            else:
                elapsed = now - self._last_ack
            if nbytes >= MIN_SAMPLE_BYTES and elapsed > 1e-4:
                self.est.add(nbytes * 8.0 / elapsed)
            self._last_ack = now
            if reply.status == "NEEDS_BUNDLE":
                self.needs_bundle = True
                continue
            rec = self.records[window_id]
            if reply.status != "OK":  # the cloud could not decode it: no decision, no latency
                rec["cloud_error"] = reply.status
                continue
            if rep_code == rec["rep_code"]:  # the live frame (not the audit copy)
                rec["latency_ms"] = (now - rec["t_arrive"]) * 1000.0 - self.rtt_ms / 2.0
                rec["decision"] = reply.decision
                rec["cloud_ms"] = reply.cloud_ms
                if rec.pop("agree_pending", False):  # raw agrees with itself, once acked (des)
                    self.certifier.feedback(False)
            if reply.disagreed >= 0:
                rec["disagreed"] = reply.disagreed
                # the certificate covers certified decisions only (kai.link.des)
                if rec["flag"] == "certified":
                    self.certifier.feedback(bool(reply.disagreed))

    # --- the loop ---------------------------------------------------------------------------
    def _link_recovered(self, raw_wire_bytes: float) -> bool:
        """Forward cached raw only if it fits between two arrivals, or as a periodic probe."""
        fits = raw_wire_bytes * 8000.0 / self.est.last_bps <= self.cfg.period_ms
        return fits or _now_s() - self._last_ack >= self.cfg.probe_s

    def _backlog_seen(self) -> float:
        """Unacknowledged bytes minus the estimated progress of the frame currently on the wire."""
        if not self._in_flight:
            return 0.0
        _, _, t_send, nbytes = self._in_flight[0]
        elapsed = _now_s() - max(t_send, self._last_ack)
        return self.backlog - min(nbytes, self.est.bps * max(elapsed, 0.0) / 8.0)

    def _load(self) -> float:
        if self.cfg.load is not None:
            return self.cfg.load
        # the core the edge runs on: an all-core mean hides it, and the first call returns 0.0
        core = psutil.Process().cpu_num()
        return min(psutil.cpu_percent(None, percpu=True)[core] / 100.0, 0.95)

    async def run(self, windows: np.ndarray, window_ids: np.ndarray) -> list[dict]:  # noqa: PLR0915
        psutil.cpu_percent(None, percpu=True)  # prime: the first reading would be 0.0
        await self.measure_rtt()
        acks = asyncio.create_task(self._acks())
        sender = asyncio.create_task(self._sender())
        reps = self.bundle.reps
        chash = [bytes.fromhex(r.contract_hash()) for r in reps]

        async def send_rep(wid: int, k: int, x: np.ndarray, live: bool) -> None:
            payload = reps[k].encode(x)
            rec = self.records[wid]
            if live:
                rec["rep_code"] = REP_CODES[reps[k].rep_id]
            rec["bytes"] += len(payload) + fr.OVERHEAD_BYTES
            rec["pace_ms"] += (
                await self._send(wid, REP_CODES[reps[k].rep_id], chash[k], payload) * 1e3
            )

        raw_wire = reps[0].size_model().q95_bytes + fr.OVERHEAD_BYTES
        period_s = self.cfg.period_ms / 1000.0
        t_start = _now_s()
        self._last_ack = t_start
        try:
            for n, (x, wid) in enumerate(zip(windows, window_ids.tolist(), strict=True)):
                t_arrive = t_start + n * period_s  # scheduled arrival: the loop may wake late
                await asyncio.sleep(max(0.0, t_arrive - _now_s()))
                if self.needs_bundle:
                    break
                if self._deferred and self.backlog == 0 and self._link_recovered(raw_wire):
                    old_wid, old_x = self._deferred.popleft()
                    await send_rep(old_wid, 0, old_x, live=True)
                t_work = _now_s()
                state = State(
                    bandwidth_bps=self.est.bps,
                    rtt_ms=self.rtt_ms,
                    backlog_bytes=self._backlog_seen(),
                    load=self._load(),
                    # time already spent since arrival is gone from the budget (kai.link.des)
                    deadline_ms=self.cfg.deadline_ms - (t_work - t_arrive) * 1000.0,
                )
                dec, safe = self._decide(x, state)
                decide_s = _now_s() - t_work
                rec = {
                    "window_id": wid, "t_arrive": t_arrive, "rep": dec.rep, "flag": dec.flag,
                    "rep_code": -1, "bytes": 0, "pace_ms": 0.0, "audited": False,
                    "bandwidth_bps": state.bandwidth_bps,
                    "backlog_bytes": state.backlog_bytes, "safe": safe.tolist(),
                    "probed": safe is not self._free,
                }  # fmt: skip
                self.records[wid] = rec
                t_send = _now_s()  # bookkeeping above is not edge compute: time decide + send only
                if dec.rep != NO_REP:
                    # select-then-encode: one encoder runs
                    await send_rep(wid, dec.rep, x, live=True)
                    if self.audit.should_audit():
                        if dec.rep != 0:
                            rec["audited"] = True
                            await send_rep(wid, 0, x, live=False)
                        elif dec.flag == "certified":
                            # raw agrees with itself: fed when its ack arrives (kai.link.des)
                            rec["agree_pending"] = True
                elif dec.flag == "deferred":
                    if len(self._deferred) >= self.cfg.defer_buffer_windows:
                        dropped_wid, _ = self._deferred.popleft()  # buffer full: drop the oldest
                        self.records[dropped_wid]["flag"] = "dropped"
                    self._deferred.append((wid, x))
                # shortcut: encoding runs inside the event loop, so it can delay the sender by at
                # most one encode; move it to a thread if encodes grow past ~1 ms
                rec["edge_ms"] = (
                    decide_s + _now_s() - t_send
                ) * 1000.0  # nothing waits for the pacer
            await self._flush_tx(sender)
            await self._drain(acks)
        finally:
            sender.cancel()
            await self._close(acks)
        return [self.records[w] for w in window_ids.tolist() if w in self.records]

    def _decide(self, x: np.ndarray, state: State) -> tuple[Decision, np.ndarray]:
        """Select-then-encode. A cascade serves the window from its free set when it can, and
        only then pays for the probe and the predictor (ADR 0011)."""
        pre = getattr(self.selector, "select_without_scores", None)
        if pre is not None:
            self._free = self.certifier.free_set(len(self.bundle.reps))
            dec = pre(self._free, state)
            if dec is not None:
                return dec, self._free
        phi = probe(x)
        ghat = self.bundle.predictor.predict(phi)
        safe = self.certifier.safe_set(ghat, phi)
        return self.selector.select(Window(safe, ghat, phi), state), safe

    async def _flush_tx(self, sender: asyncio.Task[None]) -> None:
        """Wait for shaped frames still queued; a dead sender must not hang the join."""
        joined = asyncio.create_task(self._tx.join())
        await asyncio.wait({joined, sender}, return_when=asyncio.FIRST_COMPLETED)
        if sender.done():
            joined.cancel()
            sender.result()  # re-raise what the sender hit

    async def _drain(self, acks: asyncio.Task[None]) -> None:
        """Wait for outstanding replies; raise what the reply reader hit."""
        deadline = _now_s() + 5.0 + self.cfg.deadline_ms / 1000.0
        while self._in_flight and not acks.done() and not self.needs_bundle:
            if _now_s() >= deadline:
                break
            await asyncio.sleep(0.01)
        if acks.done():
            acks.result()  # the reply reader died: raise its error instead of losing it
        if self.needs_bundle:
            raise NeedsBundleError("cloud replied NEEDS_BUNDLE: edge and cloud bundles differ")
        if self._in_flight:
            why = "the cloud closed the stream" if acks.done() else "the drain timed out"
            logger.warning("%d frames left unacknowledged: %s", len(self._in_flight), why)

    async def _close(self, acks: asyncio.Task[None]) -> None:
        acks.cancel()
        self.writer.close()
        with contextlib.suppress(ConnectionError):
            await self.writer.wait_closed()


async def run_edge(
    bundle: Bundle,
    cfg: EdgeConfig,
    prices: Prices,
    energy: EnergyModel,
    windows: np.ndarray,
    window_ids: np.ndarray,
    host: str,
    port: int,
    telemetry: Path | None = None,
    selector: Selector | None = None,
) -> list[dict]:
    for attempt in range(60):  # the cloud may still be loading its models when the edge starts
        try:
            reader, writer = await asyncio.open_connection(host, port)
            break
        except OSError:
            if attempt == 59:
                raise
            await asyncio.sleep(0.5)
    edge = Edge(
        bundle, cfg, selector or build_selector(bundle, cfg, prices, energy), reader, writer
    )
    records = await edge.run(windows, window_ids)
    if telemetry is not None:
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        telemetry.write_bytes(msgpack.packb(records))
    return records
