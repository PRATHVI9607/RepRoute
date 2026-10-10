"""Edge runtime loop (PRD §5.1): probe → predict → safe set → select → encode ONE → send.

numpy + onnxruntime only. Latency is measured on the edge clock: a window's end-to-end latency is
`ack received - window arrival - RTT/2` (the reply's return trip is not part of the uplink path).
"""

from __future__ import annotations

import asyncio
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
from kai.select.base import NO_REP, Mission, Selector, Window
from kai.select.cost import CostModel, Profiles, State
from kai.select.policies import GreedySafe, MpcSafe


@dataclass(frozen=True)
class EdgeConfig:
    alpha: float = 0.05
    variant: Variant = "v0"
    selector: str = "s1_greedy_safe"  # or "s2_mpc_safe"
    mission: Mission = "fidelity_first"
    deadline_ms: float = 250.0
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


MIN_SAMPLE_BYTES = 1000  # smaller frames cannot measure throughput (see `Edge._acks`)


class NeedsBundleError(RuntimeError):
    """The cloud holds no head for a contract hash this edge is sending."""


def build_certifier(bundle: Bundle, cfg: EdgeConfig) -> Certifier:
    entry = bundle.lambdas.get(cfg.variant, cfg.alpha)
    lam = (
        np.array([entry.lam_per_rep[r] for r in bundle.manifest.rep_ids])
        if entry.lam_per_rep is not None
        else entry.lam
    )
    assert lam is not None
    aci = AdaptiveLambda(bundle.calib_m, cfg.alpha) if cfg.use_aci else None
    drift = DriftMonitor(bundle.phi_ref) if cfg.use_drift else None
    return Certifier(lam, aci, drift)


def build_selector(
    bundle: Bundle, cfg: EdgeConfig, prices: Prices, energy: EnergyModel
) -> Selector:
    plan = CostModel(Profiles.from_list(bundle.profiles), bundle.overheads, energy, prices)
    if cfg.selector == "s2_mpc_safe":
        return MpcSafe(plan, cfg.period_ms, mission=cfg.mission)
    return GreedySafe(plan, cfg.mission)


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
        self._last_ack = 0.0
        self._pace_free_at = 0.0
        self._deferred: deque[tuple[int, np.ndarray]] = deque()
        self.records: dict[int, dict] = {}
        self.needs_bundle = False

    # --- transport --------------------------------------------------------------------------
    async def _send(self, window_id: int, rep_code: int, chash: bytes, payload: bytes) -> None:
        buf = fr.pack(fr.Frame(rep_code, chash, window_id, time.time_ns(), payload))
        if self.cfg.shape_bps:  # token-bucket pacing: emulate a rate-limited uplink in-process
            now = time.perf_counter()
            start = max(now, self._pace_free_at)
            self._pace_free_at = start + len(buf) * 8.0 / self.cfg.shape_bps
            asyncio.get_running_loop().call_later(
                max(0.0, self._pace_free_at - now), self.writer.write, buf
            )
        else:
            self.writer.write(buf)
        self.backlog += len(buf)
        self._in_flight.append((window_id, rep_code, time.perf_counter(), len(buf)))

    async def measure_rtt(self, samples: int = 5) -> float:
        rtts = []
        for _ in range(samples):
            t0 = time.perf_counter()
            self.writer.write(fr.pack(fr.Frame(PING_CODE, b"\0\0\0\0", 0, 0, b"")))
            await self.writer.drain()
            await read_reply(self.reader)
            rtts.append((time.perf_counter() - t0) * 1000.0)
        self.rtt_ms = float(min(rtts))
        return self.rtt_ms

    async def _acks(self) -> None:
        """Consume cloud replies: latency, goodput samples, audit feedback."""
        while True:
            try:
                reply = await read_reply(self.reader)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                return
            now = time.perf_counter()
            window_id, rep_code, t_send, nbytes = self._in_flight.popleft()
            self.backlog -= nbytes
            # Goodput sample. A frame sent into an idle link is acknowledged one round trip after
            # its last byte left, so the round trip is removed; a queued frame is acknowledged one
            # transmission time after its predecessor. Tiny frames finish in less than the timing
            # noise and would only measure the delay, so they do not produce a sample.
            if t_send >= self._last_ack:
                elapsed = now - t_send - self.rtt_ms / 1000.0
            else:
                elapsed = now - self._last_ack
            if nbytes >= MIN_SAMPLE_BYTES and elapsed > 1e-4:
                self.est.add(nbytes * 8.0 / elapsed)
            self._last_ack = now
            if reply.status == "NEEDS_BUNDLE":
                self.needs_bundle = True
                continue
            rec = self.records[window_id]
            if rep_code == rec["rep_code"]:  # the live frame (not the audit copy)
                rec["latency_ms"] = (now - rec["t_arrive"]) * 1000.0 - self.rtt_ms / 2.0
                rec["decision"] = reply.decision
                rec["cloud_ms"] = reply.cloud_ms
            if reply.disagreed >= 0:
                rec["disagreed"] = reply.disagreed
                self.certifier.feedback(bool(reply.disagreed))

    # --- the loop ---------------------------------------------------------------------------
    def _link_recovered(self, raw_wire_bytes: float) -> bool:
        """Forward cached raw only if it fits between two arrivals, or as a periodic probe."""
        fits = raw_wire_bytes * 8000.0 / self.est.last_bps <= self.cfg.period_ms
        return fits or time.perf_counter() - self._last_ack >= self.cfg.probe_s

    def _backlog_seen(self) -> float:
        """Unacknowledged bytes minus the estimated progress of the frame currently on the wire."""
        if not self._in_flight:
            return 0.0
        _, _, t_send, nbytes = self._in_flight[0]
        elapsed = time.perf_counter() - max(t_send, self._last_ack)
        return self.backlog - min(nbytes, self.est.bps * max(elapsed, 0.0) / 8.0)

    def _load(self) -> float:
        if self.cfg.load is not None:
            return self.cfg.load
        return min(psutil.cpu_percent(None) / 100.0, 0.95)

    async def run(self, windows: np.ndarray, window_ids: np.ndarray) -> list[dict]:
        await self.measure_rtt()
        acks = asyncio.create_task(self._acks())
        reps = self.bundle.reps
        chash = [bytes.fromhex(r.contract_hash()) for r in reps]

        async def send_rep(wid: int, k: int, x: np.ndarray, live: bool) -> None:
            payload = reps[k].encode(x)
            rec = self.records[wid]
            if live:
                rec["rep_code"] = REP_CODES[reps[k].rep_id]
            rec["bytes"] += len(payload) + fr.OVERHEAD_BYTES
            await self._send(wid, REP_CODES[reps[k].rep_id], chash[k], payload)

        raw_wire = reps[0].size_model().q95_bytes + fr.OVERHEAD_BYTES
        t_start = time.perf_counter()
        for n, (x, wid) in enumerate(zip(windows, window_ids.tolist(), strict=True)):
            await asyncio.sleep(
                max(0.0, t_start + n * self.cfg.period_ms / 1000.0 - time.perf_counter())
            )
            if self.needs_bundle:
                break
            if self._deferred and self.backlog == 0 and self._link_recovered(raw_wire):
                old_wid, old_x = self._deferred.popleft()
                await send_rep(old_wid, 0, old_x, live=True)
            t_arrive = time.perf_counter()
            phi = probe(x)
            ghat = self.bundle.predictor.predict(phi)
            safe = self.certifier.safe_set(ghat, phi)
            state = State(
                bandwidth_bps=self.est.bps,
                rtt_ms=self.rtt_ms,
                backlog_bytes=self._backlog_seen(),
                load=self._load(),
                deadline_ms=self.cfg.deadline_ms,
            )
            dec = self.selector.select(Window(safe, ghat, phi), state)
            rec = {
                "window_id": wid, "t_arrive": t_arrive, "rep": dec.rep, "flag": dec.flag,
                "rep_code": -1, "bytes": 0, "audited": False, "bandwidth_bps": state.bandwidth_bps,
                "backlog_bytes": state.backlog_bytes, "safe": safe.tolist(),
            }  # fmt: skip
            self.records[wid] = rec
            if dec.rep != NO_REP:
                await send_rep(wid, dec.rep, x, live=True)  # select-then-encode: one encoder runs
                if dec.rep != 0 and self.audit.should_audit():
                    rec["audited"] = True
                    await send_rep(wid, 0, x, live=False)
            elif dec.flag == "deferred":
                if len(self._deferred) >= self.cfg.defer_buffer_windows:
                    dropped_wid, _ = self._deferred.popleft()  # buffer full: drop the oldest
                    self.records[dropped_wid]["flag"] = "dropped"
                self._deferred.append((wid, x))
            rec["edge_ms"] = (time.perf_counter() - t_arrive) * 1000.0
        deadline = time.perf_counter() + 5.0 + self.cfg.deadline_ms / 1000.0
        while self._in_flight and time.perf_counter() < deadline and not self.needs_bundle:
            await asyncio.sleep(0.01)
        acks.cancel()
        self.writer.close()
        if self.needs_bundle:
            raise NeedsBundleError("cloud replied NEEDS_BUNDLE: edge and cloud bundles differ")
        return [self.records[w] for w in window_ids.tolist() if w in self.records]


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
