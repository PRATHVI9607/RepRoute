"""The loopback token bucket must not add asyncio's ~1 ms timer resolution to every frame."""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import deque
from typing import Any

import pytest

from kai.contracts import frame as fr
from kai.edge.runtime import Edge, EdgeConfig


class _Writer:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, buf: bytes) -> None:
        self.writes.append(buf)


def _paced_edge(shape_bps: float) -> Any:
    edge = object.__new__(Edge)
    edge.cfg = EdgeConfig(shape_bps=shape_bps)
    edge._pace_free_at = 0.0
    edge.writer = _Writer()  # type: ignore[assignment]
    edge.backlog = 0.0
    edge._in_flight = deque()
    edge._tx = asyncio.Queue()
    return edge


async def _send_and_flush(edge: Edge, payload: bytes) -> float:
    """`_send` plus the sender task, run until the queue is empty."""
    edge._tx = asyncio.Queue()  # bound to this loop
    sender = asyncio.create_task(edge._sender())
    pace_s = await edge._send(1, 7, b"\0" * 4, payload)
    await edge._tx.join()
    sender.cancel()
    return pace_s


def test_token_bucket_waits_for_the_nominal_serialization_time() -> None:
    edge = _paced_edge(256_000.0)

    payload = b"\0" * 16
    nbytes = len(payload) + fr.OVERHEAD_BYTES
    expected_s = nbytes * 8.0 / 256_000.0
    waits: list[float] = []
    paces: list[float] = []

    for _ in range(5):
        edge._pace_free_at = 0.0
        edge.writer.writes.clear()
        t0 = time.perf_counter()
        paces.append(asyncio.run(_send_and_flush(edge, payload)))
        waits.append(time.perf_counter() - t0)

    assert edge.writer.writes, "the frame must have been written"
    assert statistics.median(paces) == pytest.approx(expected_s, abs=2e-4)
    # The exact wall time moves under parallel test load; the old call_later path returned
    # immediately, so a lower bound is enough to catch a regression.
    assert min(waits) >= expected_s * 0.5


def test_a_paced_frame_is_stamped_when_its_serialization_starts() -> None:
    """Stamping after the wait made loopback goodput samples ~10 Mbit/s at any shaped rate."""
    edge = _paced_edge(256_000.0)
    payload = b"\0" * 2000  # above MIN_SAMPLE_BYTES: such frames feed the rate estimator
    serialization_s = (len(payload) + fr.OVERHEAD_BYTES) * 8.0 / 256_000.0
    t0 = time.perf_counter()
    asyncio.run(_send_and_flush(edge, payload))
    t_send = edge._in_flight[0][2]
    assert t0 <= t_send < t0 + 0.2 * serialization_s
    assert edge._pace_free_at == pytest.approx(t_send + serialization_s)


def test_a_paced_frame_does_not_block_the_reply_reader() -> None:
    """A blocking sleep froze `_acks`, so replies were stamped late by the whole pacing wait."""
    edge = _paced_edge(256_000.0)
    payload = b"\0" * (640 - fr.OVERHEAD_BYTES)  # 20 ms at 256 kbit/s

    async def main() -> int:
        ticks, stop = 0, False

        async def ticker() -> None:
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0.001)

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        before = ticks
        await _send_and_flush(edge, payload)
        during = ticks - before
        stop = True
        await task
        return during

    assert asyncio.run(main()) >= 3
    assert edge.writer.writes


def test_drain_stops_at_once_when_the_cloud_closes_the_stream(caplog) -> None:
    edge = _paced_edge(1e6)
    edge.cfg = EdgeConfig(deadline_ms=250.0)
    edge.needs_bundle = False
    edge._in_flight = deque([(1, 7, 0.0, 100)])

    async def main() -> float:
        acks = asyncio.create_task(asyncio.sleep(0))  # a reply reader that already hit EOF
        await acks
        t0 = time.perf_counter_ns()
        await edge._drain(acks)
        return (time.perf_counter_ns() - t0) / 1e9

    with caplog.at_level("WARNING", logger="kai.edge.runtime"):
        assert asyncio.run(main()) < 0.5  # not the 5 s drain timeout
    assert "unacknowledged" in caplog.text
    assert "closed the stream" in caplog.text


def test_only_certified_audits_reach_the_certifier() -> None:
    """Uncertified (liveness-first) sends are outside the certificate: no ACI feedback."""
    from collections import deque

    from kai.link.framing import Reply, pack_reply
    from kai.link.traces import HarmonicMeanEstimator

    edge = _paced_edge(1e6)
    fed: list[bool] = []
    edge.certifier = type("C", (), {"feedback": lambda self, d: fed.append(d)})()
    edge.est, edge.rtt_ms, edge._last_ack, edge.needs_bundle = (
        HarmonicMeanEstimator(1e6),
        0,
        0,
        False,
    )
    edge._in_flight = deque([(1, 7, 0.0, 100), (2, 7, 0.0, 100)])
    edge.backlog = 200.0
    edge.records = {
        w: {"flag": flag, "rep_code": 7, "t_arrive": 0.0}
        for w, flag in ((1, "certified"), (2, "uncertified"))
    }

    async def main() -> None:
        edge.reader = asyncio.StreamReader()
        for w in (1, 2):
            edge.reader.feed_data(pack_reply(Reply(w, 7, "OK", 0, 0.0, disagreed=1)))
        edge.reader.feed_eof()
        await edge._acks()

    asyncio.run(main())
    assert fed == [True]
    assert edge.records[2]["disagreed"] == 1  # still recorded, just not fed back


def test_load_reads_the_core_the_edge_runs_on(monkeypatch) -> None:
    import psutil

    edge = object.__new__(Edge)
    edge.cfg = EdgeConfig()
    monkeypatch.setattr(psutil.Process, "cpu_num", lambda self: 2)
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None, percpu=False: [5.0, 6.0, 40.0])
    assert edge._load() == pytest.approx(0.4)


def test_edge_compute_overlaps_transmission_without_a_growing_backlog() -> None:
    """ADR 0028 §13: 20 ms of edge work + 60 ms of transmission per 64 ms period is stable only
    if the loop does not wait for the pacer (serialised it needs 80 ms > 64 ms per window)."""
    edge = _paced_edge(256_000.0)
    payload = b"\0" * (1920 - fr.OVERHEAD_BYTES)  # 60 ms at 256 kbit/s
    n, period_s = 12, 0.064

    async def main() -> tuple[float, float]:
        sender = asyncio.create_task(edge._sender())
        t0 = time.perf_counter()
        lag_s = 0.0
        for k in range(n):
            await asyncio.sleep(max(0.0, t0 + k * period_s - time.perf_counter()))
            await asyncio.sleep(0.020)  # edge compute
            await edge._send(k, 7, b"\0" * 4, payload)
            lag_s = max(lag_s, time.perf_counter() - (t0 + k * period_s) - 0.020)
        loop_s = time.perf_counter() - t0
        await edge._tx.join()
        sender.cancel()
        return loop_s, lag_s

    loop_s, lag_s = asyncio.run(main())
    assert loop_s < n * period_s + 0.05  # the loop ran on schedule: it never waited for the pacer
    assert lag_s < 0.02  # no growing wait for earlier frames


def test_a_dying_sender_does_not_hang_run() -> None:
    """The join on the shaped-frame queue must not wait for a sender that has raised."""
    edge = _paced_edge(1e6)

    class _Dead:
        def write(self, buf: bytes) -> None:
            raise ConnectionResetError("closed")

    edge.writer = _Dead()

    async def main() -> None:
        sender = asyncio.create_task(edge._sender())
        await edge._send(1, 7, b"\0" * 4, b"\0" * 16)
        await edge._flush_tx(sender)

    with pytest.raises(ConnectionResetError):
        asyncio.run(asyncio.wait_for(main(), 2.0))


def test_raw_agree_feedback_is_fed_at_the_ack_not_at_the_send() -> None:
    """ADR 0028 §7: the free "raw agrees" ACI feedback waits for the raw frame's ack."""
    from kai.link.framing import Reply, pack_reply
    from kai.link.traces import HarmonicMeanEstimator

    edge = _paced_edge(1e6)
    fed: list[bool] = []
    edge.certifier = type("C", (), {"feedback": lambda self, d: fed.append(d)})()
    edge.est, edge.rtt_ms, edge._last_ack, edge.needs_bundle = (
        HarmonicMeanEstimator(1e6), 0, 0, False,
    )  # fmt: skip
    edge._in_flight = deque([(1, 7, 0.0, 100)])
    edge.backlog = 100.0
    edge.records = {1: {"flag": "certified", "rep_code": 7, "t_arrive": 0.0, "agree_pending": True}}

    async def main() -> None:
        edge.reader = asyncio.StreamReader()
        edge.reader.feed_data(pack_reply(Reply(1, 7, "OK", 0, 0.0)))
        edge.reader.feed_eof()
        assert fed == []  # not at the send
        await edge._acks()

    asyncio.run(main())
    assert fed == [False]
