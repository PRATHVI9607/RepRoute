"""The loopback token bucket must not add asyncio's ~1 ms timer resolution to every frame."""

from __future__ import annotations

import asyncio
import statistics
import time

import pytest

from kai.contracts import frame as fr
from kai.edge.runtime import Edge, EdgeConfig


class _Writer:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, buf: bytes) -> None:
        self.writes.append(buf)


def test_token_bucket_waits_for_the_nominal_serialization_time() -> None:
    edge = object.__new__(Edge)
    edge.cfg = EdgeConfig(shape_bps=256_000.0)
    edge._pace_free_at = 0.0
    edge.writer = _Writer()
    edge.backlog = 0.0
    edge._in_flight = []

    payload = b"\0" * 16
    nbytes = len(payload) + fr.OVERHEAD_BYTES
    expected_s = nbytes * 8.0 / edge.cfg.shape_bps
    waits: list[float] = []
    paces: list[float] = []

    for _ in range(5):
        edge._pace_free_at = 0.0
        edge.writer.writes.clear()
        t0 = time.perf_counter()
        pace_s = asyncio.run(edge._send(1, 7, b"\0" * 4, payload))
        waits.append(time.perf_counter() - t0)
        paces.append(pace_s)

    assert edge.writer.writes, "the frame must have been written"
    assert statistics.median(paces) == pytest.approx(expected_s, abs=2e-4)
    # The exact wall time moves under parallel test load; the old call_later path returned
    # immediately, so a lower bound is enough to catch a regression.
    assert min(waits) >= expected_s * 0.5


def test_a_paced_frame_is_stamped_when_its_serialization_starts() -> None:
    """Stamping after the wait made loopback goodput samples ~10 Mbit/s at any shaped rate."""
    edge = object.__new__(Edge)
    edge.cfg = EdgeConfig(shape_bps=256_000.0)
    edge._pace_free_at = 0.0
    edge.writer = _Writer()
    edge.backlog = 0.0
    edge._in_flight = []
    payload = b"\0" * 2000  # above MIN_SAMPLE_BYTES: such frames feed the rate estimator
    serialization_s = (len(payload) + fr.OVERHEAD_BYTES) * 8.0 / edge.cfg.shape_bps
    t0 = time.perf_counter()
    asyncio.run(edge._send(1, 7, b"\0" * 4, payload))
    t_send = edge._in_flight[0][2]
    assert t0 <= t_send < t0 + 0.2 * serialization_s
    assert edge._pace_free_at == pytest.approx(t_send + serialization_s)


def _paced_edge(shape_bps: float) -> Edge:
    edge = object.__new__(Edge)
    edge.cfg = EdgeConfig(shape_bps=shape_bps)
    edge._pace_free_at = 0.0
    edge.writer = _Writer()
    edge.backlog = 0.0
    edge._in_flight = []
    return edge


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
        await edge._send(1, 7, b"\0" * 4, payload)
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
    edge._in_flight = [(1, 7, 0.0, 100)]

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
