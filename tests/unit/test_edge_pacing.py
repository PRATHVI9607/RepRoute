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
