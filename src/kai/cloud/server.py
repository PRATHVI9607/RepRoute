"""Cloud plane: decode frames, check the contract hash, run the matching head, reply."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import typer

from kai.cloud.heads import HeadZoo
from kai.contracts import frame as fr
from kai.contracts.representation import RAW_REP_ID, REP_CODES
from kai.edge.bundle import load_bundle
from kai.link.framing import PING_CODE, Reply, pack_reply, read_frame

RAW_CODE = REP_CODES[RAW_REP_ID]
TelemetryCallback = Callable[[int, str, int, float, float, float], None]
TelemetryEvent = tuple[int, str, int, float, float, float]
logger = logging.getLogger(__name__)


class CloudServer:
    def __init__(self, zoo: HeadZoo, on_decision: TelemetryCallback | None = None) -> None:
        self.zoo = zoo
        self.frames = 0
        self.on_decision = on_decision

    def handle(self, frame: fr.Frame, pending: dict[int, dict[int, int]]) -> Reply:
        """Decide one frame. `pending` pairs the two copies of an audited window."""
        reply, event = self._decide(frame, pending)
        self._emit(event)
        return reply

    def _decide(
        self, frame: fr.Frame, pending: dict[int, dict[int, int]]
    ) -> tuple[Reply, TelemetryEvent | None]:
        """The reply plus the telemetry event, which the caller emits once the reply is sent."""
        if frame.rep_code == PING_CODE:
            return Reply(frame.window_id, PING_CODE, "PONG"), None
        rep = self.zoo.by_hash.get(frame.contract_hash)
        if rep is None or REP_CODES[rep.rep_id] != frame.rep_code:
            # the edge runs a codec or encoder version this cloud holds no head for
            return Reply(frame.window_id, frame.rep_code, "NEEDS_BUNDLE"), None
        t0 = time.perf_counter_ns()
        try:
            decision, t_dec_ms, t_head_ms = self.zoo.decide_timed(rep.rep_id, frame.payload)
        except Exception:  # CRC-valid but undecodable: answer it, keep the connection
            logger.exception(
                "undecodable payload for window_id=%d rep_id=%s", frame.window_id, rep.rep_id
            )
            return Reply(frame.window_id, frame.rep_code, "BAD_PAYLOAD"), None
        cloud_ms = (time.perf_counter_ns() - t0) / 1e6
        self.frames += 1
        event = (frame.window_id, rep.rep_id, decision, t_dec_ms, t_head_ms, cloud_ms)
        seen = pending.setdefault(frame.window_id, {})
        seen[frame.rep_code] = decision
        disagreed = -1
        if (
            len(seen) == 2 and RAW_CODE in seen
        ):  # audit pair complete: compare with the raw decision
            other = next(v for k, v in seen.items() if k != RAW_CODE)
            disagreed = int(other != seen[RAW_CODE])
            del pending[frame.window_id]
        elif len(pending) > 4096:  # unaudited windows never complete a pair: bound the memory
            pending.pop(next(iter(pending)))
        return Reply(frame.window_id, frame.rep_code, "OK", decision, cloud_ms, disagreed), event

    def _emit(self, event: TelemetryEvent | None) -> None:
        if self.on_decision is None or event is None:
            return
        try:
            self.on_decision(*event)
        except Exception:
            logger.warning(
                "Telemetry callback failed for window_id=%d rep_id=%s",
                event[0],
                event[1],
                exc_info=True,
            )

    async def serve_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        pending: dict[int, dict[int, int]] = {}
        try:
            while True:
                reply, event = self._decide(await read_frame(reader), pending)
                writer.write(pack_reply(reply))
                await writer.drain()
                self._emit(event)  # after the reply: telemetry I/O must not delay the ack
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass  # client closed the connection
        except fr.FrameError:
            # a corrupt stream cannot be resynchronised: drop the connection
            logger.warning("dropping connection on a corrupt frame", exc_info=True)
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


async def serve(
    bundle_dir: Path,
    host: str,
    port: int,
    telemetry_out: Path | None = None,
) -> None:
    load_bundle(bundle_dir)  # refuse to serve a bundle whose files no longer match its manifest
    sink = None
    on_decision = None
    if telemetry_out is not None:
        telemetry_out.parent.mkdir(parents=True, exist_ok=True)
        sink = telemetry_out.open("a", encoding="utf-8")

        def _on_decision(
            window_id: int,
            rep_id: str,
            decision: int,
            t_dec_ms: float,
            t_head_ms: float,
            cloud_ms: float,
        ) -> None:
            record = {
                "window_id": window_id,
                "rep_id": rep_id,
                "decision": decision,
                "measured_cloud_dec_ms": t_dec_ms,
                "measured_cloud_head_ms": t_head_ms,
                "cloud_ms": cloud_ms,
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()

        on_decision = _on_decision

    server = CloudServer(HeadZoo(bundle_dir), on_decision=on_decision)
    try:
        async with await asyncio.start_server(server.serve_client, host, port) as srv:
            await srv.serve_forever()
    finally:
        if sink is not None:
            sink.close()


def main(
    bundle: Path,
    host: str = "0.0.0.0",
    port: int = 7400,
    telemetry_out: Path | None = None,
) -> None:
    """Run the cloud server for one calibration bundle."""
    asyncio.run(serve(bundle, host, port, telemetry_out=telemetry_out))


if __name__ == "__main__":
    typer.run(main)
