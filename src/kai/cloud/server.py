"""Cloud plane: decode frames, check the contract hash, run the matching head, reply."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer

from kai.cloud.heads import HeadZoo
from kai.contracts import frame as fr
from kai.contracts.representation import RAW_REP_ID, REP_CODES
from kai.edge.bundle import load_bundle
from kai.link.framing import PING_CODE, Reply, pack_reply, read_frame

RAW_CODE = REP_CODES[RAW_REP_ID]


class CloudServer:
    def __init__(self, zoo: HeadZoo) -> None:
        self.zoo = zoo
        self.frames = 0

    def handle(self, frame: fr.Frame, pending: dict[int, dict[int, int]]) -> Reply:
        """Decide one frame. `pending` pairs the two copies of an audited window."""
        if frame.rep_code == PING_CODE:
            return Reply(frame.window_id, PING_CODE, "PONG")
        rep = self.zoo.by_hash.get(frame.contract_hash)
        if rep is None or REP_CODES[rep.rep_id] != frame.rep_code:
            # the edge runs a codec or encoder version this cloud holds no head for
            return Reply(frame.window_id, frame.rep_code, "NEEDS_BUNDLE")
        t0 = time.perf_counter_ns()
        decision = self.zoo.decide(rep.rep_id, frame.payload)
        cloud_ms = (time.perf_counter_ns() - t0) / 1e6
        self.frames += 1
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
        return Reply(frame.window_id, frame.rep_code, "OK", decision, cloud_ms, disagreed)

    async def serve_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        pending: dict[int, dict[int, int]] = {}
        try:
            while True:
                writer.write(pack_reply(self.handle(await read_frame(reader), pending)))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass  # client closed the connection
        except fr.FrameError:
            pass  # a corrupt stream cannot be resynchronised: drop the connection
        finally:
            writer.close()


async def serve(bundle_dir: Path, host: str, port: int) -> None:
    load_bundle(bundle_dir)  # refuse to serve a bundle whose files no longer match its manifest
    server = CloudServer(HeadZoo(bundle_dir))
    async with await asyncio.start_server(server.serve_client, host, port) as srv:
        await srv.serve_forever()


def main(bundle: Path, host: str = "0.0.0.0", port: int = 7400) -> None:
    """Run the cloud server for one calibration bundle."""
    asyncio.run(serve(bundle, host, port))


if __name__ == "__main__":
    typer.run(main)
