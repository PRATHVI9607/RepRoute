"""Frame transport over asyncio streams, plus the cloud → edge reply codec (msgpack)."""

from __future__ import annotations

import asyncio
import struct
from dataclasses import asdict, dataclass

import msgpack

from kai.contracts import frame as fr

PING_CODE = 255  # rep_code reserved for RTT probes: echoed without touching the head zoo
MAX_REPLY_BYTES = 4096  # a reply is six small fields; a longer length prefix is not a reply
_LEN = struct.Struct(">I")


async def read_frame(reader: asyncio.StreamReader) -> fr.Frame:
    """Read one frame. Raises `IncompleteReadError` at EOF and `FrameError` on corruption."""
    header = await reader.readexactly(fr.HEADER_BYTES)
    rest = await reader.readexactly(fr.payload_len(header) + 4)
    return fr.unpack(header + rest)


@dataclass(frozen=True)
class Reply:
    window_id: int
    rep_code: int
    status: str  # "OK" | "NEEDS_BUNDLE" | "BAD_PAYLOAD" | "PONG"
    decision: int = -1
    cloud_ms: float = 0.0  # decode + head time measured on the cloud
    disagreed: int = -1  # audit result once both copies of a window arrived: 0 / 1, else -1


def pack_reply(reply: Reply) -> bytes:
    body: bytes = msgpack.packb(asdict(reply))
    return _LEN.pack(len(body)) + body


async def read_reply(reader: asyncio.StreamReader) -> Reply:
    (n,) = _LEN.unpack(await reader.readexactly(4))
    if n > MAX_REPLY_BYTES:
        raise fr.FrameError(f"reply length {n} exceeds the {MAX_REPLY_BYTES}-byte limit")
    return Reply(**msgpack.unpackb(await reader.readexactly(n)))
