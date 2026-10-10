"""Wire frame (PRD §5.4). Identical 28-byte overhead for every representation.

magic(2) | proto_ver(1) | rep_id(1) | contract_hash(4) | window_id(4) | ts_ns(8) | len(4)
| payload | crc32(4)
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"KA"
PROTO_VER = 1
_HEADER = struct.Struct(">2sBB4sIQI")
_CRC = struct.Struct(">I")
HEADER_BYTES = _HEADER.size
OVERHEAD_BYTES = HEADER_BYTES + _CRC.size
assert OVERHEAD_BYTES == 28
# The length field is 32 bits and comes from the network. Raw at N = 65 536 is 128 kB; anything
# beyond this cap is a corrupt or hostile frame, not a payload.
MAX_PAYLOAD_BYTES = 1 << 20


class FrameError(ValueError):
    """Malformed, corrupted or incompatible frame."""


@dataclass(frozen=True)
class Frame:
    rep_code: int
    contract_hash: bytes  # 4 bytes
    window_id: int
    ts_ns: int
    payload: bytes

    @property
    def wire_bytes(self) -> int:
        return OVERHEAD_BYTES + len(self.payload)


def pack(frame: Frame) -> bytes:
    if len(frame.contract_hash) != 4:
        raise FrameError(f"contract_hash must be 4 bytes, got {len(frame.contract_hash)}")
    if len(frame.payload) > MAX_PAYLOAD_BYTES:
        raise FrameError(f"payload of {len(frame.payload)} bytes exceeds {MAX_PAYLOAD_BYTES}")
    head = _HEADER.pack(
        MAGIC,
        PROTO_VER,
        frame.rep_code,
        frame.contract_hash,
        frame.window_id,
        frame.ts_ns,
        len(frame.payload),
    )
    body = head + frame.payload
    return body + _CRC.pack(zlib.crc32(body))


def payload_len(header: bytes) -> int:
    """Validate a 24-byte header and return the payload length that follows it."""
    if len(header) != HEADER_BYTES:
        raise FrameError(f"truncated header: {len(header)} bytes")
    magic, ver, _, _, _, _, n = _HEADER.unpack(header)
    if magic != MAGIC:
        raise FrameError(f"bad magic {magic!r}")
    if ver != PROTO_VER:
        raise FrameError(f"unsupported protocol version {ver}")
    if n > MAX_PAYLOAD_BYTES:
        raise FrameError(f"payload length {n} exceeds the {MAX_PAYLOAD_BYTES}-byte limit")
    return int(n)


def unpack(buf: bytes) -> Frame:
    n = payload_len(buf[:HEADER_BYTES])
    if len(buf) != OVERHEAD_BYTES + n:
        raise FrameError(
            f"length mismatch: header says {n}, buffer has {len(buf) - OVERHEAD_BYTES}"
        )
    body, (crc,) = buf[:-4], _CRC.unpack(buf[-4:])
    if zlib.crc32(body) != crc:
        raise FrameError("crc32 mismatch")
    _, _, rep_code, chash, window_id, ts_ns, _ = _HEADER.unpack(buf[:HEADER_BYTES])
    return Frame(rep_code, chash, window_id, ts_ns, buf[HEADER_BYTES:-4])
