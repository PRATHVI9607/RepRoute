from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kai.contracts import frame as fr

frames = st.builds(
    fr.Frame,
    rep_code=st.integers(0, 255),
    contract_hash=st.binary(min_size=4, max_size=4),
    window_id=st.integers(0, 2**32 - 1),
    ts_ns=st.integers(0, 2**64 - 1),
    payload=st.binary(max_size=512),
)


@given(frames)
def test_round_trip_and_fixed_overhead(f: fr.Frame) -> None:
    buf = fr.pack(f)
    assert len(buf) == len(f.payload) + 28 == f.wire_bytes
    assert fr.unpack(buf) == f
    assert fr.payload_len(buf[: fr.HEADER_BYTES]) == len(f.payload)


@given(frames, st.data())
def test_any_single_bit_flip_is_detected(f: fr.Frame, data: st.DataObject) -> None:
    buf = bytearray(fr.pack(f))
    i = data.draw(st.integers(0, len(buf) * 8 - 1))
    buf[i // 8] ^= 1 << (i % 8)
    with pytest.raises(fr.FrameError):
        fr.unpack(bytes(buf))


def test_rejects_bad_magic_version_length_and_hash_size() -> None:
    good = fr.pack(fr.Frame(1, b"abcd", 2, 3, b"xyz"))
    with pytest.raises(fr.FrameError, match="magic"):
        fr.unpack(b"ZZ" + good[2:])
    with pytest.raises(fr.FrameError, match="version"):
        fr.unpack(good[:2] + b"\x09" + good[3:])
    with pytest.raises(fr.FrameError, match="length"):
        fr.unpack(good + b"\x00")
    with pytest.raises(fr.FrameError, match="truncated"):
        fr.payload_len(good[:10])
    with pytest.raises(fr.FrameError, match="4 bytes"):
        fr.pack(fr.Frame(1, b"abc", 2, 3, b""))


def test_a_hostile_length_field_is_rejected_before_anything_is_buffered() -> None:
    """The length comes from the network: 4 GB must not be read on a peer's say-so."""
    import struct

    header = struct.pack(">2sBB4sIQI", fr.MAGIC, fr.PROTO_VER, 1, b"abcd", 0, 0, 0xFFFFFFFF)
    with pytest.raises(fr.FrameError, match="limit"):
        fr.payload_len(header)
    assert (
        fr.payload_len(header[:-4] + struct.pack(">I", fr.MAX_PAYLOAD_BYTES))
        == fr.MAX_PAYLOAD_BYTES
    )
    with pytest.raises(fr.FrameError, match="exceeds"):
        fr.pack(fr.Frame(1, b"abcd", 0, 0, bytes(fr.MAX_PAYLOAD_BYTES + 1)))
