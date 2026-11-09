"""Frame decoding from split reads, and the cloud's reply-before-telemetry ordering."""

from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kai.cloud.server import CloudServer
from kai.contracts import frame as fr
from kai.contracts.representation import RAW_REP_ID, REP_CODES
from kai.link.framing import read_reply

frames = st.builds(
    fr.Frame,
    rep_code=st.integers(0, 255),
    contract_hash=st.binary(min_size=4, max_size=4),
    window_id=st.integers(0, 2**32 - 1),
    ts_ns=st.integers(0, 2**64 - 1),
    payload=st.binary(max_size=512),
)


@given(frames, st.data())
def test_split_decode_round_trips_and_rejects_any_bit_flip(f: fr.Frame, data) -> None:
    buf = fr.pack(f)
    head, rest = buf[: fr.HEADER_BYTES], buf[fr.HEADER_BYTES :]
    assert fr.unpack_parts(head, rest) == f == fr.unpack(buf)
    assert type(fr.unpack(buf).payload) is bytes
    flipped = bytearray(buf)
    i = data.draw(st.integers(0, len(buf) * 8 - 1))
    flipped[i // 8] ^= 1 << (i % 8)
    with pytest.raises(fr.FrameError):
        fr.unpack_parts(bytes(flipped[: fr.HEADER_BYTES]), bytes(flipped[fr.HEADER_BYTES :]))


def test_reply_is_on_the_wire_before_telemetry_runs_and_bad_payloads_are_logged(caplog) -> None:
    raw_code = REP_CODES[RAW_REP_ID]
    chash = b"abcd"
    calls = iter([(1, 0.1, 0.2), ValueError("undecodable")])

    def decide_timed(rep_id, payload):
        out = next(calls)
        if isinstance(out, Exception):
            raise out
        return out

    zoo = SimpleNamespace(
        by_hash={chash: SimpleNamespace(rep_id=RAW_REP_ID)}, decide_timed=decide_timed
    )
    seen_reply_bytes: list[int] = []
    client = socket.socket()

    def on_decision(*event) -> None:  # the reply must already sit in the client's buffer
        seen_reply_bytes.append(len(client.recv(64, socket.MSG_PEEK | socket.MSG_DONTWAIT)))

    async def main():
        server = CloudServer(zoo, on_decision=on_decision)
        srv = await asyncio.start_server(server.serve_client, "127.0.0.1", 0)
        client.setblocking(False)
        await asyncio.get_running_loop().sock_connect(client, srv.sockets[0].getsockname())
        reader, writer = await asyncio.open_connection(sock=client)
        writer.write(fr.pack(fr.Frame(raw_code, chash, 5, 0, b"x")))
        first = await read_reply(reader)
        writer.write(fr.pack(fr.Frame(raw_code, chash, 6, 0, b"y")))
        second = await read_reply(reader)
        writer.close()
        await writer.wait_closed()
        srv.close()
        await srv.wait_closed()
        return first, second

    with caplog.at_level("ERROR", logger="kai.cloud.server"):
        first, second = asyncio.run(main())
    assert first.status == "OK"
    assert first.decision == 1
    assert seen_reply_bytes
    assert seen_reply_bytes[0] > 0
    assert second.status == "BAD_PAYLOAD"
    assert "undecodable payload for window_id=6" in caplog.text
