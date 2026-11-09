"""Icentia11k loader on synthetic MIT annotation bytes and tiny fake WFDB segments (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from kai.data import icentia

N, V, S, Q = (icentia.CODE[k] for k in "NVSQ")
RHYTHM, NOTE = 28, 22


def _word(code: int, d: int) -> bytes:
    return ((code << 10) | d).to_bytes(2, "little")


def _encode(anns: list[tuple[int, int, bytes]]) -> bytes:
    """(sample, code, aux) → MIT annotation bytes; long or negative steps, and code 0 (whose zero
    step would read as the end word), go through SKIP as in the real files."""
    out, t = b"", 0
    for sample, code, aux in anns:
        d = sample - t
        if not 0 <= d <= 1023 or code == 0:
            skip = (d - 1) % (1 << 32)  # 32-bit two's complement, high word first
            out += _word(59, 0) + (skip >> 16).to_bytes(2, "little")
            out += (skip & 0xFFFF).to_bytes(2, "little")
            d = 1
        out += _word(code, d)
        if aux:
            out += _word(63, len(aux)) + aux + b"\0" * (len(aux) % 2)
        t = sample
    return out + _word(0, 0)


def test_parse_atr_round_trip_with_skip_aux_and_pseudo_codes() -> None:
    anns = [
        (0, NOTE, b"## time resolution: 250"),  # odd AUX length: padded
        (0, 0, b""),
        (28, N, b"None"),
        (28, RHYTHM, b"(N"),
        (2000, V, b""),  # delta > 1023: SKIP
        (1990, S, b""),  # negative step: SKIP with a negative interval
        (100_000, Q, b""),
    ]
    buf = _encode(anns)
    # NUM / SUB / CHN after an annotation carry fields, never time
    buf = buf[:-2] + _word(60, 3) + _word(61, 1) + _word(62, 0) + _word(N, 7) + _word(0, 0)
    samples, codes = icentia.parse_atr(buf + b"\xff\xff")  # bytes after the end word are ignored
    assert samples.tolist() == [a[0] for a in anns] + [100_007]
    assert codes.tolist() == [a[1] for a in anns] + [N]


def test_parse_atr_rejects_truncated_or_negative() -> None:
    with pytest.raises(ValueError, match="odd"):
        icentia.parse_atr(b"\x00")
    with pytest.raises(ValueError, match="SKIP"):
        icentia.parse_atr(_word(59, 0) + b"\x00\x00")
    with pytest.raises(ValueError, match="AUX"):
        icentia.parse_atr(_word(N, 1) + _word(63, 9) + b"abcd")
    with pytest.raises(ValueError, match="negative"):
        icentia.parse_atr(_encode([(-5, N, b"")]))


def test_label_windows_rules() -> None:
    w = 100
    beats = [
        (10, N), (50, N),  # window 0: N only → 0
        (110, N), (150, S),  # window 1: S → 1
        (210, S), (250, V),  # window 2: V wins over S → 2
        (310, V), (350, Q),  # window 3: Q drops it, even with V
        (420, RHYTHM),  # window 4: no beat → dropped
        (520, N),  # window 5: N
        (650, V),  # past the last full window (n_samples 640): ignored
    ]  # fmt: skip
    samples = np.array([b[0] for b in beats], dtype=np.int64)
    codes = np.array([b[1] for b in beats], dtype=np.int64)
    y = icentia.label_windows(samples, codes, n_samples=640, window_n=w)
    assert y.tolist() == [0, 1, 2, -1, -1, 0]


def test_cap_per_class(rng: np.random.Generator) -> None:
    y = np.array([0] * 20 + [1] * 3 + [-1] * 5 + [2] * 9, dtype=np.int64)
    idx = icentia.cap_per_class(y, 6, rng)
    assert np.all(np.diff(idx) > 0)
    assert [int((y[idx] == c).sum()) for c in (0, 1, 2)] == [6, 3, 6]
    assert (y[idx] >= 0).all()
    again = icentia.cap_per_class(y, 6, np.random.default_rng(7))
    assert np.array_equal(again, icentia.cap_per_class(y, 6, np.random.default_rng(7)))


def test_catalogue_takes_first_segment_and_validates_names() -> None:
    def seg(p: int, s: int) -> list[str]:
        rec = f"p{p:05d}"[:3] + f"/p{p:05d}/p{p:05d}_s{s:02d}"
        return [f"{rec}{ext}" for ext in (".hea", ".dat", ".atr")]

    names = seg(0, 1) + seg(0, 0) + seg(2, 3) + seg(1000, 0) + ["p01/p00003/p00003_s00.hea"]
    names += seg(4, 0)[:2] + seg(4, 5)  # s00 lacks .atr: s05 is the first usable segment
    names += ["../../etc/passwd.hea", "RECORDS"]
    assert icentia.catalogue(names, n_patients=1000) == {
        "p00/p00000/p00000_s00": "P0",
        "p00/p00002/p00002_s03": "P2",
        "p00/p00004/p00004_s05": "P4",
    }


def _write_segment(raw, rec: str, x: np.ndarray, anns: list[tuple[int, int, bytes]]) -> None:
    path = raw / rec
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.name
    checksum = int(x.astype(np.int64).sum()) % 65536
    path.with_suffix(".hea").write_text(
        f"{stem} 1 250 {len(x)}\n{stem}.dat 16 74.36(0)/mV 16 0 {x[0]} {checksum} 0 ecg\n"
    )
    x.astype("<i2").tofile(path.with_suffix(".dat"))
    path.with_suffix(".atr").write_bytes(_encode(anns))


def test_windowset_from_fake_segments(tmp_path, rng: np.random.Generator) -> None:
    w, n_win = 64, 20
    recs = {p: f"p00/p{p:05d}/p{p:05d}_s00" for p in range(4)}
    signals = {p: rng.integers(-3000, 3000, size=w * n_win + 10).astype(np.int16) for p in recs}
    beats = {  # beat code of each window (0 = none)
        0: [N] * 12 + [S] * 2 + [V] * 3 + [Q] * 2 + [0],
        1: [N] * 20,
        2: [V] * 8 + [0] * 12,
    }
    for p, codes in beats.items():
        anns = [(k * w + 5, c, b"None") for k, c in enumerate(codes) if c]
        _write_segment(tmp_path, recs[p], signals[p], anns)
    sums = "".join(f"{'0' * 64} {recs[p]}{ext}\n" for p in recs for ext in (".hea", ".dat", ".atr"))
    (tmp_path / icentia.SUMS).write_text(sums)  # P3 is listed but not downloaded

    ws, report = icentia.build_windowset(tmp_path, window_n=w, max_per_class=6, seed=1)

    assert (ws.sample_rate_hz, ws.dataset_id, ws.window_n) == (250.0, "icentia-v1", w)
    assert set(ws.condition.tolist()) == {"ambulatory"}
    assert report["windows_per_class"] == {"N": 12, "S": 2, "V": 9}
    assert report["patients_per_class"] == {"N": 2, "S": 1, "V": 2}
    assert icentia.split_labels(ws) == {"P0": 2, "P1": 0, "P2": 2}
    assert report["skipped_records"] == {"missing": [recs[3]], "unreadable": [], "too_short": []}
    for k in range(len(ws)):  # each window is the stored ADC counts at its own position
        p = int(ws.bearing_id[k][1:])
        i = int(np.flatnonzero((signals[p][: n_win * w].reshape(n_win, w) == ws.x[k]).all(1))[0])
        assert ws.label[k] == {N: 0, S: 1, V: 2}[beats[p][i]]
    again, _ = icentia.build_windowset(tmp_path, window_n=w, max_per_class=6, seed=1)
    assert np.array_equal(again.x, ws.x)
