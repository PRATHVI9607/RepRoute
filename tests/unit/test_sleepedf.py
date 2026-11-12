"""Sleep-EDF loader on tiny EDF/EDF+ files written in the published layout (no network)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import resample_poly

from kai.data import sleepedf
from kai.data.pu import quantise

START = "24.04.8916.13.00"  # dd.mm.yy + hh.mm.ss, as in the header


def _field(value: object, width: int) -> bytes:
    return str(value).ljust(width).encode("ascii")


def _edf(signals: list[dict], n_records: int, record_s: float, start: str = START) -> bytes:
    """signals: label, unit, spr, pmin, pmax, dmin, dmax, data (int16 [n_records·spr])."""
    ns = len(signals)
    head = (
        _field("0", 8)
        + _field("X X X X", 80)
        + _field("Startdate X", 80)
        + start.encode()
        + _field(256 * (ns + 1), 8)
        + _field("EDF+C", 44)
        + _field(n_records, 8)
        + _field(record_s, 8)
        + _field(ns, 4)
    )
    keys = ("label", "transducer", "unit", "pmin", "pmax", "dmin", "dmax", "pre", "spr", "res")
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    for key, width in zip(keys, widths, strict=True):
        head += b"".join(_field(s.get(key, ""), width) for s in signals)
    blocks = [s["data"].astype("<i2").reshape(n_records, s["spr"]) for s in signals]
    return head + np.concatenate(blocks, axis=1).tobytes()


def _eeg(digital: np.ndarray, n_records: int, spr: int = 3000) -> dict:
    return {
        "label": "EEG Fpz-Cz",
        "unit": "uV",
        "spr": spr,
        "pmin": -192,
        "pmax": 192,
        "dmin": -2048,
        "dmax": 2047,
        "data": digital[: n_records * spr],
    }


def _hypnogram(annotations: list[tuple[int, int, str]], start: str = START) -> bytes:
    tal = b"+0\x14\x14\x00" + b"".join(
        f"+{on}\x15{dur}\x14{text}\x14\x00".encode() for on, dur, text in annotations
    )
    tal += b"\x00" * (-len(tal) % 2)
    ann = {
        "label": "EDF Annotations",
        "spr": len(tal) // 2,
        "pmin": -1,
        "pmax": 1,
        "dmin": -32768,
        "dmax": 32767,
        "data": np.frombuffer(tal, dtype="<i2"),
    }
    return _edf([ann], 1, 0, start)


def test_reader_round_trip_scales_digital_to_microvolts(rng: np.random.Generator) -> None:
    eeg = rng.integers(-2048, 2048, size=3 * 3000).astype(np.int16)
    other = {
        "label": "Temp rectal",
        "unit": "DegC",
        "spr": 30,
        "pmin": 34,
        "pmax": 40,
        "dmin": -2849,
        "dmax": 2731,
        "data": rng.integers(-2849, 2732, size=90),
    }
    data = _edf([_eeg(eeg, 3), other], 3, 30)
    h = sleepedf.read_header(data)
    assert (h.n_records, h.record_s, h.labels) == (3, 30.0, ("EEG Fpz-Cz", "Temp rectal"))
    assert h.start == "24.04.89 16.13.00"
    x = sleepedf.read_signal(data, h, "EEG Fpz-Cz")  # records interleave EEG and Temp correctly
    assert np.allclose(x, (eeg.astype(float) + 2048) * 384 / 4095 - 192)
    assert x.min() >= -192
    assert x.max() <= 192
    assert np.allclose(
        sleepedf.read_signal(data, h, "Temp rectal"), (other["data"] + 2849) * 6 / 5580 + 34
    )
    with pytest.raises(KeyError):
        sleepedf.read_signal(data, h, "EEG Pz-Oz")
    with pytest.raises(ValueError, match="bytes"):  # truncated data never reads past the end
        sleepedf.read_header(data[:-2])
    with pytest.raises(ValueError, match="integer"):  # header fields are validated, never eval'd
        sleepedf.read_header(data[:236] + b"__import" + data[244:])


def test_tal_parsing_and_epoch_expansion() -> None:
    data = _hypnogram(
        [(0, 60, "Sleep stage W"), (60, 30, "Sleep stage 4"), (90, 30, "Movement time")]
    )
    ann = sleepedf.read_annotations(data, sleepedf.read_header(data))
    assert ann == [
        (0.0, 60.0, "Sleep stage W"),
        (60.0, 30.0, "Sleep stage 4"),
        (90.0, 30.0, "Movement time"),
    ]
    epochs = sleepedf.stage_epochs(ann)
    assert epochs == {
        0: "Sleep stage W",
        1: "Sleep stage W",
        2: "Sleep stage 4",
        3: "Movement time",
    }
    with pytest.raises(ValueError, match="grid"):
        sleepedf.stage_epochs([(15.0, 30.0, "Sleep stage 2")])
    with pytest.raises(ValueError, match="unknown"):
        sleepedf.stage_epochs([(0.0, 30.0, "Lights off")])
    bad = _hypnogram([(0, 30, "Sleep stage W")]).replace(b"+0\x1530", b"x0\x1530")
    with pytest.raises(ValueError, match="TAL"):
        sleepedf.read_annotations(bad, sleepedf.read_header(bad))


def _write_night(
    raw, name: str, n_epochs: int, stages: list[tuple[int, int, str]], rng
) -> np.ndarray:
    # about ±94 µV: full-scale white noise would overshoot ±250 µV once resampled
    eeg = rng.integers(-1000, 1000, size=n_epochs * 3000).astype(np.int16)
    (raw / f"sleep-cassette/{name}0-PSG.edf").write_bytes(_edf([_eeg(eeg, n_epochs)], n_epochs, 30))
    (raw / f"sleep-cassette/{name}C-Hypnogram.edf").write_bytes(_hypnogram(stages))
    return (eeg.astype(float) + 2048) * 384 / 4095 - 192


def test_windowset_from_fake_nights(tmp_path, rng: np.random.Generator) -> None:
    (tmp_path / "sleep-cassette").mkdir()
    m = sleepedf.MARGIN_EPOCHS
    # 70 wake, then W? N1 N2 N3(3) N3(4) REM Movement ?, 70 wake; scored past the signal's end
    sleep = [
        (1, "Sleep stage 1"),
        (2, "Sleep stage 2"),
        (3, "Sleep stage 3"),
        (4, "Sleep stage 4"),
        (5, "Sleep stage R"),
        (6, "Movement time"),
        (7, "Sleep stage ?"),
        (8, "Sleep stage 2"),
    ]
    stages = [(0, 30 * 70, "Sleep stage W")]
    stages += [(30 * (69 + k), 30, t) for k, t in sleep]
    stages += [(30 * 78, 30 * 80, "Sleep stage W")]
    phys = _write_night(tmp_path, "SC4001E", 150, stages, rng)
    _write_night(tmp_path, "SC4002E", 150, stages, rng)  # second night of subject 00
    _write_night(tmp_path, "SC4011E", 200, stages, rng)
    names = [
        "SC4001E0-PSG",
        "SC4001EC-Hypnogram",
        "SC4002E0-PSG",
        "SC4002EC-Hypnogram",
        "SC4011E0-PSG",
        "SC4011EC-Hypnogram",
        "SC4021E0-PSG",
        "SC4021EC-Hypnogram",
    ]
    sums = [f"{'0' * 64} sleep-cassette/{n}.edf" for n in names]
    sums += [f"{'0' * 64} sleep-cassette/../../etc/passwd-PSG.edf", f"{'0' * 64} RECORDS"]
    (tmp_path / sleepedf.SUMS).write_text("\n".join(sums) + "\n")

    ws, report = sleepedf.build_windowset(tmp_path)

    first, last = 70, 77  # sleep epochs; kept range is [first - 60, last + 60] = [10, 137]
    per_night = (last + m) - (first - m) + 1 - 2  # Movement time and ? dropped
    assert ws.window_n == 3072
    assert ws.window_n % 32 == 0
    assert (ws.sample_rate_hz, ws.dataset_id) == (102.4, "sleepedf-v2")
    assert len(ws) == 3 * per_night
    assert set(ws.condition.tolist()) == {"night"}
    assert sorted(set(ws.bearing_id.tolist())) == ["S00", "S01"]  # two nights of S00 share a unit
    assert report["windows_per_class"] == {
        "W": 3 * (per_night - 6),
        "N1": 3,
        "N2": 6,
        "N3": 6,
        "REM": 3,
    }
    assert report["dropped_epochs"] == {"Movement time": 3, "Sleep stage ?": 3}
    assert report["skipped_nights"] == {
        "missing": ["SC4021"],
        "channel_missing": [],
        "unreadable": [],
    }
    assert (report["n_subjects"], report["n_nights"], report["n_catalogued"]) == (2, 3, 4)
    assert report["clipped_fraction"] == 0.0
    # window 0 is epoch 10 of SC4001: the whole night resampled 100 → 102.4 Hz, then cut
    expected, _ = quantise(resample_poly(phys, 128, 125)[10 * 3072 : 11 * 3072], 250.0)
    assert np.array_equal(ws.x[0], expected)
    assert ws.label[: 70 - 10].tolist() == [0] * 60
    assert ws.label[60:65].tolist() == [1, 2, 3, 3, 4]


def test_epochs_past_the_signal_end_are_dropped(rng: np.random.Generator) -> None:
    stages = [
        (0, 30 * 10, "Sleep stage W"),
        (300, 30, "Sleep stage 2"),
        (330, 30 * 100, "Sleep stage W"),
    ]
    eeg = rng.integers(-2048, 2048, size=20 * 3000).astype(np.int16)
    x, y, dropped = sleepedf.night_epochs(_edf([_eeg(eeg, 20)], 20, 30), _hypnogram(stages))
    assert x.shape == (20, 3072)
    assert y.tolist() == [0] * 10 + [2] + [0] * 9
    assert dropped == {"beyond_signal": 60 + 10 + 1 - 20}
    with pytest.raises(ValueError, match="starts"):
        sleepedf.night_epochs(_edf([_eeg(eeg, 20)], 20, 30), _hypnogram(stages, "24.04.8916.14.00"))
