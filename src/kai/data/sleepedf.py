"""Sleep-EDF Expanded, Sleep Cassette (transfer dataset, other sensor domain): EDF → WindowSet.

Kemp et al., "Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity
of the EEG", IEEE Trans Biomed Eng 47(9), 1185-1194 (2000); Goldberger et al., "PhysioBank,
PhysioToolkit, and PhysioNet", Circulation 101(23), e215-e220 (2000). PhysioNet sleep-edfx v1.0.0,
Open Data Commons Attribution License v1.0. The Sleep Cassette study: 153 day-night PSG recordings
(`SC4ssN..-PSG.edf`) of 78 subjects, each with a hypnogram scored by hand in 30-s epochs
(`SC4ssN..-Hypnogram.edf`, EDF+ annotations). Here the "bearing" is the subject (`S<ss>`): a
subject's one or two nights share one unit, so splits leave subjects out.
One window per scored 30-s epoch of `EEG Fpz-Cz` (100 Hz), resampled 3000 → 3072 samples
(102.4 Hz) and quantised to int16 with a fixed ±250 µV full scale. Labels W / N1 / N2 / N3 (R&K
stages 3 and 4 merged) / REM; "Movement time" and "Sleep stage ?" epochs are dropped. Only epochs
from 30 min before the first sleep epoch to 30 min after the last are kept (the recordings hold
hours of wake). EDF headers and annotations are untrusted data: parsed, validated, never executed.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

from kai.contracts.windows import WindowSet
from kai.data.pu import quantise

SOURCE_RATE_HZ = 100.0
UP, DOWN = 128, 125  # 3000 samples per epoch → 3072, divisible by 32 for the wavelet
SAMPLE_RATE_HZ = SOURCE_RATE_HZ * UP / DOWN  # 102.4
EPOCH_S = 30
EPOCH_N = int(EPOCH_S * SOURCE_RATE_HZ)
MARGIN_EPOCHS = 60  # 30 min of wake kept on each side of the sleep period
DATASET_ID = "sleepedf-v2"  # v2: the whole night is resampled before cutting (ADR 0028 fix 18)
CHANNEL = "EEG Fpz-Cz"
CONDITION = "night"
# the EEG channels' physical range is about ±200 µV, so only resampling overshoot can clip
FULL_SCALE_UV = 250.0
LABELS = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
STAGES = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
UNSCORED = ("Movement time", "Sleep stage ?")
SUMS = "SHA256SUMS.txt"
_FILE = re.compile(r"sleep-cassette/(SC4(\d\d)[12])[0-9A-Z]{2}-(PSG|Hypnogram)\.edf")
_INT = re.compile(r"[+-]?\d+")
_NUM = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_ONSET = re.compile(r"[+-]\d+(?:\.\d*)?")
_DURATION = re.compile(r"\d+(?:\.\d*)?")
_SIGNAL_FIELDS = (  # EDF per-signal header: (name, width in bytes), each stored for all signals
    ("label", 16),
    ("transducer", 80),
    ("unit", 8),
    ("phys_min", 8),
    ("phys_max", 8),
    ("dig_min", 8),
    ("dig_max", 8),
    ("prefilter", 80),
    ("samples_per_record", 8),
    ("reserved", 32),
)


@dataclass(frozen=True)
class EdfHeader:
    start: str  # "dd.mm.yy hh.mm.ss" as written; compared between files, never parsed into a clock
    header_bytes: int
    n_records: int
    record_s: float
    labels: tuple[str, ...]
    units: tuple[str, ...]
    samples_per_record: tuple[int, ...]
    phys_min: tuple[float, ...]
    phys_max: tuple[float, ...]
    dig_min: tuple[int, ...]
    dig_max: tuple[int, ...]


def _int(s: str) -> int:
    if not _INT.fullmatch(s.strip()):
        raise ValueError(f"not an integer: {s!r}")
    return int(s)


def _num(s: str) -> float:
    if not _NUM.fullmatch(s.strip()):
        raise ValueError(f"not a number: {s!r}")
    return float(s)


def read_header(data: bytes) -> EdfHeader:
    """Parse and validate the EDF/EDF+ header of a whole file's bytes. Raises ValueError."""
    if len(data) < 256 or data[:8] != b"0       ":
        raise ValueError("not an EDF file")
    head = data[:256].decode("ascii")  # EDF headers are ASCII; anything else raises
    header_bytes, n_records, n_sig = _int(head[184:192]), _int(head[236:244]), _int(head[252:256])
    record_s = _num(head[244:252])
    if not 0 < n_sig <= 1024 or header_bytes != 256 * (n_sig + 1) or len(data) < header_bytes:
        raise ValueError(f"bad header: {n_sig} signals, {header_bytes} header bytes")
    text = data[256:header_bytes].decode("ascii")
    cols: dict[str, list[str]] = {}
    pos = 0
    for name, width in _SIGNAL_FIELDS:
        cols[name] = [text[pos + i * width : pos + (i + 1) * width].strip() for i in range(n_sig)]
        pos += width * n_sig
    h = EdfHeader(
        start=f"{head[168:176]} {head[176:184]}",
        header_bytes=header_bytes,
        n_records=n_records,
        record_s=record_s,
        labels=tuple(cols["label"]),
        units=tuple(cols["unit"]),
        samples_per_record=tuple(_int(s) for s in cols["samples_per_record"]),
        phys_min=tuple(_num(s) for s in cols["phys_min"]),
        phys_max=tuple(_num(s) for s in cols["phys_max"]),
        dig_min=tuple(_int(s) for s in cols["dig_min"]),
        dig_max=tuple(_int(s) for s in cols["dig_max"]),
    )
    record_bytes = 2 * sum(h.samples_per_record)
    if min(h.samples_per_record) < 1 or n_records < 1 or not 0 <= record_s < 1e6:
        raise ValueError(f"bad header: {n_records} records of {record_s} s")
    if len(data) != header_bytes + n_records * record_bytes:
        raise ValueError(
            f"{len(data)} bytes, header says {header_bytes} + {n_records}×{record_bytes}"
        )
    if any(lo >= hi for lo, hi in zip(h.dig_min, h.dig_max, strict=True)):
        raise ValueError("digital minimum not below digital maximum")
    if any(lo == hi for lo, hi in zip(h.phys_min, h.phys_max, strict=True)):
        raise ValueError("physical minimum equals physical maximum")
    return h


def _samples(data: bytes, h: EdfHeader, i: int) -> npt.NDArray[np.uint8]:
    """Raw bytes of signal i, records concatenated, as [n_records, 2·samples_per_record]."""
    start = 2 * sum(h.samples_per_record[:i])
    width = 2 * h.samples_per_record[i]
    body = np.frombuffer(data, dtype=np.uint8, offset=h.header_bytes)
    return body.reshape(h.n_records, -1)[:, start : start + width]


def read_signal(data: bytes, h: EdfHeader, label: str) -> npt.NDArray[np.float64]:
    """One signal in physical units (linear map of the digital range onto the physical range).
    Raises KeyError if the label is absent."""
    if label not in h.labels:
        raise KeyError(f"no signal {label} in {h.labels}")
    i = h.labels.index(label)
    d = np.ascontiguousarray(_samples(data, h, i)).view("<i2").ravel().astype(np.float64)
    gain = (h.phys_max[i] - h.phys_min[i]) / (h.dig_max[i] - h.dig_min[i])
    return (d - h.dig_min[i]) * gain + h.phys_min[i]


def read_annotations(data: bytes, h: EdfHeader) -> list[tuple[float, float, str]]:
    """EDF+ TALs of every "EDF Annotations" signal: (onset_s, duration_s, text), in file order.
    A TAL is "+onset[\\x15duration]\\x14text\\x14[text\\x14…]\\x00"; empty texts (the
    time-keeping TAL of each record) are skipped."""
    out: list[tuple[float, float, str]] = []
    for i, label in enumerate(h.labels):
        if label != "EDF Annotations":
            continue
        for tal in _samples(data, h, i).tobytes().split(b"\x00"):
            if not tal:
                continue
            parts = tal.decode("utf-8").split("\x14")
            onset, _, duration = parts[0].partition("\x15")
            if len(parts) < 2 or parts[-1] or not _ONSET.fullmatch(onset):
                raise ValueError(f"malformed TAL {tal[:40]!r}")
            if duration and not _DURATION.fullmatch(duration):
                raise ValueError(f"malformed duration in TAL {tal[:40]!r}")
            out += [(float(onset), float(duration or 0), t) for t in parts[1:-1] if t]
    return out


def stage_epochs(annotations: Iterable[tuple[float, float, str]]) -> dict[int, str]:
    """30-s epoch index (from the hypnogram start) → stage annotation text."""
    epochs: dict[int, str] = {}
    for onset_s, duration_s, text in annotations:
        if text not in STAGES and text not in UNSCORED:
            raise ValueError(f"unknown annotation {text!r}")
        if onset_s % EPOCH_S or duration_s % EPOCH_S or onset_s < 0:
            raise ValueError(f"{text} at {onset_s} s for {duration_s} s is off the 30-s grid")
        first = int(onset_s) // EPOCH_S
        for k in range(first, first + int(duration_s) // EPOCH_S):
            if epochs.setdefault(k, text) != text:
                raise ValueError(f"epoch {k} scored twice")
    return epochs


def catalogue(names: Iterable[str]) -> dict[str, tuple[str, str, str]]:
    """Night id (SC4ssN) → (bearing_id, PSG path, Hypnogram path) for the Sleep Cassette files in
    `names` (paths relative to the dataset root); names not in the published layout are ignored."""
    files: dict[tuple[str, str], str] = {}
    subject: dict[str, str] = {}
    for name in sorted(names):
        if m := _FILE.fullmatch(name):
            night, ss, kind = m.groups()
            if files.setdefault((night, kind), name) != name:
                raise ValueError(f"two {kind} files for night {night}")
            subject[night] = f"S{ss}"
    return {
        night: (subject[night], files[(night, "PSG")], files[(night, "Hypnogram")])
        for night in sorted(subject)
        if (night, "PSG") in files and (night, "Hypnogram") in files
    }


def read_sums(raw_dir: Path) -> dict[str, str]:
    """name → SHA-256 from the mirror's SHA256SUMS.txt."""
    lines = (raw_dir / SUMS).read_text().splitlines()
    return {name: digest for digest, name in (s.split(maxsplit=1) for s in lines if s)}


def night_epochs(
    psg: bytes, hyp: bytes
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], Counter[str]]:
    """Kept 30-s epochs of one night: [n, 3072] µV at 102.4 Hz, labels, and dropped-epoch counts.
    The whole night is resampled once, then cut on the 30-s grid (3000 → 3072 samples exactly), so
    epoch edges carry no filter transients (ADR 0028 fix 18).
    Raises KeyError if the channel is absent and ValueError on anything unexpected."""
    hp, hh = read_header(psg), read_header(hyp)
    if hp.start != hh.start:  # onsets count from the hypnogram start
        raise ValueError(f"PSG starts {hp.start}, hypnogram {hh.start}")
    x = read_signal(psg, hp, CHANNEL)
    i = hp.labels.index(CHANNEL)
    if hp.units[i] != "uV" or hp.samples_per_record[i] != hp.record_s * SOURCE_RATE_HZ:
        raise ValueError(f"{CHANNEL}: {hp.units[i]}, {hp.samples_per_record[i]}/{hp.record_s} s")
    epochs = stage_epochs(read_annotations(hyp, hh))
    sleep = [k for k, t in epochs.items() if STAGES.get(t, 0) > 0]
    if not sleep:
        raise ValueError("no sleep epoch")
    n_signal = len(x) // EPOCH_N
    dropped: Counter[str] = Counter()
    kept: list[int] = []
    for k in range(max(0, min(sleep) - MARGIN_EPOCHS), max(sleep) + MARGIN_EPOCHS + 1):
        text = epochs.get(k)
        if text is None:
            dropped["unannotated"] += 1
        elif text in UNSCORED:
            dropped[text] += 1
        elif k >= n_signal:
            dropped["beyond_signal"] += 1
        else:
            kept.append(k)
    idx = np.asarray(kept, dtype=np.int64)
    night = resample_poly(x[: n_signal * EPOCH_N], UP, DOWN)
    windows = night.reshape(n_signal, EPOCH_N * UP // DOWN)[idx]
    labels = np.asarray([STAGES[epochs[k]] for k in kept], dtype=np.int64)
    return windows, labels, dropped


def build_windowset(
    raw_dir: Path, full_scale_uv: float = FULL_SCALE_UV
) -> tuple[WindowSet, dict[str, object]]:
    """Every catalogued night found under `raw_dir` (the mirror layout, SHA256SUMS.txt included)."""
    nights = catalogue(read_sums(raw_dir))
    xs, bids, labels = [], [], []
    dropped: Counter[str] = Counter()
    skipped: dict[str, list[str]] = {k: [] for k in ("missing", "channel_missing", "unreadable")}
    clipped = 0
    for night, (subject, psg, hyp) in nights.items():
        if not (raw_dir / psg).exists() or not (raw_dir / hyp).exists():
            skipped["missing"].append(night)
            continue
        try:
            epochs, y, d = night_epochs((raw_dir / psg).read_bytes(), (raw_dir / hyp).read_bytes())
        except KeyError:
            skipped["channel_missing"].append(night)
            continue
        except (ValueError, UnicodeDecodeError):  # malformed file: counted, never hidden
            skipped["unreadable"].append(night)
            continue
        q, c = quantise(epochs, full_scale_uv)
        clipped += c
        dropped += d
        xs.append(q)
        bids += [subject] * len(q)
        labels.append(y)
    x = np.concatenate(xs)
    y = np.concatenate(labels)
    ws = WindowSet(
        x=x,
        window_id=np.arange(len(x), dtype=np.int64),
        bearing_id=np.asarray(bids),
        condition=np.full(len(x), CONDITION),
        label=y,
        sample_rate_hz=SAMPLE_RATE_HZ,
        dataset_id=DATASET_ID,
    )
    rails = np.abs(x) == np.iinfo(np.int16).max  # quantise clips to ±32767
    report: dict[str, object] = {
        "n_windows": len(x),
        "n_subjects": len(set(bids)),
        "n_nights": len(xs),
        "windows_per_class": {c: int((y == k).sum()) for c, k in LABELS.items()},
        "n_catalogued": len(nights),
        "dropped_epochs": dict(dropped),
        "skipped_nights": skipped,
        "clipped_samples": clipped,
        "clipped_fraction": clipped / x.size,
        "windows_with_clipping": int(rails.any(axis=1).sum()),
        "min": int(x.min()),
        "max": int(x.max()),
        "peak_abs": int(np.abs(x.astype(np.int32)).max()),
        "window_n": int(x.shape[1]),
        "channel": CHANNEL,
        "full_scale_uv": full_scale_uv,
        "sample_rate_hz": SAMPLE_RATE_HZ,
    }
    return ws, report
