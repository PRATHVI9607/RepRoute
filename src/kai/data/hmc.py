"""HMC (Haaglanden Medisch Centrum) sleep staging v1.1, ADR 0028 §4: EDF → WindowSet.

Alvarez-Estevez & Rijsman, "Haaglanden Medisch Centrum sleep staging database", PhysioNet v1.1
(CC BY 4.0); 151 subjects, one night each (`recordings/SNxxx.edf`), with a hypnogram scored in
30-s epochs (`SNxxx_sleepscoring.edf`, EDF+ annotations; a `.txt` CSV is the fallback). The unit
("bearing") is the subject `SNxxx`. One window per scored epoch of `EEG C4-M1` (256 Hz); the WHOLE
night is resampled by 2/5 to 102.4 Hz first and then cut into 3072-sample epochs on the 30-s grid
(no per-epoch edge effects, unlike the Sleep-EDF v1 build, which stays as it is). Quantised to
int16 at a fixed ±250 µV full scale. Labels W / N1 / N2 / N3 / REM; the zero-duration "Lights
off" / "Lights on" markers are skipped, any other non-stage annotation is an error. 30 min of wake
is kept on each side of the sleep period. EDF headers and annotations are untrusted data.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

from kai.contracts.windows import WindowSet
from kai.data.pu import quantise
from kai.data.sleepedf import read_annotations, read_header, read_signal, read_sums

SOURCE_RATE_HZ = 256.0
UP, DOWN = 2, 5
SAMPLE_RATE_HZ = SOURCE_RATE_HZ * UP / DOWN  # 102.4
EPOCH_S = 30
EPOCH_N = int(EPOCH_S * SAMPLE_RATE_HZ)  # 3072
MARGIN_EPOCHS = 60
DATASET_ID = "hmc-v1"
CHANNEL = "EEG C4-M1"
CONDITION = "night"
FULL_SCALE_UV = 250.0
LABELS = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
STAGES = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}
MARKERS = ("Lights off", "Lights on")
_FILE = re.compile(r"recordings/(SN\d{3})(_sleepscoring)?\.(edf|txt)")
_FLOAT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)")


def stage_epochs(annotations: Iterable[tuple[float, float, str]]) -> dict[int, int]:
    """30-s epoch index (from the recording start) → stage label. Raises ValueError."""
    epochs: dict[int, int] = {}
    for onset_s, duration_s, text in annotations:
        if text in MARKERS:  # zero-duration, off the 30-s grid
            continue
        if text not in STAGES:
            raise ValueError(f"unknown annotation {text!r}")
        if onset_s % EPOCH_S or duration_s % EPOCH_S or onset_s < 0:
            raise ValueError(f"{text} at {onset_s} s for {duration_s} s is off the 30-s grid")
        first = int(onset_s) // EPOCH_S
        for k in range(first, first + int(duration_s) // EPOCH_S):
            if epochs.setdefault(k, STAGES[text]) != STAGES[text]:
                raise ValueError(f"epoch {k} scored twice")
    return epochs


def read_scoring_csv(text: str) -> list[tuple[float, float, str]]:
    """Fallback hypnogram: rows `onset_s, duration_s, annotation`; header rows are skipped."""
    return [
        (float(r[0]), float(r[1]), r[2].strip())
        for r in csv.reader(io.StringIO(text))
        if len(r) >= 3 and _FLOAT.fullmatch(r[0].strip()) and _FLOAT.fullmatch(r[1].strip())
    ]


def catalogue(names: Iterable[str]) -> dict[str, tuple[str, str]]:
    """Subject id (SNxxx) → (recording path, hypnogram path); the EDF hypnogram wins over `.txt`."""
    rec: dict[str, str] = {}
    hyp: dict[str, dict[str, str]] = {}
    for name in sorted(names):
        if m := _FILE.fullmatch(name):
            sn, scoring, ext = m.groups()
            if scoring:
                hyp.setdefault(sn, {})[ext] = name
            elif ext == "edf":
                rec[sn] = name
    return {sn: (rec[sn], hyp[sn].get("edf") or hyp[sn]["txt"]) for sn in sorted(rec) if sn in hyp}


def night_epochs(
    psg: bytes, hyp_name: str, hyp: bytes
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], Counter[str]]:
    """Kept epochs of one night: [n, 3072] µV at 102.4 Hz, labels, dropped counts.
    Raises KeyError if the channel is absent and ValueError on anything unexpected."""
    hp = read_header(psg)
    i = hp.labels.index(CHANNEL) if CHANNEL in hp.labels else -1
    if i < 0:
        raise KeyError(CHANNEL)
    if hp.units[i] != "uV" or hp.samples_per_record[i] != hp.record_s * SOURCE_RATE_HZ:
        raise ValueError(f"{CHANNEL}: {hp.units[i]}, {hp.samples_per_record[i]}/{hp.record_s} s")
    if hyp_name.endswith(".edf"):
        hh = read_header(hyp)
        if hp.start != hh.start:  # onsets count from the recording start
            raise ValueError(f"PSG starts {hp.start}, hypnogram {hh.start}")
        ann = read_annotations(hyp, hh)
    else:
        ann = read_scoring_csv(hyp.decode("utf-8"))
    epochs = stage_epochs(ann)
    sleep = [k for k, v in epochs.items() if v > 0]
    if not sleep:
        raise ValueError("no sleep epoch")
    x = resample_poly(read_signal(psg, hp, CHANNEL), UP, DOWN)  # whole night, then cut
    n_signal = len(x) // EPOCH_N
    dropped: Counter[str] = Counter()
    kept: list[int] = []
    for k in range(max(0, min(sleep) - MARGIN_EPOCHS), max(sleep) + MARGIN_EPOCHS + 1):
        if k not in epochs:
            dropped["unannotated"] += 1
        elif k >= n_signal:
            dropped["beyond_signal"] += 1
        else:
            kept.append(k)
    windows = x[: n_signal * EPOCH_N].reshape(n_signal, EPOCH_N)[np.asarray(kept, dtype=np.int64)]
    return windows, np.asarray([epochs[k] for k in kept], dtype=np.int64), dropped


def build_windowset(
    raw_dir: Path, full_scale_uv: float = FULL_SCALE_UV
) -> tuple[WindowSet, dict[str, object]]:
    """Every catalogued subject found under `raw_dir` (SHA256SUMS.txt included)."""
    sums = read_sums(raw_dir)
    subjects = catalogue(sums)
    xs, bids, labels = [], [], []
    dropped: Counter[str] = Counter()
    skipped: dict[str, list[str]] = {
        k: [] for k in ("missing", "corrupt", "channel_missing", "unreadable")
    }
    clipped = 0
    for sn, (psg, hyp) in subjects.items():
        if not (raw_dir / psg).exists() or not (raw_dir / hyp).exists():
            skipped["missing"].append(sn)
            continue
        psg_b, hyp_b = (raw_dir / psg).read_bytes(), (raw_dir / hyp).read_bytes()
        # hash the bytes that are parsed (no read-after-check gap); mismatch = corrupt, skipped
        if any(hashlib.sha256(b).hexdigest() != sums[n] for b, n in ((psg_b, psg), (hyp_b, hyp))):
            skipped["corrupt"].append(sn)
            continue
        try:
            epochs, y, d = night_epochs(psg_b, hyp, hyp_b)
        except KeyError:
            skipped["channel_missing"].append(sn)
            continue
        except (ValueError, UnicodeDecodeError):  # malformed file: counted, never hidden
            skipped["unreadable"].append(sn)
            continue
        q, c = quantise(epochs, full_scale_uv)
        clipped += c
        dropped += d
        xs.append(q)
        bids += [sn] * len(q)
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
    rails = np.abs(x) == np.iinfo(np.int16).max
    report: dict[str, object] = {
        "n_windows": len(x),
        "n_subjects": len(set(bids)),
        "n_nights": len(xs),
        "windows_per_class": {c: int((y == k).sum()) for c, k in LABELS.items()},
        "n_catalogued": len(subjects),
        "dropped_epochs": dict(dropped),
        "skipped_nights": skipped,
        "clipped_samples": clipped,
        "clipped_fraction": clipped / x.size,
        "windows_with_clipping": int(rails.any(axis=1).sum()),
        "peak_abs": int(np.abs(x.astype(np.int32)).max()),
        "window_n": int(x.shape[1]),
        "channel": CHANNEL,
        "full_scale_uv": full_scale_uv,
        "sample_rate_hz": SAMPLE_RATE_HZ,
    }
    return ws, report
