"""Icentia11k single-lead ambulatory ECG (transfer dataset): WFDB segments → balanced WindowSet.

Tan et al., "Icentia11k Single Lead Continuous Raw Electrocardiogram Dataset", PhysioNet v1.0
(2022), doi:10.13026/kk0v-r952, CC BY-NC-SA 4.0 (research use, never commit the data). 11,000
patients, ~50 segments of ~70 min each, 250 Hz, one lead, WFDB format 16, beat annotations in
MIT format (.atr). Only the first segment of patients p00000..p00999 is used. The "bearing" is
the patient. Windows are non-overlapping `window_n` samples from the start of the segment;
label 2 if the window holds a V beat, else 1 if it holds an S beat, else 0 (N beats only);
windows with a Q beat or no beat are dropped. Per patient at most `max_per_class` windows of
each class are kept, so a patient may contribute to several classes. Headers and annotation
files are untrusted data: parsed and checked, never executed.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kai.contracts.windows import WindowSet
from kai.data.ptbxl import read_lead

SAMPLE_RATE_HZ = 250.0
DATASET_ID = "icentia-v1"
CONDITION = "ambulatory"
LEAD = "ecg"
SUMS = "SHA256SUMS.txt"
N_PATIENTS = 1000
LABELS = {"N": 0, "S": 1, "V": 2}
CODE = {"N": 1, "V": 5, "S": 9, "Q": 13}  # WFDB annotation codes of the four Icentia11k beats
_SKIP, _NUM, _SUB, _CHN, _AUX = 59, 60, 61, 62, 63
# p00/p00000/p00000_s00: group dir, patient, segment; the backreferences tie the three together
_RECORD = re.compile(r"p(\d\d)/(p\1\d{3})/\2_s(\d\d)")


def parse_atr(buf: bytes) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """(sample, code) of every annotation in an MIT-format annotation file.

    Each 16-bit little-endian word is a 6-bit code and a 10-bit sample delta. SKIP adds a 32-bit
    signed interval (high word first) to the time; NUM / SUB / CHN / AUX attach fields to the
    previous annotation and are consumed (AUX skips its padded byte string). A zero word ends it.
    Raises ValueError on truncated fields or a negative sample."""
    if len(buf) % 2:
        raise ValueError(f"annotation file has an odd length {len(buf)}")
    w = np.frombuffer(buf, dtype="<u2").tolist()
    samples: list[int] = []
    codes: list[int] = []
    t, i = 0, 0
    while i < len(w):
        code, d = w[i] >> 10, w[i] & 0x3FF
        i += 1
        if code == 0 and d == 0:
            break
        if code == _SKIP:
            if i + 2 > len(w):
                raise ValueError("truncated SKIP")
            skip = (w[i] << 16) | w[i + 1]
            t += skip - (1 << 32) if skip >= 1 << 31 else skip
            i += 2
        elif code == _AUX:
            i += (d + 1) // 2
            if i > len(w):
                raise ValueError("truncated AUX")
        elif code not in (_NUM, _SUB, _CHN):
            t += d
            if t < 0:
                raise ValueError(f"annotation at negative sample {t}")
            samples.append(t)
            codes.append(code)
    return np.asarray(samples, dtype=np.int64), np.asarray(codes, dtype=np.int64)


def label_windows(
    samples: npt.NDArray[np.int64], codes: npt.NDArray[np.int64], n_samples: int, window_n: int
) -> npt.NDArray[np.int64]:
    """Label of each full non-overlapping window from sample 0: 2 = holds a V beat, else 1 = an S
    beat, else 0 = N beats only; -1 = holds a Q beat or no beat (dropped)."""
    n_win = n_samples // window_n
    inside = samples < n_win * window_n
    k, c = samples[inside] // window_n, codes[inside]

    def has(name: str) -> npt.NDArray[np.bool_]:
        return np.bincount(k[c == CODE[name]], minlength=n_win) > 0

    y = np.full(n_win, -1, dtype=np.int64)
    for name in ("N", "S", "V"):  # later assignments win: V over S over N
        y[has(name)] = LABELS[name]
    y[has("Q")] = -1
    return y


def cap_per_class(
    y: npt.NDArray[np.int64], max_per_class: int, rng: np.random.Generator
) -> npt.NDArray[np.intp]:
    """Ascending indices of at most `max_per_class` random windows of each class (y ≥ 0)."""
    picks = [np.flatnonzero(y == c) for c in sorted(LABELS.values())]
    keep = [rng.choice(p, size=min(max_per_class, len(p)), replace=False) for p in picks]
    return np.sort(np.concatenate(keep)).astype(np.intp)


def catalogue(sum_names: list[str], n_patients: int = N_PATIENTS) -> dict[str, str]:
    """record path → bearing_id ("P<patient>") for the first segment of each of the first
    `n_patients` patients with .hea, .dat and .atr listed. Unmatched names never become paths."""
    names = set(sum_names)
    first: dict[int, tuple[int, str]] = {}
    for name in sorted(names):
        rec = name.removesuffix(".hea")
        m = _RECORD.fullmatch(rec)
        if rec == name or m is None or not {f"{rec}.dat", f"{rec}.atr"} <= names:
            continue
        patient, segment = int(m.group(2)[1:]), int(m.group(3))
        if patient < n_patients and (patient not in first or segment < first[patient][0]):
            first[patient] = (segment, rec)
    return {rec: f"P{p}" for p, (_, rec) in sorted(first.items())}


def read_sums(raw_dir: Path) -> dict[str, str]:
    """file name → SHA-256 from the mirror's SHA256SUMS.txt."""
    lines = (raw_dir / SUMS).read_text().split("\n")
    return {name: digest for digest, name in (s.split(maxsplit=1) for s in lines if s)}


def split_labels(ws: WindowSet) -> dict[str, int]:
    """bearing_id → highest class among its windows. Only for stratifying patient-level splits:
    unlike the other datasets a patient here carries windows of several classes."""
    out: dict[str, int] = {}
    for b, y in zip(ws.bearing_id.tolist(), ws.label.tolist(), strict=True):
        out[b] = max(out.get(b, y), y)
    return out


def build_windowset(
    raw_dir: Path,
    window_n: int = 4096,
    max_per_class: int = 6,
    seed: int = 20261012,
    n_patients: int = N_PATIENTS,
) -> tuple[WindowSet, dict[str, object]]:
    """Balanced windows of the catalogued segments found under `raw_dir` (see the module doc)."""
    records = catalogue(list(read_sums(raw_dir)), n_patients)
    xs: list[npt.NDArray[np.int16]] = []
    bids: list[str] = []
    labels: list[npt.NDArray[np.int64]] = []
    skipped: dict[str, list[str]] = {k: [] for k in ("missing", "unreadable", "too_short")}
    available: Counter[str] = Counter()
    seen_codes: Counter[int] = Counter()
    patients_per_class: Counter[str] = Counter()
    for rec, patient in records.items():
        if not all((raw_dir / f"{rec}{ext}").exists() for ext in (".hea", ".dat", ".atr")):
            skipped["missing"].append(rec)
            continue
        try:
            fs = (raw_dir / f"{rec}.hea").read_text().split(maxsplit=3)[2].split("/")[0]
            if float(fs) != SAMPLE_RATE_HZ:
                raise ValueError(f"{rec}: sample rate {fs}")
            x = read_lead(raw_dir / rec, LEAD)
            samples, codes = parse_atr((raw_dir / f"{rec}.atr").read_bytes())
        except (KeyError, ValueError, IndexError):  # malformed header, data or annotations: counted
            skipped["unreadable"].append(rec)
            continue
        if len(x) < window_n:
            skipped["too_short"].append(rec)
            continue
        seen_codes.update(codes.tolist())
        y = label_windows(samples, codes, len(x), window_n)
        available.update({"dropped_q_or_no_beat": int((y < 0).sum())})
        available.update({c: int((y == k).sum()) for c, k in LABELS.items()})
        idx = cap_per_class(y, max_per_class, np.random.default_rng([seed, int(patient[1:])]))
        if not len(idx):
            continue
        xs.extend(x[i * window_n : (i + 1) * window_n] for i in idx)
        bids.extend([patient] * len(idx))
        labels.append(y[idx])
        patients_per_class.update(c for c, k in LABELS.items() if (y[idx] == k).any())
    x_all = np.stack(xs)
    y_all = np.concatenate(labels)
    ws = WindowSet(
        x=x_all,
        window_id=np.arange(len(x_all), dtype=np.int64),
        bearing_id=np.asarray(bids),
        condition=np.full(len(x_all), CONDITION),
        label=y_all,
        sample_rate_hz=SAMPLE_RATE_HZ,
        dataset_id=DATASET_ID,
    )
    rails = (x_all == np.iinfo(np.int16).min) | (x_all == np.iinfo(np.int16).max)
    strata = Counter(split_labels(ws).values())
    report: dict[str, object] = {
        "n_windows": len(x_all),
        "n_patients": len(set(bids)),
        "windows_per_class": {c: int((y_all == k).sum()) for c, k in LABELS.items()},
        "patients_per_class": {c: patients_per_class[c] for c in LABELS},
        "patients_by_split_stratum": {c: strata[k] for c, k in LABELS.items()},
        "windows_before_cap": dict(available),
        "annotation_codes": {str(k): v for k, v in sorted(seen_codes.items())},
        "n_catalogued": len(records),
        "skipped_records": skipped,
        "clipped_fraction": float(rails.mean()),
        "windows_with_clipping": int(rails.any(axis=1).sum()),
        "min": int(x_all.min()),
        "max": int(x_all.max()),
        "peak_abs": int(np.abs(x_all.astype(np.int32)).max()),
        "window_n": window_n,
        "max_per_class": max_per_class,
        "seed": seed,
    }
    return ws, report
