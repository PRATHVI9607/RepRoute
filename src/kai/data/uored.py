"""UORED-VAFCLS bearing dataset (transfer dataset): MAT recordings → int16 WindowSet.

Lab-plane only (uses scipy.io). Sehri, Dumond & Bouchard, "University of Ottawa constant load and
speed rolling-element bearing vibration and acoustic fault signature datasets", Data in Brief 49
(2023) 109327; Mendeley Data v5, doi:10.17632/y2px5tg92h.5, CC BY 4.0. A recording
`<type>_<bearing>_<state>.mat` holds one variable named like the file stem: 420 000 × 5 float64
(10 s at 42 kHz), columns accelerometer [m/s²], microphone, speed [rpm], load [N], temperature.
Speed and load are written in row 0 only. Bearings 1–5 are inner-race, 6–10 outer-race, 11–15 ball,
16–20 cage runs to failure; state 0 healthy, 1 developing fault, 2 faulty. Each run is one
physical bearing, so its healthy and faulty recordings share a bearing_id.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kai.contracts.windows import WindowSet
from kai.data.pu import quantise, segment

SAMPLE_RATE_HZ = 42_000.0
DATASET_ID = "uored-v1"
FAULT_TYPES = "IOBC"  # inner race, outer race, ball, cage: bearings 1-5, 6-10, 11-15, 16-20
# measured peak |accelerometer| over the 30 used files is 824.6 m/s² (I_5_2), so this clips nothing
FULL_SCALE = 825.0


def bearing_id(n: int) -> str:
    return f"{FAULT_TYPES[(n - 1) // 5]}{n:02d}"


# fault-type stratum of each physical bearing (for make_l1_splits: a bearing has two labels)
STRATA = {bearing_id(n): (n - 1) // 5 for n in range(1, 21)}
# every published recording; "developing fault" (state 1) and ball / cage faulty ones are unused
ALL_FILES = [f"H_{n}_0.mat" for n in range(1, 21)] + [
    f"{bearing_id(n)[0]}_{n}_{state}.mat" for n in range(1, 21) for state in (1, 2)
]
# file name → (bearing_id, label): healthy state of all 20 → 0, faulty inner → 1, faulty outer → 2
RECORDINGS: dict[str, tuple[str, int]] = {
    **{f"H_{n}_0.mat": (bearing_id(n), 0) for n in range(1, 21)},
    **{f"I_{n}_2.mat": (bearing_id(n), 1) for n in range(1, 6)},
    **{f"O_{n}_2.mat": (bearing_id(n), 2) for n in range(6, 11)},
}


def build_windowset(
    raw_dir: Path, window_n: int = 4096, full_scale: float = FULL_SCALE
) -> tuple[WindowSet, dict[str, float | list[str]]]:
    """Quantise and window the accelerometer of the 3-class recordings in `raw_dir`."""
    xs, bids, conds, labels = [], [], [], []
    clipped = total = 0
    skipped: list[str] = []
    dropped: list[str] = []
    peak = 0.0
    for path in sorted(raw_dir.glob("*.mat")):
        if path.name not in RECORDINGS:  # developing fault, ball and cage faults: no label
            dropped.append(path.name)
            continue
        bearing, label = RECORDINGS[path.name]
        try:
            table = np.asarray(loadmat(path)[path.stem], dtype=np.float64)
            signal, load_n = table[:, 0], table[0, 3]
        except Exception:  # an unreadable recording is counted, never hidden
            skipped.append(path.name)
            continue
        samples, c = quantise(signal, full_scale)
        clipped += c
        total += len(samples)
        peak = max(peak, float(np.abs(signal).max()))
        w = segment(samples, window_n)
        xs.append(w)
        bids += [bearing] * len(w)
        conds += [f"1750rpm_{load_n:.0f}N"] * len(w)  # nominal speed; load as logged in the file
        labels += [label] * len(w)
    x = np.concatenate(xs)
    ws = WindowSet(
        x=x,
        window_id=np.arange(len(x), dtype=np.int64),
        bearing_id=np.asarray(bids),
        condition=np.asarray(conds),
        label=np.asarray(labels, dtype=np.int64),
        sample_rate_hz=SAMPLE_RATE_HZ,
        dataset_id=DATASET_ID,
    )
    report: dict[str, float | list[str]] = {
        "n_windows": float(len(x)),
        "n_bearings": float(len(set(bids))),
        "n_recordings": float(len(xs)),
        "skipped_recordings": skipped,
        "dropped_files": dropped,
        "clipped_fraction": clipped / max(total, 1),
        "peak_abs": peak,
        "full_scale": full_scale,
    }
    return ws, report
