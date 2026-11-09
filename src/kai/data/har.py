"""Smartphone human activity (transfer dataset, sanity check): HAPT raw accelerometer → WindowSet.

Reyes-Ortiz et al., "Transition-Aware Human Activity Recognition Using Smartphones",
Neurocomputing 171 (2016); UCI ML Repository #341 (HAPT), CC BY 4.0. Same 30 subjects and lab as
UCI HAR (Anguita et al., ESANN 2013), but HAPT ships the continuous recordings the HAR windows were
cut from: `RawData/acc_expXX_userYY.txt` (50 Hz, 3-axis, g) and `RawData/labels.txt` (experiment,
user, activity, start, end; 1-based inclusive sample indices). Here the "bearing" is the user.
Activities 1-6 only (WALKING … LAYING → 0..5); postural transitions 7-12 are dropped. The signal
is the raw acceleration magnitude in g (gravity kept), cut into non-overlapping `window_n`-sample
windows lying fully inside one labelled interval, quantised to int16 at a fixed 4 g full scale.
Files are untrusted data: parsed as numbers, paths built only from the parsed integers.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kai.contracts.windows import WindowSet

SAMPLE_RATE_HZ = 50.0
DATASET_ID = "har-v1"
CONDITION = "lab"
FULL_SCALE_G = 4.0
COUNTS_PER_G = np.iinfo(np.int16).max / FULL_SCALE_G
ACTIVITIES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
LABELS = {name: a - 1 for a, name in ACTIVITIES.items()}  # activity id a → label a - 1
RAW = "RawData"


def acc_path(raw_dir: Path, exp: int, user: int) -> Path:
    return raw_dir / RAW / f"acc_exp{exp:02d}_user{user:02d}.txt"


def read_labels(raw_dir: Path) -> npt.NDArray[np.int64]:
    """labels.txt as an int [n, 5] array (exp, user, activity, start, end). ValueError if not."""
    rows = np.loadtxt(raw_dir / RAW / "labels.txt", dtype=np.int64, ndmin=2)
    if rows.shape[1] != 5:
        raise ValueError(f"labels.txt has {rows.shape[1]} columns, expected 5")
    return rows


def read_acc(path: Path) -> npt.NDArray[np.float64]:
    """One recording as float [n, 3] in g. ValueError if it is not three numeric columns."""
    acc = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if acc.shape[1] != 3 or not np.isfinite(acc).all():
        raise ValueError(f"{path.name}: expected finite [n, 3], got {acc.shape}")
    return acc


def quantise(magnitude_g: npt.NDArray[np.float64]) -> npt.NDArray[np.int16]:
    """Magnitude in g → int16 counts, 4 g full scale; values above saturate at the rail."""
    q = np.rint(magnitude_g * COUNTS_PER_G)
    return np.clip(q, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)


def build_windowset(raw_dir: Path, window_n: int = 256) -> tuple[WindowSet, dict[str, object]]:
    """Every whole `window_n` window inside each activity-1..6 interval of labels.txt."""
    rows = read_labels(raw_dir)
    xs: list[npt.NDArray[np.int16]] = []
    bids: list[str] = []
    labels: list[int] = []
    dropped: Counter[str] = Counter()
    skipped: dict[str, list[str]] = {"missing": [], "unreadable": []}
    signals: dict[tuple[int, int], npt.NDArray[np.float64] | None] = {}
    for exp, user, activity, start, end in rows.tolist():
        if activity not in ACTIVITIES:
            dropped[f"activity_{activity}"] += 1
            continue
        key = (exp, user)
        if key not in signals:
            path = acc_path(raw_dir, exp, user)
            signals[key] = None
            if not path.exists():
                skipped["missing"].append(path.name)
            else:
                try:
                    signals[key] = np.sqrt((read_acc(path) ** 2).sum(axis=1))
                except ValueError:  # malformed recording: counted, never hidden
                    skipped["unreadable"].append(path.name)
        mag = signals[key]
        if mag is None:
            dropped["recording_unavailable"] += 1
            continue
        if not 1 <= start <= end <= len(mag):
            dropped["interval_out_of_range"] += 1
            continue
        seg = mag[start - 1 : end]  # 1-based inclusive → 0-based slice
        k = len(seg) // window_n
        if k == 0:
            dropped["interval_shorter_than_window"] += 1
            continue
        xs.append(quantise(seg[: k * window_n].reshape(k, window_n)))
        bids += [f"U{user}"] * k
        labels += [activity - 1] * k
    x = np.concatenate(xs)
    y = np.asarray(labels, dtype=np.int64)
    ws = WindowSet(
        x=x,
        window_id=np.arange(len(x), dtype=np.int64),
        bearing_id=np.asarray(bids),
        condition=np.full(len(x), CONDITION),
        label=y,
        sample_rate_hz=SAMPLE_RATE_HZ,
        dataset_id=DATASET_ID,
    )
    rails = (x == np.iinfo(np.int16).min) | (x == np.iinfo(np.int16).max)
    report: dict[str, object] = {
        "n_windows": len(x),
        "n_users": len(set(bids)),
        "windows_per_class": {c: int((y == k).sum()) for c, k in LABELS.items()},
        "n_intervals": len(rows),
        "dropped_intervals": dict(dropped),
        "skipped_recordings": skipped,
        "clipped_fraction": float(rails.mean()),
        "clipped_samples": int(rails.sum()),
        "windows_with_clipping": int(rails.any(axis=1).sum()),
        "min": int(x.min()),
        "max": int(x.max()),
        "peak_g": float(x.max() / COUNTS_PER_G),
        "window_n": window_n,
        "full_scale_g": FULL_SCALE_G,
        "counts_per_g": COUNTS_PER_G,
    }
    return ws, report
