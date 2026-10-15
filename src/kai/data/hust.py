"""HUST bearing dataset (transfer dataset): MAT recordings → int16 WindowSet.

Lab-plane only (uses scipy.io). Hong & Thuan, "HUST bearing: a practical dataset for ball bearing
fault diagnosis", Mendeley Data v3, doi:10.17632/cbv7jyx4p9.3, CC BY 4.0. A recording
`<defect><bearing type><load>.mat` (I402 = inner race, 6204, 200 W) holds 10 s of `data` sampled at
51.2 kHz and stored as float64; it is quantised once to int16 with a fixed full scale.
The `fs` variable in the files is the shaft frequency in Hz, not the sample rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from kai.contracts.windows import WindowSet
from kai.data.pu import quantise, segment

SAMPLE_RATE_HZ = 51_200.0
DATASET_ID = "hust-v1"
LABELS = {"N": 0, "I": 1, "O": 2}  # normal / inner race / outer race; B, IB, IO, OB are unused
BEARING_TYPES = "45678"  # last digit of the bearing designation 6204 … 6208
CONDITIONS = ("0W", "200W", "400W")  # brake load; file codes 00 / 02 / 04
# the published signals saturate at ±51.15 (I50x and I60x reach it), so this clips nothing
FULL_SCALE = 51.2

# file name → (bearing_id, condition, label); one (defect, bearing type) pair is one bearing_id
RECORDINGS: dict[str, tuple[str, str, int]] = {
    f"{defect}{t}0{2 * i}.mat": (f"{defect}620{t}", cond, label)
    for defect, label in LABELS.items()
    for t in BEARING_TYPES
    for i, cond in enumerate(CONDITIONS)
}


def build_windowset(
    raw_dir: Path, window_n: int = 4096, full_scale: float = FULL_SCALE
) -> tuple[WindowSet, dict[str, float | list[str]]]:
    """Quantise and window the 3-class recordings found in `raw_dir`. Returns (WindowSet, QA)."""
    xs, bids, conds, labels = [], [], [], []
    clipped = total = 0
    skipped: list[str] = []
    dropped: list[str] = []
    peak = 0.0
    for path in sorted(raw_dir.glob("*.mat")):
        if path.name not in RECORDINGS:  # ball / combined defects have no 3-class label
            dropped.append(path.name)
            continue
        bearing, cond, label = RECORDINGS[path.name]
        try:
            signal = np.asarray(loadmat(path)["data"], dtype=np.float64).ravel()
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
        conds += [cond] * len(w)
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
