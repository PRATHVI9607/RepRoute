"""Paderborn University (PU) bearing dataset: archive extraction → int16 WindowSet.

Lab-plane only (uses scipy.io). Recordings are 4 s of `vibration_1` at 64 kHz stored as float64;
they are quantised once to int16 with a fixed full scale so that "raw" is what an ADC would deliver.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import loadmat

from kai.contracts.windows import WindowSet

SAMPLE_RATE_HZ = 64_000.0
CONDITIONS = ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")
DATASET_ID = "pu-v1"


Origin = Literal["healthy", "artificial", "real"]


@dataclass(frozen=True)
class BearingMeta:
    label: int | None  # 0 healthy, 1 inner race, 2 outer race, None = combined damage (unused)
    origin: Origin


def _meta(codes: str, label: int | None, origin: Origin) -> dict[str, BearingMeta]:
    return {c: BearingMeta(label, origin) for c in codes.split()}


BEARINGS: dict[str, BearingMeta] = {
    **_meta("K001 K002 K003 K004 K005 K006", 0, "healthy"),
    **_meta("KI01 KI03 KI05 KI07 KI08", 1, "artificial"),
    **_meta("KI04 KI14 KI16 KI17 KI18 KI21", 1, "real"),
    **_meta("KA01 KA03 KA05 KA06 KA07 KA08 KA09", 2, "artificial"),
    **_meta("KA04 KA15 KA16 KA22 KA30", 2, "real"),
    **_meta("KB23 KB24 KB27", None, "real"),
}


def extract(archive: Path, dest: Path, measurements: range) -> None:
    """Extract only the requested measurement indices of one bearing archive."""
    masks = [f"*_{i}.mat" for i in measurements]
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("unrar"):
        cmd = ["unrar", "x", "-o+", "-inul", str(archive), *masks, f"{dest}/"]
    elif shutil.which("7z"):
        cmd = ["7z", "x", "-y", "-bso0", f"-o{dest}", str(archive), *masks, "-r"]
    else:
        raise RuntimeError("need `unrar` or `7z` on PATH to extract the PU archives")
    subprocess.run(cmd, check=True)


def load_vibration(mat_path: Path) -> np.ndarray:
    """Return the float64 `vibration_1` channel of one recording."""
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    rec = next(v for k, v in mat.items() if not k.startswith("__"))
    for channel in np.atleast_1d(rec.Y):
        if channel.Name == "vibration_1":
            return np.asarray(channel.Data, dtype=np.float64)
    raise ValueError(f"{mat_path.name}: no vibration_1 channel")


def quantise(signal: np.ndarray, full_scale: float) -> tuple[np.ndarray, int]:
    """float → int16 with a fixed full scale. Returns (samples, number of clipped samples)."""
    q = np.rint(signal / full_scale * 32767.0)
    clipped = int(np.count_nonzero(np.abs(q) > 32767))
    return np.clip(q, -32767, 32767).astype(np.int16), clipped


def segment(samples: np.ndarray, window_n: int) -> np.ndarray:
    """Non-overlapping windows (hop = N); the tail that does not fill a window is dropped."""
    n = len(samples) // window_n
    return samples[: n * window_n].reshape(n, window_n)


def build_windowset(
    raw_dir: Path,
    interim_dir: Path,
    window_n: int = 4096,
    n_measurements: int = 5,
    full_scale: float = 16.0,
    bearings: list[str] | None = None,
) -> tuple[WindowSet, dict[str, float | list[str]]]:
    """Extract, quantise and window the 3-class PU subset. Returns the WindowSet and a QA report."""
    names = bearings or [b for b, m in BEARINGS.items() if m.label is not None]
    xs, bids, conds, labels = [], [], [], []
    clipped = total = 0
    skipped: list[str] = []
    peak = 0.0
    for name in names:
        label = BEARINGS[name].label
        if label is None:
            raise ValueError(f"{name} has combined damage and no 3-class label")
        extract(raw_dir / f"{name}.rar", interim_dir, range(1, n_measurements + 1))
        for cond in CONDITIONS:
            for i in range(1, n_measurements + 1):
                path = interim_dir / name / f"{cond}_{name}_{i}.mat"
                try:
                    signal = load_vibration(path)
                except Exception:  # a few published recordings are corrupt; count, never hide
                    skipped.append(path.name)
                    continue
                samples, c = quantise(signal, full_scale)
                clipped += c
                total += len(samples)
                peak = max(peak, float(np.abs(signal).max()))
                w = segment(samples, window_n)
                xs.append(w)
                bids += [name] * len(w)
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
        "n_bearings": float(len(names)),
        "skipped_recordings": skipped,
        "clipped_fraction": clipped / max(total, 1),
        "peak_abs": peak,
        "full_scale": full_scale,
    }
    return ws, report
