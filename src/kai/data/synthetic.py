"""Synthetic bearing-like vibration data: the tiny fixture for tests and the CI mini-paper.

Not a physical simulator. It reproduces the structure that matters for Kai: several physical
bearings per class, bearing-to-bearing variation, operating conditions, and fault signatures of
varying severity so that cheap representations disagree with the raw model on some windows.
"""

from __future__ import annotations

import numpy as np

from kai.contracts.windows import WindowSet

CONDITIONS = ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")
_SHAFT_HZ = {"N15": 25.0, "N09": 15.0}
_FAULT_ORDER = {1: 4.95, 2: 3.05}  # inner / outer race characteristic orders (6203-type)


def _window(
    rng: np.random.Generator, n: int, fs: float, label: int, cond: str, f_res: float, sev: float
) -> np.ndarray:
    t = np.arange(n) / fs
    shaft = _SHAFT_HZ[cond[:3]]
    load = 1.0 if cond.endswith("F10") else 0.6
    x = 0.3 * np.sin(2 * np.pi * shaft * t + rng.uniform(0, 2 * np.pi))
    x += 0.15 * np.sin(2 * np.pi * 2 * shaft * t + rng.uniform(0, 2 * np.pi))
    x += rng.normal(0.0, 0.35, n)
    if label:
        period = fs / (_FAULT_ORDER[label] * shaft)
        starts = np.arange(rng.uniform(0, period), n, period * (1 + rng.normal(0, 0.01)))
        for s in starts.astype(int):
            m = min(n - s, 200)
            tau = np.arange(m) / fs
            amp = sev * load * (1 + 0.2 * rng.normal())
            if label == 1:  # inner race defects rotate through the load zone
                amp *= 0.5 * (1 + np.cos(2 * np.pi * shaft * s / fs))
            x[s : s + m] += amp * np.exp(-tau * 900.0) * np.sin(2 * np.pi * f_res * tau)
    return x


def make_synthetic(
    rng: np.random.Generator,
    bearings_per_class: int = 6,
    windows_per_condition: int = 12,
    window_n: int = 1024,
    sample_rate_hz: float = 16_000.0,
    full_scale: float = 8.0,
) -> WindowSet:
    """Build a WindowSet with `3 * bearings_per_class` bearings and four operating conditions."""
    xs, bids, conds, labels = [], [], [], []
    for label, prefix in enumerate(("K0", "KI", "KA")):
        for b in range(bearings_per_class):
            bid = f"{prefix}{b + 1:02d}"
            # inner and outer defects ring in different bands so the classes generalise across
            # bearings; faint faults (low severity) still give hard windows
            f_res = rng.uniform(2500.0, 3500.0) if label == 1 else rng.uniform(5000.0, 6500.0)
            sev = rng.uniform(0.8, 3.0)
            gain = rng.uniform(0.85, 1.15)
            for cond in CONDITIONS:
                for _ in range(windows_per_condition):
                    xs.append(
                        gain * _window(rng, window_n, sample_rate_hz, label, cond, f_res, sev)
                    )
                    bids.append(bid)
                    conds.append(cond)
                    labels.append(label)
    x = np.clip(np.rint(np.asarray(xs) / full_scale * 32767.0), -32768, 32767).astype(np.int16)
    return WindowSet(
        x=x,
        window_id=np.arange(len(x), dtype=np.int64),
        bearing_id=np.asarray(bids),
        condition=np.asarray(conds),
        label=np.asarray(labels, dtype=np.int64),
        sample_rate_hz=sample_rate_hz,
        dataset_id="synthetic-v1",
    )
