"""Uplink throughput traces: piecewise-constant, cyclic. numpy only.

The cellular presets here are **synthetic** (log-normal AR(1) around a published-order-of-magnitude
median, rescaled to the sensor-data regime). They are labelled as such everywhere; real traces can
be loaded with `from_csv` and are documented in `configs/scenario/traces/README.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Trace:
    name: str
    step_s: float
    bps: F64  # throughput during [i·step, (i+1)·step), repeated cyclically

    def __post_init__(self) -> None:
        if self.bps.max() <= 0:
            raise ValueError("a trace needs at least one interval with non-zero throughput")

    def rate(self, t_s: float) -> float:
        return float(self.bps[int(t_s / self.step_s) % len(self.bps)])

    def finish_time(self, t0_s: float, nbytes: float) -> float:
        """Time at which `nbytes` handed to the link at `t0_s` have been transmitted."""
        bits, t = nbytes * 8.0, t0_s
        i = int(t / self.step_s)
        if (i + 1) * self.step_s <= t:  # float division can land one segment short
            i += 1
        while True:  # the segment index advances by integer steps, so this always progresses
            rate = float(self.bps[i % len(self.bps)])
            seg_end = (i + 1) * self.step_s
            if rate > 0 and bits <= rate * (seg_end - t):
                return t + bits / rate
            bits -= rate * (seg_end - t)
            t, i = seg_end, i + 1


def static(bps: float) -> Trace:
    return Trace(f"static_{bps:g}", 1.0, np.array([bps], dtype=np.float64))


def markov_bursty(
    rng: np.random.Generator,
    good_bps: float,
    bad_bps: float,
    p_good_to_bad: float = 0.05,
    p_bad_to_good: float = 0.2,
    duration_s: float = 600.0,
    step_s: float = 0.1,
) -> Trace:
    """Two-state Markov-modulated link (Gilbert–Elliott) for controlled burst experiments."""
    n = int(duration_s / step_s)
    flips = rng.random(n)
    good = np.empty(n, dtype=bool)
    state = True
    for i in range(n):
        state = (flips[i] >= p_good_to_bad) if state else (flips[i] < p_bad_to_good)
        good[i] = state
    return Trace("markov_bursty", step_s, np.where(good, good_bps, bad_bps).astype(np.float64))


# name → (median bps, log-sd, AR(1) coefficient, outage probability); synthetic, see docstring
CELLULAR = {
    "3g_synth": (0.4e6, 0.9, 0.95, 0.02),
    "4g_synth": (4e6, 0.8, 0.93, 0.01),
    "5g_synth": (25e6, 1.1, 0.90, 0.02),
}


def cellular(
    rng: np.random.Generator, kind: str, duration_s: float = 600.0, step_s: float = 0.5
) -> Trace:
    """Synthetic cellular-like throughput: log-normal AR(1) with occasional near-outages."""
    median, sd, ar, p_out = CELLULAR[kind]
    n = int(duration_s / step_s)
    noise = rng.normal(0.0, sd * np.sqrt(1 - ar * ar), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        x[i] = ar * x[i - 1] + noise[i]
    bps = median * np.exp(x)
    bps[rng.random(n) < p_out] *= 0.02
    return Trace(kind, step_s, bps)


def from_csv(path: Path, step_s: float, column: int = 0, scale: float = 1.0) -> Trace:
    """Load a measured throughput trace: one sample per line (bps × `scale`)."""
    bps = np.loadtxt(path, delimiter=",", usecols=column, ndmin=1) * scale
    return Trace(path.stem, step_s, bps.astype(np.float64))


class HarmonicMeanEstimator:
    """Throughput estimate: harmonic mean of the last k goodput samples (standard in ABR)."""

    def __init__(self, initial_bps: float, k: int = 5) -> None:
        self.k = k
        self._samples = [max(initial_bps, 1.0)]  # a measured trace may start in an outage

    def add(self, bps: float) -> None:
        self._samples = [*self._samples[-(self.k - 1) :], max(bps, 1.0)]

    @property
    def bps(self) -> float:
        return len(self._samples) / sum(1.0 / s for s in self._samples)

    @property
    def last_bps(self) -> float:
        """The most recent sample: reacts at once when a link comes back, unlike the mean."""
        return self._samples[-1]
