"""Probe φ(x): the cheap features computed for every window (PRD §6.1). numpy only — edge-safe."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kai.reps import features as ft

N_BANDS = 8
PROBE_VERSION = "2"  # bump when the features change: cached probes are keyed on it
PROBE_NAMES: tuple[str, ...] = (
    *ft.TIME_STAT_NAMES,
    "spectral_centroid",
    "spectral_entropy",
    *(f"band_{i}" for i in range(N_BANDS)),
)


def probe(x: npt.NDArray[np.int16]) -> npt.NDArray[np.float64]:
    """One FFT plus time statistics: 20 features, far cheaper than any encoder worth selecting."""
    xf = ft.to_float(x)
    power = ft.power_spectrum(xf)
    edges = ft.log_band_edges(len(power) - 1, N_BANDS)
    out: npt.NDArray[np.float64] = np.concatenate(
        [ft.time_stats(xf), ft.spectral_shape(power), ft.log_band_energies(power, edges)]
    )
    return out
