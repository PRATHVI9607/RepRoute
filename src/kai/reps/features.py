"""Cheap numpy-only signal features shared by R3, R4 and the probe φ(x). Edge-safe."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
_EPS = 1e-12

TIME_STAT_NAMES = (
    "rms",
    "peak",
    "crest",
    "kurtosis",
    "skew",
    "p2p",
    "shape_factor",
    "impulse_factor",
    "margin_factor",
    "zcr",
)


def to_float(x: npt.NDArray[np.int16]) -> F64:
    """int16 window → float64 in [-1, 1)."""
    return x.astype(np.float64) / 32768.0


def time_stats(xf: F64) -> F64:
    """The ten time-domain statistics in TIME_STAT_NAMES.

    Written for the edge: one pass per moment, products instead of `**` (numpy's pow is slow).
    """
    n = len(xf)
    c = xf - xf.mean()
    c2 = c * c
    var = float(c2.sum()) / n
    absx = np.abs(xf)
    rms = float(np.sqrt(np.dot(xf, xf) / n))
    peak = float(absx.max())
    mean_abs = float(absx.sum()) / n
    sign = np.signbit(c)
    return np.array(
        [
            rms,
            peak,
            peak / (rms + _EPS),
            float(np.dot(c2, c2)) / n / (var * var + _EPS),
            float(np.dot(c2, c)) / n / (var**1.5 + _EPS),
            float(xf.max() - xf.min()),
            rms / (mean_abs + _EPS),
            peak / (mean_abs + _EPS),
            peak / ((float(np.sqrt(absx).sum()) / n) ** 2 + _EPS),
            np.count_nonzero(sign[1:] != sign[:-1]) / (n - 1),
        ]
    )


def power_spectrum(xf: F64) -> F64:
    """One-sided power spectrum of the mean-removed window (DC bin included, ≈ 0)."""
    spec = np.fft.rfft(xf - xf.mean())
    return (spec.real**2 + spec.imag**2) / len(xf)


def spectral_shape(power: F64) -> F64:
    """Spectral centroid (fraction of Nyquist) and normalised spectral entropy."""
    n = len(power)
    p = power / (power.sum() + _EPS)
    centroid = float(np.dot(p, _ramp(n))) / (n - 1)
    entropy = -float(np.dot(p, np.log(p + _EPS))) / np.log(n)
    return np.array([centroid, entropy])


@lru_cache(maxsize=8)
def _ramp(n: int) -> F64:
    return np.arange(n, dtype=np.float64)


@lru_cache(maxsize=32)
def log_band_edges(n_bins: int, n_bands: int, lo: int = 1) -> npt.NDArray[np.intp]:
    """`n_bands + 1` strictly increasing bin edges, log-spaced over [lo, n_bins]."""
    edges = np.rint(np.geomspace(lo, n_bins, n_bands + 1)).astype(np.intp)
    for i in range(1, len(edges)):  # low bands would otherwise collapse onto the same bin
        edges[i] = max(edges[i], edges[i - 1] + 1)
    if edges[-1] > n_bins:
        raise ValueError(f"{n_bands} bands do not fit in {n_bins} bins")
    edges.setflags(write=False)  # cached and shared
    return edges


def log_band_energies(power: F64, edges: npt.NDArray[np.intp]) -> F64:
    """log10 of the summed power in each band [edges[i], edges[i+1])."""
    sums = np.add.reduceat(power, edges[:-1])
    sums[-1] = power[edges[-2] : edges[-1]].sum()  # reduceat's last segment runs to the end
    return np.log10(sums + _EPS)


def envelope(xf: F64) -> F64:
    """Hilbert envelope via FFT (numpy only; scipy is not allowed on the edge)."""
    n = len(xf)
    spec = np.fft.fft(xf)
    h = np.zeros(n)
    h[0] = 1.0
    h[1 : (n + 1) // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    return np.abs(np.fft.ifft(spec * h))
