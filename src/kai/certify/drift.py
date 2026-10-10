"""Probe-feature drift monitor (PRD §6.4): two-sample KS test per feature, Bonferroni-corrected.

When the recent windows no longer look like the calibration windows the exchangeability premise of
the certificate is in doubt, and the runtime enters safe mode (raw only). numpy only — edge-safe.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from kai.certify.crc import F64


def ks_statistic(a: F64, b: F64) -> float:
    """Two-sample Kolmogorov–Smirnov statistic sup|F_a - F_b|."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def ks_pvalue(stat: float, n: int, m: int) -> float:
    """Asymptotic two-sided p-value (Kolmogorov distribution with the Stephens correction)."""
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * stat
    if lam < 0.2:  # the alternating series does not converge here; the p-value is ≈ 1
        return 1.0
    total = sum((-1) ** (j - 1) * math.exp(-2.0 * lam * lam * j * j) for j in range(1, 101))
    return float(min(1.0, max(0.0, 2.0 * total)))


class DriftMonitor:
    """Sliding window of recent probe vectors compared against a calibration reference."""

    def __init__(self, reference: F64, window: int = 200, level: float = 0.01, stride: int = 50):
        self.reference = np.asarray(reference, dtype=np.float64)
        self.window, self.level, self.stride = window, level, stride
        self._recent: deque[F64] = deque(maxlen=window)
        self._seen = 0
        self.drift = False
        self.min_pvalue = 1.0

    def update(self, phi: F64) -> bool:
        """Add one probe vector; re-test every `stride` windows once the buffer is full."""
        self._recent.append(np.asarray(phi, dtype=np.float64))
        self._seen += 1
        if len(self._recent) == self.window and self._seen % self.stride == 0:
            recent = np.stack(self._recent)
            n_feat = recent.shape[1]
            self.min_pvalue = min(
                ks_pvalue(
                    ks_statistic(self.reference[:, j], recent[:, j]),
                    len(self.reference),
                    self.window,
                )
                for j in range(n_feat)
            )
            self.drift = self.min_pvalue < self.level / n_feat
        return self.drift
