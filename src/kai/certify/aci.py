"""Audit-in-the-loop Adaptive Conformal Inference (PRD §6.4; Gibbs & Candès, NeurIPS 2021).

With probability `p_audit` the edge also sends the raw window, so the cloud observes the realised
disagreement `d_π` of the representation that was actually chosen. Audited windows are a uniformly
random subsequence, so running ACI on that subsequence tracks the realised policy risk without
inverse-probability weights (whose 1/p spikes would make the threshold swing between extremes).
This is an empirical drift mitigation, not a finite-sample guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kai.certify.crc import F64, crc_threshold


@dataclass
class AuditSampler:
    p_audit: float
    rng: np.random.Generator

    def should_audit(self) -> bool:
        return bool(self.rng.random() < self.p_audit)


def ipw_risk(audited_losses: int, n_decisions: int, p_audit: float) -> float:
    """Unbiased estimate of the realised risk over all decisions from the audited ones."""
    return audited_losses / (p_audit * n_decisions) if n_decisions else 0.0


@dataclass
class AdaptiveLambda:
    """Online threshold: re-reads the calibration quantile at a working level α_t."""

    m_calib: F64  # per-window first-loss thresholds from calibration
    alpha: float  # long-run target risk
    gamma: float = 0.02
    alpha_t: float = field(init=False)

    def __post_init__(self) -> None:
        self.alpha_t = self.alpha
        self.m_calib = np.sort(self.m_calib)

    @property
    def lam(self) -> float:
        return crc_threshold(self.m_calib, self.alpha_t)

    def update(self, disagreed: bool) -> None:
        """Call once per *audited* decision: α_t ← α_t + γ(α - err_t), clipped to [0, 1]."""
        self.alpha_t = float(
            np.clip(self.alpha_t + self.gamma * (self.alpha - disagreed), 0.0, 1.0)
        )
