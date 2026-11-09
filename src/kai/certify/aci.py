"""Audit-in-the-loop Adaptive Conformal Inference (PRD §6.4; Gibbs & Candès, NeurIPS 2021).

With probability `p_audit` the edge also sends the raw window, so the cloud observes the realised
disagreement `d_π` of the representation that was actually chosen. Audited windows are a uniformly
random subsequence, so running ACI on that subsequence tracks the realised policy risk without
inverse-probability weights (whose 1/p spikes would make the threshold swing between extremes).
This is an empirical drift mitigation, not a finite-sample guarantee.

Target: α_t is steered by the realised loss of the *policy's* certified choice, while `m_calib`
holds the V0 safe-set thresholds and only supplies the ladder of λ values. A policy inside S_λ
loses no more often than the safe set does, so on exchangeable data α_t tends above α, steering
the realised policy risk towards α (it sits at the clip of 1 if the policy's risk at λ = ALL is
already below α). Once α_t > α the safe set S_λt is tuned to this one policy and its own risk can
exceed α: with ACI on, the any-policy bound no longer holds and swapping the selector voids the
target. Feed it certified decisions only (see `kai.link.des`): uncertified sends are outside the
certificate. The DES feeds back every certified audit at decision time; the edge only when the
cloud's reply arrives, so on a lossy link the edge's audited subsequence depends on link state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kai.certify.crc import F64, crc_allowed_losses, threshold_for_count


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
    """Online threshold: re-reads the calibration quantile at a working level α_t.

    α_t tracks the realised risk of the audited certified decisions (module docstring).
    """

    m_calib: F64  # per-window first-loss thresholds from calibration
    alpha: float  # long-run target risk
    gamma: float = 0.02
    alpha_t: float = field(init=False)

    def __post_init__(self) -> None:
        self.alpha_t = self.alpha
        self.m_calib = np.sort(self.m_calib)

    @property
    def lam(self) -> float:
        # m_calib is sorted once in __post_init__; this is O(1) per window
        return threshold_for_count(
            self.m_calib, crc_allowed_losses(len(self.m_calib), self.alpha_t)
        )

    def update(self, disagreed: bool) -> None:
        """Call once per *audited* decision: α_t ← α_t + γ(α - err_t), clipped to [0, 1]."""
        self.alpha_t = float(
            np.clip(self.alpha_t + self.gamma * (self.alpha - disagreed), 0.0, 1.0)
        )
