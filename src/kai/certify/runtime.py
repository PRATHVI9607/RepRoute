"""Run-time certifier: scores → safe set, with optional drift safe-mode and ACI. Edge-safe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kai.certify.aci import AdaptiveLambda
from kai.certify.crc import F64, RAW_ONLY, Bool, safe_set
from kai.certify.drift import DriftMonitor


@dataclass
class Certifier:
    """Turns sufficiency scores into S_λ(x). The same object drives the simulator and the edge."""

    lam: float | F64  # calibrated threshold: scalar (V0/V1/V2) or per representation (V1 per-rep)
    aci: AdaptiveLambda | None = None  # if set, overrides `lam` with the online threshold
    drift: DriftMonitor | None = None  # if set and drifting, only raw is safe
    menu: Bool | None = None  # optional menu restriction (within-family ablations); raw stays in
    # cascade (ADR 0011): scores are min(ĝ_r(x), prior_r), constants fitted on ptrain. Any
    # representation with prior_r ≤ λ is then safe for every window, without computing ĝ.
    prior: F64 | None = None

    def __post_init__(self) -> None:
        if self.prior is not None and self.aci is not None:
            raise ValueError("ACI re-reads thresholds calibrated without a prior")

    @property
    def threshold(self) -> float | F64:
        return self.aci.lam if self.aci is not None else self.lam

    def safe_set(self, ghat: F64, phi: F64 | None = None) -> Bool:
        if self.drift is not None and phi is not None and self.drift.update(phi):
            return safe_set(ghat, RAW_ONLY)
        scores = ghat if self.prior is None else np.minimum(ghat, self.prior)
        return self._restrict(safe_set(scores, self.threshold))

    def free_set(self, n_reps: int) -> Bool:
        """Members of the safe set of *every* window: known before the probe runs."""
        if self.prior is None or self.safe_mode:
            free = np.zeros(n_reps, dtype=bool)
            free[0] = True
            return free
        return self._restrict(safe_set(self.prior, self.threshold))

    def _restrict(self, safe: Bool) -> Bool:
        if self.menu is not None:
            safe &= self.menu
            safe[..., 0] = True
        return safe

    def feedback(self, disagreed: bool) -> None:
        """Report the audited outcome of one decision (cloud compared it with the raw decision)."""
        if self.aci is not None:
            self.aci.update(disagreed)

    @property
    def safe_mode(self) -> bool:
        return bool(self.drift is not None and self.drift.drift)
