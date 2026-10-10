"""Run-time certifier: scores → safe set, with optional drift safe-mode and ACI. Edge-safe."""

from __future__ import annotations

from dataclasses import dataclass

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

    def safe_set(self, ghat: F64, phi: F64 | None = None) -> Bool:
        if self.drift is not None and phi is not None and self.drift.update(phi):
            return safe_set(ghat.copy(), RAW_ONLY)
        safe = safe_set(ghat.copy(), self.aci.lam if self.aci is not None else self.lam)
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
