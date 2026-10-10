"""Lab-side certification steps shared by the bundle builder and the experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from kai.certify import crc, ltt
from kai.certify.fit import fit_predictor
from kai.certify.predictor import Predictor
from kai.contracts.schemas import LambdaEntry, LambdaTable, Partition, Variant
from kai.tasks.pipeline import Table

ALPHAS = (0.02, 0.05, 0.10)


@dataclass(frozen=True)
class Scored:
    """A fold's Disagreement Table together with the sufficiency scores of every window."""

    table: Table
    predictor: Predictor
    ghat: np.ndarray  # [n, K]

    def part(self, partition: Partition) -> tuple[np.ndarray, np.ndarray]:
        """(ĝ, d) of one partition."""
        m = self.table.of(partition)
        return self.ghat[m], self.table.d[m]


def score(table: Table, kind: Literal["mlp", "logistic"] = "mlp", seed: int = 0) -> Scored:
    """Fit ĝ on `ptrain` only and score every window."""
    m = table.of("ptrain")
    predictor = fit_predictor(table.phi[m], table.d[m], table.rep_ids, kind, seed)
    return Scored(table, predictor, predictor.predict(table.phi))


def calibrate(
    ghat_cal: np.ndarray,
    d_cal: np.ndarray,
    rep_ids: tuple[str, ...],
    alphas: tuple[float, ...] = ALPHAS,
    delta: float = 0.1,
    cost_cal: np.ndarray | None = None,
) -> LambdaTable:
    """λ̂ for every α and variant. V2 needs per-window costs under the nominal state distribution."""
    n = len(ghat_cal)
    m0 = crc.first_loss_threshold(ghat_cal, d_cal)
    entries = []
    for a in alphas:
        entries.append(LambdaEntry(variant="v0", alpha=a, lam=crc.crc_threshold(m0, a), n_calib=n))
        entries.append(
            LambdaEntry(
                variant="v1", alpha=a, delta=delta, lam=ltt.ltt_threshold(m0, a, delta), n_calib=n
            )
        )
        per_rep = ltt.ltt_per_rep(ghat_cal, d_cal, a, delta)
        entries.append(
            LambdaEntry(
                variant="v1_per_rep",
                alpha=a,
                delta=delta,
                lam_per_rep=dict(zip(rep_ids, per_rep.tolist(), strict=True)),
                n_calib=n,
            )
        )
        if cost_cal is not None:
            m2 = crc.policy_loss_threshold(ghat_cal, d_cal, cost_cal)
            entries.append(
                LambdaEntry(variant="v2", alpha=a, lam=crc.crc_threshold(m2, a), n_calib=n)
            )
    return LambdaTable(entries=entries)


def threshold(
    table: LambdaTable, variant: Variant, alpha: float, rep_ids: tuple[str, ...]
) -> float | np.ndarray:
    """The calibrated threshold in the form `Certifier` expects."""
    entry = table.get(variant, alpha)
    if entry.lam_per_rep is not None:
        return np.array([entry.lam_per_rep[r] for r in rep_ids])
    assert entry.lam is not None
    return entry.lam
