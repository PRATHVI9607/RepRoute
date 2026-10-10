"""Small hand-made menu and workload shared by the selector and simulator tests."""

from __future__ import annotations

import numpy as np

from kai.contracts.schemas import EnergyModel, Prices, RepProfile
from kai.link.des import Workload
from kai.select.cost import CostModel, Overheads, Profiles

REP_IDS = ("r0_raw", "lossless", "wavelet", "features", "latent")
# rep_id, size (bytes), t_enc, t_dec, t_head (ms)
_ROWS = [
    ("r0_raw", 2048, 0.01, 0.01, 1.0),
    ("lossless", 1500, 0.30, 0.20, 1.0),
    ("wavelet", 600, 1.00, 0.60, 1.0),
    ("features", 128, 0.50, 0.00, 0.3),
    ("latent", 16, 3.00, 0.00, 0.3),
]
PRICES = Prices(name="toy", story="test", p_byte=1.0, p_j=0.0, p_cpu_ms=5.0, p_late_ms=1000.0)
ENERGY = EnergyModel(p_cpu_w=2.0, e_bit_j=1e-7)
OVERHEADS = Overheads(0.05, 0.02, 0.01)


def toy_model(prices: Prices = PRICES, exponent: float = 1.0) -> CostModel:
    profiles = [
        RepProfile(
            rep_id=r, size_mean_bytes=s, size_q95_bytes=s, t_enc_ms=e, t_enc_mad_ms=0.0,
            t_enc_q95_ms=e, t_dec_ms=d, t_head_ms=h, load_exponent=exponent,
        )
        for r, s, e, d, h in _ROWS
    ]  # fmt: skip
    return CostModel(Profiles.from_list(profiles), OVERHEADS, ENERGY, prices)


def toy_workload(rng: np.random.Generator, n: int, n_probe: int = 4) -> Workload:
    """Scores that are informative about disagreement; cheaper representations disagree more."""
    base = np.array([0.0, 0.0, 0.03, 0.15, 0.30])
    q = np.clip(base * 3 * rng.random((n, 1)) ** 1.5 * (1 + 0.3 * rng.normal(size=(n, 5))), 0, 1)
    d = rng.random((n, 5)) < q
    ghat = np.clip(q + rng.normal(0, 0.03, (n, 5)), 0, 1)
    ghat[:, 0], d[:, 0] = 0.0, False
    sizes = np.tile(np.array([r[1] for r in _ROWS], dtype=np.int32), (n, 1))
    return Workload(ghat=ghat, phi=rng.normal(size=(n, n_probe)), d=d, payload_bytes=sizes)
