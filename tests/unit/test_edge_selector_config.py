"""Selector configuration must fail closed instead of silently changing policy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kai.contracts.schemas import LambdaEntry, LambdaTable, RepProfile
from kai.edge.bundle import Bundle
from kai.edge.runtime import EdgeConfig, build_certifier, build_selector
from kai.select.policies import CascadeSafe, GreedySafe, MpcSafe
from toy import ENERGY, PRICES, toy_model

REP_IDS = ("r0_raw", "lossless", "wavelet", "features", "latent")


def _bundle(*, cascade: dict[float, tuple[np.ndarray, float]] | None = None) -> Bundle:
    model = toy_model()
    p = model.profiles
    profiles = [
        RepProfile(
            rep_id=rep_id,
            size_mean_bytes=float(p.size_bytes[i]),
            size_q95_bytes=float(p.size_bytes[i]),
            t_enc_ms=float(p.t_enc_ms[i]),
            t_enc_mad_ms=0.0,
            t_enc_q95_ms=float(p.t_enc_ms[i]),
            t_dec_ms=float(p.t_dec_ms[i]),
            t_head_ms=float(p.t_head_ms[i]),
            load_exponent=float(p.load_exponent[i]),
        )
        for i, rep_id in enumerate(p.rep_ids)
    ]
    return Bundle(
        path=Path("."),
        manifest=SimpleNamespace(  # type: ignore[arg-type]
            rep_ids=list(p.rep_ids), bundle_id="test", git_sha="test", files={}
        ),
        reps=[],
        predictor=SimpleNamespace(),  # type: ignore[arg-type]
        lambdas=LambdaTable(entries=[LambdaEntry(variant="v0", alpha=0.05, lam=0.1, n_calib=10)]),
        profiles=profiles,
        overheads=model.overheads,
        prices=PRICES,
        energy=ENERGY,
        mission="fidelity_first",
        calib_m=np.zeros((10, len(p.rep_ids))),
        phi_ref=np.zeros((10, 4)),
        cascade=cascade,
    )


def test_edge_config_rejects_unknown_selector() -> None:
    with pytest.raises(ValueError, match="unknown selector"):
        EdgeConfig(selector="s2_mpc")  # a plausible typo for s2_mpc_safe


@pytest.mark.parametrize("option", [{"use_aci": True}, {"use_drift": True}])
def test_edge_config_rejects_cascade_with_online_certificate(option: dict[str, bool]) -> None:
    with pytest.raises(ValueError, match=r"cascade.*neither ACI nor"):
        EdgeConfig(selector="s1_cascade", **option)


def test_builders_reject_an_unknown_selector_even_without_config_validation() -> None:
    cfg = SimpleNamespace(
        selector="s2_mpc",
        period_ms=64.0,
        mission="fidelity_first",
        alpha=0.05,
        variant="v0",
        use_aci=False,
        use_drift=False,
    )
    with pytest.raises(ValueError, match="unknown selector"):
        build_selector(_bundle(), cfg, PRICES, ENERGY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown selector"):
        build_certifier(_bundle(), cfg)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("selector", "policy"),
    [
        ("s1_greedy_safe", GreedySafe),
        ("s2_mpc_safe", MpcSafe),
        ("s1_cascade", CascadeSafe),
    ],
)
def test_supported_selector_names_build_the_expected_policy(selector: str, policy: type) -> None:
    cascade = {0.05: (np.full(len(REP_IDS), np.inf), 0.1)} if selector == "s1_cascade" else None
    cfg = EdgeConfig(selector=selector)
    bundle = _bundle(cascade=cascade)

    assert isinstance(build_selector(bundle, cfg, PRICES, ENERGY), policy)
    assert build_certifier(bundle, cfg) is not None


def test_cascade_still_rejects_a_bundle_without_cascade_calibration() -> None:
    cfg = EdgeConfig(selector="s1_cascade")
    with pytest.raises(ValueError, match="predates the cascade"):
        build_certifier(_bundle(), cfg)
