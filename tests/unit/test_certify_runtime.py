"""Predictor inference parity, probe, ACI and the drift monitor."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression

from kai.certify.aci import AdaptiveLambda, AuditSampler, ipw_risk
from kai.certify.crc import ALL, RAW_ONLY, crc_threshold
from kai.certify.drift import DriftMonitor, ks_pvalue, ks_statistic
from kai.certify.fit import fit_predictor
from kai.certify.predictor import Predictor
from kai.certify.probe import PROBE_NAMES, probe
from kai.contracts.windows import WindowSet

REPS = ("r0_raw", "a", "b", "c", "d")


def toy(rng: np.random.Generator, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    phi = rng.normal(size=(n, 6))
    d = np.zeros((n, len(REPS)), bool)
    d[:, 1] = rng.random(n) < 1 / (1 + np.exp(-(2 * phi[:, 0] - 1)))  # learnable
    d[:, 2] = rng.random(n) < 1 / (1 + np.exp(-(phi[:, 1] * phi[:, 2])))  # needs a non-linear model
    d[:, 4] = True  # column 3 never disagrees, column 4 always does
    return phi, d


@pytest.mark.parametrize("kind", ["mlp", "logistic"])
def test_predictor_is_informative_and_round_trips(
    kind: str, rng: np.random.Generator, tmp_path
) -> None:
    phi, d = toy(rng)
    pred = fit_predictor(phi, d, REPS, kind=kind)  # type: ignore[arg-type]
    g = pred.predict(phi)
    assert g.shape == d.shape
    assert not g[:, 0].any()  # raw scores exactly 0
    assert ((g >= 0) & (g <= 1)).all()
    assert g[d[:, 1], 1].mean() > g[~d[:, 1], 1].mean() + 0.2
    assert g[:, 3].max() < 0.01  # never disagreed on ptrain → near-zero constant
    assert g[:, 4].min() > 0.99
    assert np.allclose(pred.predict(phi[5]), g[5])  # single window == batch row
    pred.save(tmp_path / "p.npz")
    assert np.array_equal(Predictor.load(tmp_path / "p.npz").predict(phi), g)


def test_stacked_and_per_net_evaluation_agree(rng: np.random.Generator) -> None:
    phi, d = toy(rng)
    mlp = fit_predictor(phi, d, REPS, kind="mlp")
    assert mlp._stacked is not None  # constants are padded to the fitted shape: one pass for all
    # mixing depths forces the general per-net path; both must give the same scores
    mixed = Predictor(
        REPS, mlp.mean, mlp.scale, (fit_predictor(phi, d, REPS, "logistic").nets[0], *mlp.nets[1:])
    )
    assert mixed._stacked is None
    assert np.allclose(mixed.predict(phi)[:, 2:], mlp.predict(phi)[:, 2:], atol=1e-12)
    assert mixed.predict(phi[3]).shape == (len(REPS),)


def test_logistic_predictor_matches_sklearn(rng: np.random.Generator) -> None:
    phi, d = toy(rng)
    pred = fit_predictor(phi, d, REPS, kind="logistic")
    z = (phi - pred.mean) / pred.scale
    ref = LogisticRegression(C=1.0, max_iter=2000).fit(z, d[:, 1]).predict_proba(z)[:, 1]
    assert np.allclose(pred.predict(phi)[:, 1], ref, atol=1e-9)


def test_mlp_beats_logistic_on_an_interaction(rng: np.random.Generator) -> None:
    phi, d = toy(rng, 3000)
    te_phi, te_d = toy(rng, 3000)

    def separation(kind: str) -> float:
        g = fit_predictor(phi, d, REPS, kind=kind).predict(te_phi)[:, 2]  # type: ignore[arg-type]
        return float(g[te_d[:, 2]].mean() - g[~te_d[:, 2]].mean())

    assert separation("mlp") > separation("logistic") + 0.05


def test_probe_is_finite_cheap_and_named(ws: WindowSet) -> None:
    phi = np.stack([probe(w) for w in ws.x[:50]])
    assert phi.shape == (50, len(PROBE_NAMES)) == (50, 20)
    assert np.isfinite(phi).all()
    assert np.isfinite(probe(np.zeros(ws.window_n, np.int16))).all()  # silence must not produce NaN


def test_aci_tightens_under_excess_disagreement_and_relaxes_without() -> None:
    m = np.random.default_rng(0).random(2000)
    aci = AdaptiveLambda(m, alpha=0.1, gamma=0.02)
    start = aci.lam
    assert start == crc_threshold(m, 0.1)
    for _ in range(40):
        aci.update(True)
    assert aci.alpha_t == 0.0
    assert aci.lam == RAW_ONLY
    for _ in range(2000):
        aci.update(False)
    assert aci.alpha_t == 1.0
    assert aci.lam == ALL


def test_aci_tracks_the_target_risk_after_a_shift() -> None:
    """After the disagreement process shifts, static λ over-shoots; ACI returns to ≈ α."""
    rng = np.random.default_rng(1)
    alpha = 0.1
    m_cal = rng.random(5000)  # calibrated when the loss threshold m ~ U(0, 1)
    aci = AdaptiveLambda(m_cal, alpha, gamma=0.01)
    static = crc_threshold(m_cal, alpha)
    aci_losses, static_losses = [], []
    for _ in range(30_000):
        m_t = rng.random() ** 2  # shifted: losses now occur at smaller thresholds
        loss = bool(aci.lam >= m_t)
        aci_losses.append(loss)
        static_losses.append(static >= m_t)
        aci.update(loss)
    assert np.mean(static_losses) > 0.25  # the stale threshold is badly miscalibrated
    assert abs(np.mean(aci_losses[5000:]) - alpha) < 0.015


def test_audit_sampler_rate_and_ipw_estimate() -> None:
    sampler = AuditSampler(0.03, np.random.default_rng(2))
    n = 100_000
    audited = sum(sampler.should_audit() for _ in range(n))
    assert abs(audited / n - 0.03) < 0.003
    assert ipw_risk(30, 10_000, 0.03) == pytest.approx(0.1)
    assert ipw_risk(0, 0, 0.03) == 0.0


def test_ks_matches_scipy(rng: np.random.Generator) -> None:
    for shift in (0.0, 0.3, 1.0):
        a, b = rng.normal(size=400), rng.normal(shift, 1, size=250)
        ref = stats.ks_2samp(a, b, method="asymp")
        stat = ks_statistic(a, b)
        assert stat == pytest.approx(ref.statistic, abs=1e-12)
        assert ks_pvalue(stat, len(a), len(b)) == pytest.approx(ref.pvalue, abs=0.02)
    assert ks_pvalue(0.0, 100, 100) == 1.0


def test_drift_monitor_fires_on_shift_only(rng: np.random.Generator) -> None:
    ref = rng.normal(size=(2000, 5))
    calm = DriftMonitor(ref, window=200, stride=50)
    assert not any(calm.update(rng.normal(size=5)) for _ in range(2000))
    shifted = DriftMonitor(ref, window=200, stride=50)
    fired = [shifted.update(rng.normal(size=5) + np.array([0, 0, 0.8, 0, 0])) for _ in range(400)]
    assert not any(fired[:199])  # needs a full buffer before it can speak
    assert fired[-1]
    assert shifted.min_pvalue < 1e-6


def test_predictor_cache_is_refitted_when_the_fold_table_changes(fold_dir, tmp_path) -> None:
    import shutil
    from dataclasses import replace

    from kai.bench import evaluate as ev
    from kai.tasks.pipeline import Table

    work = tmp_path / "fold"
    shutil.copytree(fold_dir, work)
    first = ev.scored_fold(work, "logistic")
    again = ev.scored_fold(work, "logistic")
    assert len(list(work.glob("predictor_logistic_*.npz"))) == 1  # second call read the cache
    assert np.array_equal(first.ghat, again.ghat)
    table = Table.load(work / "table.npz")
    replace(table, yhat=np.roll(table.yhat, 1, axis=0)).save(work / "table.npz")
    ev.scored_fold(
        work, "logistic"
    )  # a fold rebuilt in place: the old predictor must not be reused
    assert len(list(work.glob("predictor_logistic_*.npz"))) == 2


def test_a_fold_records_the_recipe_it_was_trained_with(fold_dir) -> None:
    import json

    from kai.tasks.pipeline import FoldConfig

    stored = json.loads((fold_dir / "fold_config.json").read_text())
    assert stored == FoldConfig(epochs_latent=12, head_trials=3, device="cpu").recipe()
    assert "device" not in stored  # where it ran does not change what was trained
    assert stored != FoldConfig(f_raw="env", epochs_latent=12, head_trials=3).recipe()
