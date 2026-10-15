"""Certificate validity on a synthetic population (PRD §6.3, §12.2 #1).

Six representations, an imperfect predictor (logit noise σ = 0.8) and correlated disagreement. The
expected-risk guarantee must hold on average over calibration draws for *any* policy inside the
safe set; the PAC variants must hold in at least a 1 - δ fraction of draws.
"""

from __future__ import annotations

import numpy as np
import pytest

from kai.certify import crc, ltt

K = 7  # raw + 6 representations
BASE_RATE = np.array([0.0, 0.01, 0.04, 0.10, 0.18, 0.30, 0.45])
COST = np.array([1.0, 0.8, 0.45, 0.2, 0.1, 0.05, 0.02])


def population(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw (ĝ, d): a shared difficulty makes disagreements correlated across representations."""
    difficulty = rng.beta(1.2, 3.0, size=(n, 1))
    q = np.clip(BASE_RATE * 4.0 * difficulty**1.5 * (1 + 0.5 * rng.normal(size=(n, K))), 0, 1)
    d = rng.random((n, K)) < q
    logit = np.log(np.clip(q, 1e-4, 1 - 1e-4)) - np.log1p(-np.clip(q, 1e-4, 1 - 1e-4))
    ghat = 1.0 / (1.0 + np.exp(-(logit + rng.normal(0, 0.8, size=(n, K)))))  # imperfect predictor
    d[:, 0], ghat[:, 0] = False, 0.0
    return ghat, d


def policy_risk(ghat: np.ndarray, d: np.ndarray, lam: float | np.ndarray, kind: str) -> float:
    safe = crc.safe_set(ghat, lam)
    if kind == "worst":  # adversary: picks a disagreeing member whenever one exists
        return float((safe & d).any(axis=1).mean())
    cost = np.where(safe, COST, np.inf)  # cheapest member of the safe set
    return float(d[np.arange(len(d)), cost.argmin(axis=1)].mean())


N_TEST = 200_000


def tolerance(alpha: float, risks: list[float]) -> float:
    """3σ of the two Monte-Carlo errors: calibration draws, and the one shared test sample."""
    return 3 * float(np.std(risks)) / np.sqrt(len(risks)) + 3 * np.sqrt(
        alpha * (1 - alpha) / N_TEST
    )


@pytest.mark.parametrize("alpha", [0.02, 0.05, 0.10])
@pytest.mark.parametrize("n_cal", [200, 1000])
def test_expected_risk_is_controlled_for_any_policy(alpha: float, n_cal: int) -> None:
    rng = np.random.default_rng([int(alpha * 100), n_cal])
    g_te, d_te = population(rng, N_TEST)
    m_te = np.sort(crc.first_loss_threshold(g_te, d_te))
    g_small, d_small = g_te[:20_000], d_te[:20_000]
    worst, greedy_gap, costs = [], [], []
    for _ in range(300):
        g_cal, d_cal = population(rng, n_cal)
        lam = crc.crc_threshold(crc.first_loss_threshold(g_cal, d_cal), alpha)
        # the worst policy inside the safe set realises exactly the safe-set loss 1[λ ≥ m]
        worst.append(np.searchsorted(m_te, lam, side="right") / N_TEST)
        greedy_gap.append(
            policy_risk(g_small, d_small, lam, "worst")
            - policy_risk(g_small, d_small, lam, "greedy")
        )
        costs.append(np.where(crc.safe_set(g_small, lam), COST, np.inf).min(axis=1).mean())
    assert np.mean(worst) <= alpha + tolerance(alpha, worst)  # the guarantee
    assert np.mean(worst) >= alpha - 2 / (n_cal + 1) - tolerance(alpha, worst)  # ...and it is tight
    assert min(greedy_gap) >= 0  # pointwise: d_π ≤ max over the safe set, for every calibration
    assert np.mean(costs) < 0.9  # the safe set really admits cheaper representations than raw
    # an in-expectation guarantee is routinely exceeded by a single calibration draw (§6.3)
    assert np.mean(np.array(worst) > alpha) > 0.15


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_ltt_is_valid_with_high_probability(alpha: float) -> None:
    delta = 0.1
    rng = np.random.default_rng(int(alpha * 1000))
    g_te, d_te = population(rng, 20_000)
    single, per_rep, crc_viol = [], [], []
    for _ in range(300):
        g_cal, d_cal = population(rng, 1000)
        m = crc.first_loss_threshold(g_cal, d_cal)
        single.append(policy_risk(g_te, d_te, ltt.ltt_threshold(m, alpha, delta), "worst"))
        per_rep.append(
            policy_risk(g_te, d_te, ltt.ltt_per_rep(g_cal, d_cal, alpha, delta), "worst")
        )
        crc_viol.append(policy_risk(g_te, d_te, crc.crc_threshold(m, alpha), "worst") > alpha)
    tol = 0.002  # test-set sampling noise on the risk estimate
    assert np.mean(np.array(single) > alpha + tol) <= delta + 0.04
    assert np.mean(np.array(per_rep) > alpha + tol) <= delta + 0.04
    assert np.mean(single) < alpha  # PAC control is strictly more conservative than CRC
    assert np.mean(crc_viol) > delta  # ...which is exactly why both variants exist


def test_policy_specific_calibration_controls_the_greedy_policy() -> None:
    alpha, rng = 0.05, np.random.default_rng(5)
    g_te, d_te = population(rng, 20_000)
    cost = np.broadcast_to(COST, g_te.shape)
    risks_v2, risks_v0 = [], []
    for _ in range(300):
        g_cal, d_cal = population(rng, 500)
        lam2 = crc.crc_threshold(crc.policy_loss_threshold(g_cal, d_cal, cost[: len(g_cal)]), alpha)
        lam0 = crc.crc_threshold(crc.first_loss_threshold(g_cal, d_cal), alpha)
        risks_v2.append(policy_risk(g_te, d_te, lam2, "greedy"))
        risks_v0.append(policy_risk(g_te, d_te, lam0, "greedy"))
        assert lam2 >= lam0  # V2 is never more conservative than V0
    assert np.mean(risks_v2) <= alpha + 3 * np.std(risks_v2) / np.sqrt(300) + 0.002
    assert np.mean(risks_v2) > np.mean(risks_v0)  # ...and spends the risk budget V0 leaves unused


def test_too_few_calibration_windows_means_raw_only() -> None:
    g, d = population(np.random.default_rng(0), 18)
    assert crc.crc_threshold(crc.first_loss_threshold(g, d), 0.05) == crc.RAW_ONLY  # n < ⌈1/α⌉ - 1
    assert not crc.safe_set(g, crc.RAW_ONLY)[:, 1:].any()
    assert crc.safe_set(g, crc.RAW_ONLY)[:, 0].all()


def test_constant_scores_give_one_valid_safe_set_for_every_window() -> None:
    """V0 with ĝ_r = the ptrain disagreement rate: no probe, same any-policy bound (ADR 0008)."""
    from kai.bench.calibrate import static_safe_set

    rng = np.random.default_rng(5)
    _, d_test = population(rng, N_TEST)
    alpha = 0.10
    risks, sizes = [], []
    for _ in range(200):
        _, d_ptrain = population(rng, 1000)
        _, d_cal = population(rng, 1000)
        mask = static_safe_set(d_ptrain, d_cal, alpha)
        assert mask.shape == (K,)
        assert mask[0]  # raw is always a member
        risks.append(float((d_test & mask).any(axis=1).mean()))  # the worst policy inside the set
        sizes.append(int(mask.sum()))
    assert np.mean(risks) <= alpha + tolerance(alpha, risks)
    assert np.mean(sizes) > 2  # not vacuous: it certifies more than raw alone


def test_cascade_union_keeps_the_guarantee_and_the_free_set_is_always_safe() -> None:
    """Static set at α/2 ∪ per-window set at α/2: any policy inside, including one that only
    ever uses the free part, stays within α in expectation (ADR 0011)."""
    from kai.bench.calibrate import cascade_certificate
    from kai.certify.runtime import Certifier

    rng = np.random.default_rng(11)
    g_te, d_te = population(rng, N_TEST)
    alpha = 0.10
    worst, free_sizes, gain = [], [], []
    for _ in range(200):
        _, d_ptrain = population(rng, 1000)
        g_cal, d_cal = population(rng, 1000)
        prior, lam = cascade_certificate(g_cal, d_cal, d_ptrain, alpha)
        cert = Certifier(lam, prior=prior)
        safe = cert.safe_set(g_te)
        free = cert.free_set(K)
        assert (safe[:, free]).all()  # the free set is inside every window's safe set
        worst.append(float((safe & d_te).any(axis=1).mean()))
        free_sizes.append(int(free.sum()))
        gain.append(float(safe.sum(axis=1).mean() - free.sum()))
    assert np.mean(worst) <= alpha + tolerance(alpha, worst)
    assert np.mean(free_sizes) > 1  # it certifies something for free ...
    assert np.mean(gain) > 0  # ... and the per-window part adds to it
