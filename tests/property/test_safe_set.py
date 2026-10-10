"""Safe-set laws (PRD §12.2 #2) and exactness of the calibration thresholds."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy import stats

from kai.certify import crc, ltt

shapes = st.tuples(st.integers(1, 40), st.integers(2, 7))


@st.composite
def tables(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    n, k = draw(shapes)
    # few distinct values on purpose: ties are where threshold logic breaks
    ghat = draw(
        hnp.arrays(np.float64, (n, k), elements=st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.9, 1.0]))
    )
    d = draw(hnp.arrays(np.bool_, (n, k)))
    ghat[:, 0], d[:, 0] = 0.0, False
    return ghat, d


lams = st.floats(-1.0, 1.0)


@given(tables(), lams, lams)
def test_safe_set_grows_and_loss_is_monotone(
    t: tuple[np.ndarray, np.ndarray], a: float, b: float
) -> None:
    ghat, d = t
    lo, hi = sorted((a, b))
    s_lo, s_hi = crc.safe_set(ghat, lo), crc.safe_set(ghat, hi)
    assert (s_hi | ~s_lo).all()  # S_lo ⊆ S_hi
    assert s_lo[:, 0].all()  # raw is always a member
    assert (crc.safe_set_loss(ghat, d, hi) | ~crc.safe_set_loss(ghat, d, lo)).all()


@given(tables(), lams)
def test_loss_equals_threshold_form(t: tuple[np.ndarray, np.ndarray], lam: float) -> None:
    ghat, d = t
    m = crc.first_loss_threshold(ghat, d)
    assert np.array_equal(crc.safe_set_loss(ghat, d, lam), lam >= m)


@given(tables(), st.sampled_from([0.02, 0.05, 0.1, 0.3, 0.6]))
def test_crc_threshold_is_the_largest_valid_one(
    t: tuple[np.ndarray, np.ndarray], alpha: float
) -> None:
    ghat, d = t
    m = crc.first_loss_threshold(ghat, d)
    n = len(m)
    lam = crc.crc_threshold(m, alpha)

    def bound(x: float) -> float:
        return (crc.safe_set_loss(ghat, d, x).sum() + 1) / (n + 1)

    if lam == crc.RAW_ONLY:
        assert 1 / (n + 1) > alpha or bound(0.0) > alpha  # nothing above raw-only is admissible
    else:
        assert bound(lam) <= alpha + 1e-12
    for candidate in np.unique(m[np.isfinite(m)]):
        if candidate > lam:
            assert bound(float(candidate)) > alpha  # any larger loss point breaks the bound


@given(tables(), st.data())
def test_policy_threshold_matches_brute_force(
    t: tuple[np.ndarray, np.ndarray], data: st.DataObject
) -> None:
    ghat, d = t
    n, k = ghat.shape
    # distinct ĝ per row so that entry order is unambiguous
    ghat = (np.argsort(np.argsort(ghat + np.arange(k) * 1e-3, axis=1), axis=1) + 1) / (k + 1.0)
    ghat[:, 0] = 0.0
    cost = data.draw(
        hnp.arrays(np.float64, (n, k), elements=st.sampled_from([1.0, 2.0, 3.0, 5.0, np.inf]))
    )
    cost[:, 0] = 4.0
    m = crc.policy_loss_threshold(ghat, d, cost)
    for i in range(n):
        expected = np.inf
        for lam in np.sort(ghat[i, 1:]):
            safe = crc.safe_set(ghat[i : i + 1], float(lam))[0]
            choice = int(np.where(safe, cost[i], np.inf).argmin())
            if d[i, choice]:
                expected = lam
                break
        assert m[i] == expected


@given(st.integers(0, 60), st.integers(1, 60), st.floats(0.01, 0.99))
def test_binom_cdf_matches_scipy(k: int, n: int, p: float) -> None:
    assert abs(ltt.binom_cdf(k, n, p) - stats.binom.cdf(k, n, p)) < 1e-9


def test_binom_cdf_edges_and_certified_counts() -> None:
    assert ltt.binom_cdf(-1, 10, 0.3) == 0.0
    assert ltt.binom_cdf(10, 10, 0.3) == 1.0
    assert ltt.binom_cdf(3, 10, 0.0) == 1.0
    assert ltt.binom_cdf(3, 10, 1.0) == 0.0
    for n, alpha, delta in [
        (100, 0.1, 0.1),
        (1000, 0.05, 0.1),
        (20, 0.05, 0.1),
        (5000, 0.02, 0.05),
    ]:
        c = ltt.max_certified_losses(n, alpha, delta)
        assert stats.binom.cdf(c + 1, n, alpha) > delta
        assert c == -1 or stats.binom.cdf(c, n, alpha) <= delta
    assert ltt.max_certified_losses(20, 0.05, 0.1) == -1  # 0.95**20 > 0.1: nothing certifiable
    assert ltt.ltt_threshold(np.array([]), 0.1, 0.1) == crc.RAW_ONLY
    assert ltt.ltt_threshold(np.full(500, np.inf), 0.1, 0.1) == crc.ALL


def test_policy_threshold_breaks_cost_ties_like_the_selector() -> None:
    """Two equally cheap representations: argmin takes the lower index, and so must calibration.

    Regression: r6 and r7 both cost 16 bytes; calibrating as if the first *admitted* one were kept
    under-counted the loss whenever the lower-index one disagreed.
    """
    ghat = np.array([[0.0, 0.75, 0.5]])  # column 2 enters the safe set first
    d = np.array([[False, True, False]])  # ...but column 1 is the one that disagrees
    cost = np.array([[4.0, 1.0, 1.0]])  # and both cost the same
    assert crc.policy_loss_threshold(ghat, d, cost)[0] == 0.75
    safe = crc.safe_set(ghat.copy(), 0.75)[0]
    assert int(np.where(safe, cost[0], np.inf).argmin()) == 1  # what greedy really picks
