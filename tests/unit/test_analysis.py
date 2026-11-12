"""Regime theory and statistics helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from kai.analysis import phase, stats
from kai.select.cost import State
from toy import REP_IDS, toy_model

MODEL = toy_model()
RAW, LOSSLESS, WAVELET, FEATURES, LATENT = range(5)


def test_break_even_is_where_the_two_latencies_cross() -> None:
    for load in (0.0, 0.5, 0.8):
        b = phase.break_even_bps(MODEL, LATENT, FEATURES, load)
        lat = MODEL.latency_ms(State(b, 20.0, 0.0, load, 1e9))
        assert lat[LATENT] == pytest.approx(lat[FEATURES])
        below = MODEL.latency_ms(State(b * 0.9, 20.0, 0.0, load, 1e9))
        above = MODEL.latency_ms(State(b * 1.1, 20.0, 0.0, load, 1e9))
        assert below[LATENT] < below[FEATURES]  # compact wins on slow links
        assert above[LATENT] > above[FEATURES]  # cheap-to-encode wins on fast links


def test_zero_bandwidth_is_infeasible_without_a_divide_warning() -> None:
    state = State(bandwidth_bps=0.0, rtt_ms=20.0, backlog_bytes=1000.0, load=0.5, deadline_ms=250.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        latency = MODEL.latency_ms(state)
        cost, feasible = MODEL.evaluate(state)

    assert np.isinf(latency).all()
    assert np.isinf(cost).all()
    assert not feasible.any()


def test_break_even_shrinks_as_edge_load_rises() -> None:
    b = [phase.break_even_bps(MODEL, LATENT, FEATURES, rho) for rho in (0.0, 0.5, 0.8, 0.95)]
    assert b == sorted(b, reverse=True)
    assert b[0] / b[-1] == pytest.approx(20.0)  # Δt ∝ 1/(1-ρ) when only encode time differs


def test_break_even_edge_cases() -> None:
    with pytest.raises(ValueError, match="larger"):
        phase.break_even_bps(MODEL, RAW, LATENT, 0.0)
    # features is smaller than wavelet *and* faster end to end: it wins at every bandwidth
    assert phase.break_even_bps(MODEL, FEATURES, WAVELET, 0.0) == float("inf")


@pytest.mark.parametrize("load", [0.0, 0.5, 0.8])
def test_envelope_matches_a_brute_force_scan(load: float) -> None:
    grid = np.logspace(3, 9, 6001)
    best = phase.argmin_map(MODEL, grid, np.array([load]))[0]
    changes = np.flatnonzero(np.diff(best))
    predicted = phase.latency_envelope(MODEL, load)
    assert len(predicted) == len(changes) >= 1
    for boundary, g in zip(predicted, changes, strict=True):
        assert (boundary.low, boundary.high) == (best[g], best[g + 1])
        assert grid[g] <= boundary.bps <= grid[g + 1]
        assert boundary.bps == pytest.approx(
            phase.break_even_bps(MODEL, boundary.low, boundary.high, load)
        )
    assert REP_IDS[best[0]] == "latent"  # slowest links: the smallest payload
    assert best[-1] == phase.intercepts_ms(MODEL, load).argmin()  # fastest links: least fixed time
    assert len(phase.latency_envelope(MODEL, 0.8)) >= len(phase.latency_envelope(MODEL, 0.0))


def test_envelope_respects_the_allowed_mask_and_deadline() -> None:
    allowed = np.array([True, True, False, False, False])
    env = phase.latency_envelope(MODEL, 0.0, allowed)
    assert [(b.low, b.high) for b in env] == [(LOSSLESS, RAW)]
    grid = np.logspace(3, 8, 50)
    m = phase.argmin_map(MODEL, grid, np.array([0.0, 0.8]), deadline_ms=30.0, by="cost")
    assert (m[:, 0] == -1).all()  # 1 kbps cannot deliver anything within 30 ms
    assert (m[:, -1] >= 0).all()
    assert set(np.unique(phase.argmin_map(MODEL, grid, np.array([0.0]), allowed=allowed))) <= {0, 1}


def test_empirical_crossings_recover_the_theory_from_noisy_measurements() -> None:
    rng = np.random.default_rng(0)
    grid = np.logspace(4, 8, 25)
    lat = np.stack([MODEL.latency_ms(State(b, 0.0, 0.0, 0.5, 1e9)) for b in grid])
    measured = lat * (1 + rng.normal(0, 0.01, lat.shape))
    theory = phase.latency_envelope(MODEL, 0.5)
    found = phase.crossings(grid, measured)
    assert [(b.low, b.high) for b in found] == [(b.low, b.high) for b in theory]
    for f, t in zip(found, theory, strict=True):
        assert abs(f.bps - t.bps) / t.bps < 0.2  # the H6 tolerance


def test_bootstrap_intervals_cover_and_handle_degenerate_input() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.3, 0.1, 40)
    lo, hi = stats.bca_ci(x, rng)
    assert lo < 0.3 < hi
    assert stats.bca_ci(np.array([0.2, 0.2, 0.2]), rng) == pytest.approx((0.2, 0.2))
    assert stats.bca_ci(np.array([0.7]), rng) == (0.7, 0.7)
    mean, lo, hi = stats.paired_bca(x + 0.05, x, rng)
    assert mean == pytest.approx(0.05)
    assert lo == pytest.approx(0.05)
    assert hi == pytest.approx(0.05)


def test_block_bootstrap_is_wider_than_iid_when_blocks_are_correlated() -> None:
    rng = np.random.default_rng(2)
    blocks = np.repeat(np.arange(8), 500)
    rates = rng.uniform(0.0, 0.2, 8)  # each bearing has its own disagreement rate
    losses = rng.random(4000) < rates[blocks]
    lo, hi = stats.block_bootstrap_ci(losses, blocks, rng)
    iid = 1.96 * np.sqrt(losses.mean() * (1 - losses.mean()) / 4000)
    assert lo < losses.mean() < hi
    assert (hi - lo) > 3 * (2 * iid)


def test_holm_and_one_sided_tests() -> None:
    out = stats.holm({"h1": 0.001, "h2": 0.04, "h3": 0.03, "h4": 0.5})
    assert out["h1"] == (pytest.approx(0.004), True)
    assert not out["h2"][1]
    assert not out["h4"][1]
    assert out["h3"][0] == pytest.approx(0.09)
    v = np.array([0.31, 0.28, 0.35, 0.30, 0.33])
    assert stats.one_sided_p(v, 0.2) < 0.01
    assert stats.one_sided_p(v, 0.2, greater=False) > 0.99
    assert stats.one_sided_p(np.array([0.5, 0.5]), 0.2) == 1.0  # no spread is no evidence
    assert stats.one_sided_p(np.array([0.1]), 0.2) == 1.0
    assert stats.one_sided_p(np.array([0.5, 0.6, -np.inf]), 0.2) == 1.0  # a failed unit


def test_grid_helpers_ignore_cells_and_folds_that_carry_no_information() -> None:
    import pandas as pd

    from kai.bench.experiments import CELL_KEYS, dominant_rep, reduction

    cell = dict.fromkeys(CELL_KEYS, 0.0)
    rows = [
        {
            "fold": "a",
            **cell,
            "policy": "oracle",
            "share/r0_raw": 0.0,
            "share/r6": 0.0,
        },  # all deferred
        {"fold": "b", **cell, "policy": "oracle", "share/r0_raw": 0.1, "share/r6": 0.9},
    ]
    main = dominant_rep(pd.DataFrame(rows), "oracle")
    assert main.tolist() == ["none", "r6"]  # the empty cell is not counted as a raw choice
    frame = pd.DataFrame(
        [
            {"fold": f, **cell, "policy": p, "cost": c}
            for f, p, c in [("a", "ours", 1.0), ("a", "theirs", 0.0), ("b", "ours", 1.0),
                            ("b", "theirs", 2.0), ("c", "ours", 3.0), ("c", "theirs", 4.0),
                            ("d", "ours", 0.0), ("d", "theirs", 0.0)]
        ]
    )  # fmt: skip
    out = reduction(frame, "ours", "theirs", np.random.default_rng(0))
    # a fold whose reference is 0 is never dropped: worse than nothing (a) or unchanged (d)
    assert out["per_fold"] == pytest.approx([-np.inf, 0.5, 0.25, 0.0])
    assert out["mean"] == -np.inf
    assert out["folds"] == ["a", "b", "c", "d"]


def test_stored_grid_results_are_reused_and_extended_with_new_policies(
    tmp_path, monkeypatch
) -> None:
    import json

    import pandas as pd
    from omegaconf import OmegaConf

    from kai.bench import experiments as ex
    from kai.bench.experiments import out_dir, stored_or_run

    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    monkeypatch.setattr(ex, "provenance", lambda cfg: {"config_hash": "h", "split_id": "s"})
    monkeypatch.setattr(ex, "known_policies", lambda cfg, which: {"a", "b"})
    pd.DataFrame({"policy": ["a", "b"], "cost": [1.0, 2.0]}).to_parquet(
        out_dir("e3") / "results.parquet"
    )
    (out_dir("e3") / "results.meta.json").write_text(json.dumps(ex.provenance(None)))
    calls = []

    def run(only: tuple[str, ...]) -> pd.DataFrame:
        calls.append(only)
        return pd.DataFrame({"policy": ["b"], "cost": [5.0]})

    fresh = stored_or_run(OmegaConf.create({"reuse": False}), "e3", run)
    assert calls == [()]  # without reuse everything is simulated
    assert fresh.cost.tolist() == [5.0]
    same = stored_or_run(OmegaConf.create({"reuse": True, "add_policies": []}), "e3", run)
    assert sorted(same.policy) == ["a", "b"]
    assert len(calls) == 1  # nothing simulated
    merged = stored_or_run(OmegaConf.create({"reuse": True, "add_policies": ["b"]}), "e3", run)
    assert calls[-1] == ("b",)  # only the new policy is simulated ...
    assert dict(zip(merged.policy, merged.cost, strict=True)) == {
        "a": 1.0,
        "b": 5.0,
    }  # ... and replaces its rows


def test_noise_aware_crossings_drop_ties_within_the_measured_mad() -> None:
    """ADR 0022: an argmin change smaller than the noise is a tie, a clear one stays."""
    bw = np.array([1e5, 1e6, 1e7])
    lat = np.array([[10.0, 12.0], [10.2, 10.1], [30.0, 11.0]])  # rep 1 wins from 1e6 up
    assert len(phase.crossings(bw, lat)) == 1
    tight = np.full_like(lat, 0.01)
    loose = np.full_like(lat, 0.5)  # 2·√2·0.5 ≈ 1.4 ms > the 0.1 ms gap at 1e6
    assert len(phase.crossings(bw, lat, tight)) == 1
    assert phase.crossings(bw, lat, loose) == []
    assert phase.crossings(bw, lat, np.full_like(lat, np.nan)) == []  # unknown noise: no claim


def _old_block_bootstrap_ci(losses, blocks, rng, n_boot=2000, level=0.95):  # pre-ADR-0024 copy
    ids = np.unique(blocks)
    sums = np.array([losses[blocks == b].sum() for b in ids], dtype=np.float64)
    counts = np.array([(blocks == b).sum() for b in ids], dtype=np.float64)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def test_block_bootstrap_ci_matches_the_per_unit_mask_version() -> None:
    gen = np.random.default_rng(3)
    losses = gen.random(400)
    blocks = gen.choice(np.array(["K1", "K2", "KA1", "KI7", "KB3", "x"]), size=400)
    new = stats.block_bootstrap_ci(losses, blocks, np.random.default_rng(9), n_boot=300)
    old = _old_block_bootstrap_ci(losses, blocks, np.random.default_rng(9), n_boot=300)
    assert new == pytest.approx(old, rel=1e-12)  # bincount vs pairwise sum: last-ulp differences


def test_nb_p_is_larger_than_plain_p_and_equal_at_zero_ratio() -> None:
    v = np.array([0.31, 0.27, 0.35, 0.29, 0.33, 0.30])
    plain = stats.one_sided_p(v, 0.25)
    assert stats.nb_corrected_one_sided_p(v, 0.25, 0.0) == pytest.approx(plain)
    assert stats.nb_corrected_one_sided_p(v, 0.25, 0.5) > plain
    lo = stats.nb_corrected_one_sided_p(-v, -0.25, 0.0, greater=False)
    assert lo == pytest.approx(stats.one_sided_p(-v, -0.25, greater=False))
    assert stats.nb_corrected_one_sided_p(np.ones(5), 0.0, 0.3) == 1.0


def test_median_se_rule_counts_a_crossing_the_mad_rule_drops() -> None:
    import pandas as pd

    from kai.bench import experiments as ex

    a, b = REP_IDS[:2]
    rows = [
        {"load": 0.0, "bandwidth_bps": bw, "rep_id": r, "latency_ms_measured": m,
         "latency_ms_mad": 5.0, "latency_ms_model": m}
        for bw, ms in ((1e5, {a: 100.0, b: 110.0}), (1e6, {a: 110.0, b: 100.0}))
        for r, m in ms.items()
    ]  # fmt: skip
    frame = pd.DataFrame(rows)
    mad, _ = ex._boundaries(frame, MODEL, None, noise_aware=True)
    se, _ = ex._boundaries(frame, MODEL, None, median_se_n=25)
    assert len(mad) == 0
    assert len(se) == 1


def test_median_se_uses_the_delivered_count_per_point_when_stored() -> None:
    import pandas as pd

    from kai.bench import experiments as ex

    a, b = REP_IDS[:2]
    rows = [
        {"load": 0.0, "bandwidth_bps": bw, "rep_id": r, "latency_ms_measured": m,
         "latency_ms_mad": 5.0, "latency_ms_model": m, "n_delivered": 4}
        for bw, ms in ((1e5, {a: 100.0, b: 110.0}), (1e6, {a: 110.0, b: 100.0}))
        for r, m in ms.items()
    ]  # fmt: skip
    frame = pd.DataFrame(rows)
    assert len(ex._boundaries(frame, MODEL, None, median_se_n=25)[0]) == 0  # n=4 widens the SE
    older = frame.drop(columns="n_delivered")  # older E4 rows: falls back to n_windows
    assert len(ex._boundaries(older, MODEL, None, median_se_n=25)[0]) == 1
