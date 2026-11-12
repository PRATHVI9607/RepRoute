"""ADR 0028 §2, §3 C/D: feasible-cell comparisons, crossover k*, measured slow-edge profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from kai.bench import experiments as ex


def rows(fold: str, policy: str, cost: float, util: float, n_cells: int = 2) -> list[dict]:
    return [
            {
                "fold": fold, "policy": policy, "link": "static", "bandwidth_bps": 1e6 * (i + 1),
                "deadline_ms": 250.0, "load": 0.0, "n_sensors": 1, "alpha": 0.05,
                "cost": cost, "cost_cap5": cost - 1.0, "cost_cap20": cost + 1.0,
                "cost_bytes": cost / 2, "cost_energy_cpu": cost / 4, "cost_energy_link": 0.0,
                "cost_cpu": cost / 4, "cost_late": 0.0, "edge_util": util, "slo_violation": 0.0,
            }
        for i in range(n_cells)
    ]  # fmt: skip


def frame_of(spec: dict[str, tuple[float, float]], folds=("f0", "f1", "f2")) -> pd.DataFrame:
    return pd.DataFrame([r for f in folds for p, (c, u) in spec.items() for r in rows(f, p, c, u)])


def test_feasible_reduction_uses_only_cells_feasible_for_both() -> None:
    frame = frame_of({"kai_s1_v0": (5.0, 0.5), "b14_edge_inference": (10.0, 0.5)})
    # in the first cell of f0 b14 is overloaded: that cell must not count for either policy
    bad = (
        (frame.policy == "b14_edge_inference") & (frame.fold == "f0") & (frame.bandwidth_bps == 1e6)
    )
    frame.loc[bad, ["edge_util", "cost"]] = [3.0, 1000.0]
    r = ex.feasible_reduction(
        frame, "kai_s1_v0", "b14_edge_inference", rng=np.random.default_rng(0)
    )
    assert r["per_fold"] == pytest.approx([0.5, 0.5, 0.5])
    assert r["n_cells"] == 5
    assert r["structural"] is True  # no spread: the CI is a point
    assert r["missing_folds"] == []


def test_feasible_reduction_lists_a_fold_without_a_shared_feasible_cell() -> None:
    frame = frame_of({"a": (5.0, 0.5), "b": (10.0, 0.5)})
    frame.loc[(frame.policy == "b") & (frame.fold == "f1"), "edge_util"] = 1.0  # rho >= 1
    r = ex.feasible_reduction(frame, "a", "b")
    assert r["folds"] == ["f0", "f2"]
    assert r["missing_folds"] == ["f1"]


def test_cap_and_energy_variants_reprice_the_same_cells() -> None:
    frame = frame_of({"a": (5.0, 0.5), "b": (10.0, 0.5)})
    cap5 = ex.feasible_reduction(frame, "a", "b", "cap5")["mean"]
    cap20 = ex.feasible_reduction(frame, "a", "b", "cap20")["mean"]
    assert cap5 == pytest.approx(1 - 4 / 9)
    assert cap20 == pytest.approx(1 - 6 / 11)
    # energy x3: cost + 2 * cost_energy_cpu = 5 + 2.5 = 7.5 against 10 + 5 = 15
    assert ex.feasible_reduction(frame, "a", "b", 3.0)["mean"] == pytest.approx(0.5)
    assert ex.feasible_reduction(frame, "a", "b", 1.0)["mean"] == pytest.approx(0.5)


def test_k_star_interpolates_in_log_k_and_labels_the_ends() -> None:
    ks = np.array([1.0, 2.0, 4.0, 8.0])
    val, label = ex.k_star(ks, np.array([-0.2, -0.1, 0.1, 0.3]))
    assert label == "crossing"
    assert val == pytest.approx(2.0 * 2.0**0.5)  # root halfway between k=2 and k=4 in log k
    assert ex.k_star(ks, np.array([0.1, 0.2, 0.3, 0.4])) == (1.0, "<= 1")
    assert ex.k_star(ks, np.array([-0.1, -0.2, -0.1, -0.3])) == (None, "> 8")
    # the last k must stay positive: an earlier excursion does not make k* smaller
    val, _ = ex.k_star(ks, np.array([0.1, -0.1, 0.1, 0.2]))
    assert val == pytest.approx(2.0 * np.sqrt(2.0))
    assert ex.k_star(ks, np.full(4, np.nan)) == (None, "undefined")
    # undefined points are skipped, not counted as negative
    assert ex.k_star(ks, np.array([-0.1, 0.1, np.nan, np.nan]))[1] == "crossing"
    assert ex.sign_changes(ks, np.array([-0.1, 0.1, -0.1, 0.1])) == pytest.approx(
        [np.sqrt(2.0), 2.0 * np.sqrt(2.0), 4.0 * np.sqrt(2.0)]
    )


def test_kstar_summary_is_seeded_and_reports_fold_range() -> None:
    ks = (1.0, 2.0, 4.0, 8.0)
    delta = np.array([[-0.2, -0.1, 0.1, 0.3], [-0.2, 0.1, 0.1, 0.3], [-0.3, -0.2, -0.1, 0.4]])
    a = ex.kstar_summary(ks, ["f0", "f1", "f2"], delta, seed=0, name="kai")
    b = ex.kstar_summary(ks, ["f0", "f1", "f2"], delta, seed=0, name="kai")
    assert a == b
    assert a["k_star_fold_range"][0] < a["k_star_mean_curve"] < a["k_star_fold_range"][1]
    assert (
        a["k_star_ci_lo"] <= a["k_star_mean_curve"] <= a["k_star_ci_hi"]
        or a["k_star_boot_n_numeric"]
    )
    assert len(a["delta_per_fold"]) == 4
    assert len(a["delta_per_fold"][0]) == 3


def test_c_summary_reports_shares_terms_sensitivities_and_structural() -> None:
    frame = frame_of(
        {"kai_s1_v0": (5.0, 0.5), "b14_edge_inference": (10.0, 1.5), "b4_best_single": (8.0, 0.2)}
    )
    s = ex.c_summary(frame, seed=0, block=32, k=3.0)
    assert s["infeasible_share/b14_edge_inference"] == 1.0
    assert s["infeasible_share/kai_s1_v0"] == 0.0
    assert np.isnan(s["cost_terms/b14_edge_inference/cost"])  # no feasible cell, no cost
    assert s["cost_terms/kai_s1_v0/cost_bytes"] == 2.5
    assert np.isnan(s["cmp/kai_s1_v0/vs_b14/mean"])  # nothing to compare over
    assert s["cmp/kai_s1_v0/vs_b14/missing_folds"] == ["f0", "f1", "f2"]
    assert s["cmp/kai_s1_v0/vs_b4/mean"] == pytest.approx(1 - 5 / 8)
    assert s["cmp/kai_s1_v0/vs_b4/structural"] is True
    assert s["cap5/kai_s1_v0/vs_b4/mean"] == pytest.approx(1 - 4 / 7)
    assert {f"energy_e{e:g}/kai_s1_v0/vs_b4/mean" for e in (0.3, 1.0, 3.0)} <= set(s)
    assert "cmp/kai_s1_v0_classical/vs_b4/mean" not in s  # policy absent


def test_ci_rng_depends_only_on_its_own_block_alpha_and_variant() -> None:
    a = ex.ci_rng(0, 2, 0.05, "v0").random(3)
    assert (ex.ci_rng(0, 2, 0.05, "v0").random(3) == a).all()
    for other in (
        ex.ci_rng(0, 2, 0.05, "v1"),
        ex.ci_rng(0, 2, 0.1, "v0"),
        ex.ci_rng(0, 3, 0.05, "v0"),
    ):
        assert (other.random(3) != a).any()


def test_adding_an_exploratory_policy_leaves_the_preregistered_ci_unchanged() -> None:
    base = frame_of({"kai_s1_v0": (5.0, 0.5), "b4_best_single": (10.0, 0.5)}, folds=tuple("abcde"))
    rng = np.random.default_rng(1)
    base["cost"] *= 1 + 0.2 * rng.standard_normal(len(base))  # spread, so BCa really resamples
    extra = pd.concat([base, base[base.policy == "kai_s1_v0"].assign(policy="kai_s1_v0_late")])
    r0 = ex.red(base, "kai_s1_v0", "b4_best_single", 0, 3)
    ex.x_reduction(extra, "kai_s1_v0_late", "b4_best_single", 0, 3)
    r1 = ex.red(extra, "kai_s1_v0", "b4_best_single", 0, 3)
    assert (r0["ci_lo"], r0["ci_hi"]) == (r1["ci_lo"], r1["ci_hi"])


def test_x_reduction_skips_folds_one_policy_lacks_and_names_them() -> None:
    frame = frame_of({"a": (5.0, 0.5), "b": (10.0, 0.5)})
    frame = frame[~((frame.policy == "a") & (frame.fold == "f2"))]
    with pytest.raises(ValueError, match="no result in some folds"):
        ex.red(frame, "a", "b", 0, 3)  # pre-registered reductions stay strict
    r, missing = ex.x_reduction(frame, "a", "b", 0, 3)
    assert missing == ["f2"]
    assert r["folds"] == ["f0", "f1"]
    assert r["mean"] == pytest.approx(0.5)
    out: dict[str, Any] = {}
    ex.put_x(out, "X/a", r["mean"], missing)
    assert out["X/a/missing_folds"] == ["f2"]
    none, miss = ex.x_reduction(frame[frame.policy == "b"], "a", "b", 0, 3)
    assert np.isnan(none["mean"])
    assert miss == ["f0", "f1", "f2"]


# --- profiles (D) -------------------------------------------------------------------------------


def write_profile(path: Path, enc: float, head: float, probe: float, dec: float = 1.0) -> None:
    prof = [
        {"rep_id": r, "size_mean_bytes": 10, "size_q95_bytes": 12,
         "t_enc_ms": enc * (i + 1), "t_enc_mad_ms": 0.1 * enc,
         "t_enc_q95_ms": 2 * enc, "t_dec_ms": dec, "t_head_ms": head * (i + 1)}
        for i, r in enumerate(("r0_raw", "r1"))
    ]  # fmt: skip
    ov = {"t_probe_ms": probe, "t_pred_ms": probe, "t_sel_ms": probe}
    path.write_text(json.dumps({"profiles": prof, "overheads": ov}))


@pytest.fixture
def profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    d = tmp_path / "artifacts" / "profiles"
    d.mkdir(parents=True)
    write_profile(d / "local.json", enc=1.0, head=0.5, probe=0.1)
    write_profile(d / "slow400.json", enc=8.0, head=4.0, probe=0.8, dec=99.0)
    return d


def test_profile_path_follows_kai_profile_but_reads_stay_local(profiles, monkeypatch) -> None:
    monkeypatch.delenv("KAI_PROFILE", raising=False)
    assert ex.profile_path() == profiles / "local.json"
    monkeypatch.setenv("KAI_PROFILE", "slow400")
    assert ex.profile_path() == profiles / "slow400.json"
    assert ex.local_profile_path() == profiles / "local.json"
    got, _ = ex.load_profiles()  # E4 / E7 / bundles never read the slow profile
    assert got[0].t_enc_ms == 1.0


def test_edge_profile_replaces_edge_times_only(profiles) -> None:
    got, ov = ex.load_profiles(edge_profile=profiles / "slow400.json")
    assert [p.t_enc_ms for p in got] == [8.0, 16.0]
    assert got[0].t_enc_q95_ms == 16.0
    assert ov.t_probe_ms == 0.8
    assert got[0].t_dec_ms == 1.0
    assert got[0].t_head_ms == 0.5
    heads, decs = ex.edge_head_ms(profiles / "slow400.json", ("r1", "r0_raw"))
    assert heads.tolist() == [8.0, 4.0]  # in the order asked for, not the file's
    assert len(decs) == 2


def test_measured_slowdown_is_the_median_over_operations(profiles) -> None:
    s, ratios = ex.slowdown_ratios(profiles / "local.json", profiles / "slow400.json")
    assert s == 8.0
    assert ratios["t_enc/r1"] == 8.0
    assert ratios["t_head/r0_raw"] == 8.0
    assert ratios["t_probe"] == 8.0


# --- the experiments, with the simulator stubbed -------------------------------------------------


def c_frame(k: float) -> pd.DataFrame:
    """Kai cost 5 always; b14 beats Kai below k = 5 (cost 4), loses above (cost 8) and is
    overloaded (and so has no cost) from k = 30."""
    b14_cost, b14_util = (4.0 if k < 5 else 8.0), (0.5 if k < 30 else 2.0)
    return frame_of(
        {
            "kai_s1_v0": (5.0, 0.5), "b14_edge_inference": (b14_cost, b14_util),
            "b4_best_single": (9.0, 0.2), "b1_raw": (20.0, 0.1), "oracle": (3.0, 0.5),
        }
    )  # fmt: skip


def test_e3k_writes_one_run_per_k_and_the_crossover(monkeypatch) -> None:
    from kai.bench.config import load_config

    written: dict[str, dict] = {}
    monkeypatch.setattr(ex, "run_grid", lambda cfg, cells, which, k, **kw: c_frame(k))
    monkeypatch.setattr(ex, "grid_cells", lambda cfg, alphas=None: [])
    monkeypatch.setattr(ex, "rss_summary", lambda cfg: {"rss/f_raw": 5})
    monkeypatch.setattr(
        ex, "finish", lambda name, cfg, summary, frame=None, **kw: written.setdefault(name, summary)
    )
    cfg = load_config()
    ex.e3k_crossover(cfg)
    assert written["e3_kstar"]["rss/f_raw"] == 5
    assert set(written) == {f"e3_k{k:g}" for k in ex.K_GRID} | {"e3_kstar"}
    assert written["e3_k10"]["p_cpu_w"] == pytest.approx(float(cfg.profile.energy.p_cpu_w) / 10)
    assert written["e3_k30"]["infeasible_share/b14_edge_inference"] == 1.0
    star = written["e3_kstar"]["kai_s1_v0"]
    expected = float(np.exp(np.log(3.0) + 0.4 * np.log(5.0 / 3.0)))  # Δ = -0.25 at 3, +0.375 at 5
    assert star["k_star_mean_curve"] == pytest.approx(expected)
    assert star["k_star_per_fold"] == pytest.approx([expected] * 3)
    assert star["k_star_ci_lo"] == pytest.approx(expected)  # folds identical: a point CI
    assert star["delta_mean"][-1] is None or np.isnan(star["delta_mean"][-1])  # k=100 undefined


def test_e3d_uses_the_measured_slowdown_for_power_and_the_slow_profile(
    profiles, monkeypatch
) -> None:
    from kai.bench.config import load_config

    seen: dict[str, Any] = {}
    written: dict[str, dict] = {}

    def fake_grid(cfg, cells, which, k, **kw):
        seen.update(kw, k=k)
        return c_frame(2.0)

    monkeypatch.setattr(ex, "run_grid", fake_grid)
    monkeypatch.setattr(ex, "grid_cells", lambda cfg, alphas=None: [])
    monkeypatch.setattr(
        ex, "finish", lambda name, cfg, summary, frame=None, **kw: written.setdefault(name, summary)
    )
    cfg = load_config()
    ex.e3d_measured(cfg)
    assert seen["edge_profile"] == profiles / "slow400.json"
    assert seen["k"] == 1.0
    s = written["e3_slow400"]
    assert s["slowdown_s"] == 8.0
    assert s["p_cpu_w"] == pytest.approx(float(cfg.profile.energy.p_cpu_w) / 8.0)
    assert s["slowdown_quantiles_min_q25_q50_q75_max"][2] == 8.0
    assert s["cmp/kai_s1_v0/vs_b14/mean"] == pytest.approx(1 - 5 / 4)


def test_e3d_without_a_slow_profile_says_what_to_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="KAI_PROFILE=slow400"):
        ex.e3d_measured(OmegaConf.create({}))


def test_e3s_covers_b1_and_names_folds_a_policy_lacks(monkeypatch) -> None:
    from kai.bench.config import load_config

    assert "b1_raw" in ex.SLOW_POLICIES
    written: dict[str, dict] = {}
    frame = frame_of({"kai_s1_v0": (5.0, 0.5), "b4_best_single": (9.0, 0.5), "b1_raw": (20.0, 0.5)})
    frame = frame[~((frame.policy == "b1_raw") & (frame.fold == "f2"))]  # b1 missing on one fold
    monkeypatch.setattr(ex, "run_grid", lambda *a, **kw: frame)
    monkeypatch.setattr(ex, "grid_cells", lambda cfg, alphas=None: [])
    monkeypatch.setattr(
        ex, "finish", lambda name, cfg, summary, frame=None, **kw: written.setdefault(name, summary)
    )
    ex.e3s_slow_edge(load_config())
    s = written["e3_slow10"]
    assert s["kai_s1_v0/cost_reduction_vs_b1"] == pytest.approx(1 - 5 / 20)
    assert s["kai_s1_v0/cost_reduction_vs_b1/missing_folds"] == ["f2"]
    assert s["kai_s1_v0/cost_reduction_vs_b4"] == pytest.approx(1 - 5 / 9)
    assert written["e3_slow100"]["p_cpu_w"] == pytest.approx(0.01)


def test_kstar_mean_curve_uses_only_folds_defined_at_every_k() -> None:
    ks = (1.0, 2.0, 4.0)
    nan = np.nan
    delta = np.array([[-0.2, 0.1, 0.3], [nan, 0.1, 0.3], [-0.4, 0.3, 0.5]])
    a = ex.kstar_summary(ks, ["f0", "f1", "f2"], delta, seed=0, name="kai", n_boot=20)
    assert a["n_folds_used"] == 2
    assert a["folds_excluded"] == ["f1"]
    assert a["n_folds_defined_per_k"] == [2, 3, 3]
    assert a["delta_mean"] == pytest.approx([-0.3, 0.2, 0.4])  # f1 not averaged in at k = 2, 4


def test_code_hash_ignores_report_and_harness_code(monkeypatch, tmp_path) -> None:
    root = Path(ex.__file__).resolve().parents[1]
    hashed = {p.relative_to(root).as_posix() for pat in ex._SIM_FILES for p in root.glob(pat)}
    assert "bench/evaluate.py" in hashed
    assert "bench/calibrate.py" in hashed
    assert "link/des.py" in hashed
    assert "certify/crc.py" in hashed
    assert "bench/experiments.py" not in hashed
    assert "bench/realrun.py" not in hashed


@pytest.mark.parametrize("fn", ["e4_phase", "e7_sim2real", "e7d_docker"])
def test_phase_and_real_runs_refuse_a_slow_edge_profile(monkeypatch, fn) -> None:
    monkeypatch.setenv("KAI_PROFILE", "slow400")
    with pytest.raises(RuntimeError, match="laptop profile"):
        getattr(ex, fn)(None)


def test_rss_is_stored_in_the_kstar_summary(monkeypatch) -> None:
    monkeypatch.setattr(ex, "fold_dirs", lambda cfg, level="L1": [Path("f0")])
    import kai.profile.memory as mem

    monkeypatch.setattr(
        mem, "peak_rss_bytes", lambda kind, d: {"f_raw": 7, "kai_classical": 3}[kind]
    )
    monkeypatch.setattr(mem, "baseline_rss_bytes", lambda: 1)
    assert ex.rss_summary(None) == {  # type: ignore[arg-type]
        "rss/f_raw": 7, "rss/kai_classical": 3, "rss/baseline": 1,
    }  # fmt: skip


def test_e3k_reduced_grid_runs_only_those_k_and_folds(monkeypatch) -> None:
    from omegaconf import OmegaConf

    from kai.bench.config import load_config

    written: dict[str, dict] = {}
    seen: list[int] = []

    def fake_grid(cfg, cells, which, k, **kw):
        seen.append(kw["max_folds"])
        return c_frame(k)

    monkeypatch.setattr(ex, "run_grid", fake_grid)
    monkeypatch.setattr(ex, "grid_cells", lambda cfg, alphas=None: [])
    monkeypatch.setattr(ex, "rss_summary", lambda cfg: {})
    monkeypatch.setattr(
        ex, "finish", lambda name, cfg, summary, frame=None, **kw: written.setdefault(name, summary)
    )
    cfg = load_config()
    ex.e3k_crossover(OmegaConf.merge(cfg, {"k_grid": [1, 10, 100], "max_folds": 1}))
    assert set(written) == {"e3_k1", "e3_k10", "e3_k100", "e3_kstar"}
    assert seen == [1, 1, 1]
    assert written["e3_kstar"]["k_grid"] == [1.0, 10.0, 100.0]
    assert written["e3_kstar"]["max_folds"] == 1
    with pytest.raises(ValueError, match="subset"):
        ex.e3k_crossover(OmegaConf.merge(cfg, {"k_grid": [4]}))
