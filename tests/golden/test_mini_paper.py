"""Mini-paper (PRD §9.6, §12.2 #9–#10): the whole pipeline on the tiny synthetic fixture.

`kai synth` → `kai folds` → E1 … E8, in a throw-away workspace. Guards against silent breakage of
any stage, checks that reruns are identical, and compares headline numbers with stored references
(`reference.json`; regenerate deliberately with `KAI_UPDATE_GOLDEN=1`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from kai import cli
from kai.analysis import plots
from kai.bench import experiments as ex
from kai.bench.config import load_config
from kai.bench.registry import content_hash, registry_path

pytestmark = [pytest.mark.xdist_group("mini"), pytest.mark.slow]
REFERENCE = Path(__file__).with_name("reference.json")
OVERRIDES = ["dataset=synthetic", "scenario=mini", "workers=2", "alphas=[0.05,0.1]"]
# headline numbers that must not drift silently, with the absolute tolerance allowed across machines
GOLDEN = {
    "e1/f_raw_macro_f1": 0.10,
    "e2/v0/a0.1/risk_mean": 0.06,
    "e2/v0/a0.1/bytes_vs_raw": 0.15,
    "e2/v2/a0.1/risk_mean": 0.06,
}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    work = tmp_path_factory.mktemp("work")
    old = os.environ.get("KAI_WORK")
    os.environ["KAI_WORK"] = str(work)
    try:
        cli.synth(seed=0, bearings_per_class=10, windows_per_condition=8)
        cli.folds(
            repeats=1, folds=2, level="L1", seed=0, epochs_raw=25, epochs_latent=12, f_raw="spec",
            head_trials=3, device="cpu", force=False, overrides=["dataset=synthetic"],
        )  # fmt: skip
        cfg = load_config(OVERRIDES)
        out = {
            "e1": ex.e1_profile(cfg, repeats=5),
            "e2": ex.e2_certificate(cfg),
            "e3": ex.e3_main(cfg),
            "e5": ex.e5_ablation(cfg),
            "e6": ex.e6_drift(cfg),
            "e8": ex.e8_overhead(cfg, n=300),
            "e7": ex.e7_sim2real(cfg, n_windows=40),
            "e4": ex.e4_phase(cfg, n_windows=6, grid_points=5, loads=(0.0,)),
            "e2_again": ex.e2_certificate(cfg),
        }
        out["_work"] = {
            "path": str(work),
            "registry_hash": content_hash(),
            "figures": [str(p) for p in plots.make_all()],
        }
        yield out
    finally:
        if old is None:
            del os.environ["KAI_WORK"]
        else:
            os.environ["KAI_WORK"] = old


def test_every_stage_ran_and_left_its_artifacts(results: dict) -> None:
    work = Path(results["_work"]["path"])
    for name in ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"):
        assert (work / "experiments" / name / "summary.json").exists(), name
    registry = pd.read_parquet(work / "experiments" / "registry.parquet")
    assert set(registry.experiment) == {"e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"}
    assert registry.lock_hash.nunique() == 1
    assert registry.config_hash.nunique() == 1
    assert (work / "artifacts" / "profiles" / "local.json").exists()
    assert len(list((work / "artifacts" / "bundles").glob("*/manifest.json"))) == 1
    figures = [Path(p) for p in results["_work"]["figures"]]
    assert len(figures) == len(plots.FIGURES)  # every figure was drawn from the stored results
    assert all(p.stat().st_size > 5000 for p in figures)


def test_e1_profiles_the_whole_menu(results: dict) -> None:
    e1 = results["e1"]
    assert e1["n_reps_profiled"] == 9
    assert e1["n_folds"] == 2
    assert e1["f_raw_macro_f1"] > e1["majority_macro_f1"] + 0.3
    assert 0 < e1["t_probe_ms"] < 5
    assert 0 < e1["t_sel_ms"] < 5


def test_e2_reports_every_variant_and_respects_ordering(results: dict) -> None:
    e2 = results["e2"]
    for alpha in ("0.05", "0.1"):
        v0, v1 = e2[f"v0/a{alpha}/risk_mean"], e2[f"v1/a{alpha}/risk_mean"]
        assert 0 <= v1 <= v0 <= 1  # the PAC threshold is never looser than the CRC one
        assert e2[f"v0/a{alpha}/risk_ci_lo"] <= v0 <= e2[f"v0/a{alpha}/risk_ci_hi"]
        assert 0 < e2[f"v0/a{alpha}/bytes_vs_raw"] <= 1
        assert e2[f"v2/a{alpha}/bytes_vs_raw"] <= e2[f"v0/a{alpha}/bytes_vs_raw"] + 1e-9
    assert isinstance(e2["H2a_pass"], bool)


def test_e3_compares_every_policy_and_orders_them_sanely(results: dict) -> None:
    work = Path(results["_work"]["path"])
    frame = pd.read_parquet(work / "experiments" / "e3" / "results.parquet")
    assert frame.policy.nunique() == 15
    assert frame.fold.nunique() == 2
    by = frame.groupby("policy")
    assert by.risk_sent.mean()["b1_raw"] == 0  # raw never disagrees with itself
    assert by.risk_sent.mean()["oracle"] == 0  # nor does the oracle
    assert by.cost.sum()["oracle"] <= by.cost.sum()["kai_s1_v0"] * 1.02
    assert by.rate_uncertified.mean()["kai_s1_v0"] == 0  # fidelity_first never sends uncertified
    assert by.rate_deferred.mean()["kai_s1_v0_live"] == 0  # liveness_first never defers
    share = frame[[c for c in frame.columns if c.startswith("share/")]].sum(axis=1)
    assert (share <= 1 + 1e-9).all()
    for key in ("H1_pass", "H3_pass", "H3/cost_reduction_vs_b4", "H1/b4_extra_cost_vs_oracle"):
        assert key in results["e3"]


def test_e5_ablations_behave_as_designed(results: dict) -> None:
    e5 = results["e5"]
    # a restricted menu can never be cheaper than the full one under the same certificate
    assert e5["cost_total/compression_only"] >= e5["cost_total/kai_s1_v0"] * 0.999
    assert e5["cost_total/learned_only"] >= e5["cost_total/kai_s1_v0"] * 0.999
    assert e5["cost_total/oracle"] <= e5["cost_total/kai_s1_v0"] * 1.02
    assert "H4_pass" in e5
    assert "H5_pass" in e5


def test_real_socket_experiments_produce_measurements(results: dict) -> None:
    e7, e4, e8 = results["e7"], results["e4"], results["e8"]
    assert len(e7["rows"]) == 3
    assert all(r["delivered_real"] > 0.9 for r in e7["rows"])
    assert 0 <= e7["min_selection_agreement"] <= 1
    assert e4["n_boundaries_theory"] >= 1
    assert e8["s1_greedy_safe/decisions_per_s"] > 1000
    assert e8["s2_mpc_h3/decisions_per_s"] > 100
    assert results["e6"]["L1/n_folds"] == 2


def test_same_seed_and_config_reproduce_identical_results(results: dict) -> None:
    assert results["e2"] == results["e2_again"]
    registry = pd.read_parquet(registry_path_in(results))
    e2_rows = registry[registry.experiment == "e2"]
    assert len(e2_rows) == 2
    assert e2_rows.metrics.nunique() == 1  # identical metrics, different run ids
    assert e2_rows.run_id.nunique() == 2
    assert len(results["_work"]["registry_hash"]) == 64


def registry_path_in(results: dict) -> Path:
    return Path(results["_work"]["path"]) / "experiments" / "registry.parquet"


def test_headline_numbers_match_the_stored_references(results: dict) -> None:
    got = {k: float(results[k.split("/", 1)[0]][k.split("/", 1)[1]]) for k in GOLDEN}
    if os.environ.get("KAI_UPDATE_GOLDEN") or not REFERENCE.exists():
        REFERENCE.write_text(json.dumps(got, indent=1) + "\n")
    want = json.loads(REFERENCE.read_text())
    for key, tol in GOLDEN.items():
        assert got[key] == pytest.approx(want[key], abs=tol), key
    assert registry_path  # imported API stays available to callers
