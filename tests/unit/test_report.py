"""scripts/report.py helpers (the script is not a package: load it by path)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "report", Path(__file__).parents[2] / "scripts" / "report.py"
)
report = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(report)  # type: ignore[union-attr]


def test_window_period_prefers_e8_then_the_window_set(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    (tmp_path / "data" / "processed").mkdir(parents=True)
    np.savez(
        tmp_path / "data" / "processed" / "w.npz",
        x=np.zeros((3, 200), np.int16),
        sample_rate_hz=100.0,
    )
    assert report.window_period_ms() == 2000.0
    (tmp_path / "experiments" / "e8").mkdir(parents=True)
    (tmp_path / "experiments" / "e8" / "summary.json").write_text(
        json.dumps({"window_period_ms": 640.0})
    )
    assert report.window_period_ms() == 640.0


def test_seed_table_joins_on_representation_not_position(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "data" / "splits" / "d_L1_x.json").write_text("{}")
    # (seed, fold) -> disagreement per rep; "b" is listed first, so first-appearance order is b, a
    values = {
        (0, 0): {"b": 0.5, "a": 0.1},
        (0, 1): {"b": 0.5, "a": 0.3},
        (1, 0): {"b": 0.6, "a": 0.2},
    }
    for (seed, fold), m in values.items():
        d = tmp_path / "artifacts" / "folds" / "x" / f"s{seed}_r0f{fold}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(
            json.dumps({f"disagreement/test/{k}": v for k, v in m.items()})
        )
    t = report.seed_variance().set_index("representation")
    assert t.loc["a", "sd over folds (seed 0)"] > 0.1
    assert t.loc["b", "sd over folds (seed 0)"] == 0.0


def test_budget_sweep_picks_the_cheapest_policy_that_fits() -> None:
    s = {
        "X26/required_bytes/b4_best_single": 1000.0,
        "X26/required_bytes/kai_s1_v0": 5000.0,
        "X26/required_bytes/kai_s1_v0_committee": 9000.0,
        "X26/required_bytes/b14_edge_inference": 90000.0,
        "X26/required_bytes/oracle": 1.0,  # not deployable: never listed
        "X26/size/f_raw": 80000.0,
    }
    by = pd.DataFrame(
        {
            "policy": ["b4_best_single", "kai_s1_v0", "kai_s1_v0_committee", "b14_edge_inference"],
            "cost_total_per_fold": [10.0, 8.0, 6.0, 7.0],
        }
    )
    text = "\n".join(report.budget_sweep(s, by))
    assert "| 1000 | 1 | b4_best_single | 10 |" in text
    assert "| 5000 | 2 | kai_s1_v0 | 8 |" in text
    assert "best committee policy cheaper" in text  # at 9000 the committee beats the rest
    assert "b14 becomes available" in text
    assert "| f_raw | 80000 |" in text
    assert "| oracle |" not in text
    assert report.budget_sweep({}, by) == []


def test_h6_verdict_of_record_is_the_first_e4_registry_row() -> None:
    def row(exp: str, passed: float) -> dict:
        return {"experiment": exp, "metrics": json.dumps({"H6_pass": passed})}

    reg = pd.DataFrame([row("e3", 0.0), row("e4", 0.0), row("e4", 1.0)])
    assert report.h6_of_record(reg, {"H6_pass": True}) is False  # the later pass changes nothing
    assert report.h6_of_record(reg[reg.experiment == "e3"], {"H6_pass": True}) is True
    assert report.h6_of_record(None, {"H6_pass": False}) is False


def _c_summary(k: float) -> dict:
    s: dict = {"edge_slowdown": k, "p_cpu_w": 1.0 / k}
    for p in report.C_POLICIES:
        s[f"infeasible_share/{p}"] = 0.25
        for t in report.COST_TERMS:
            s[f"cost_terms/{p}/{t}"] = 1.5
    for ours in report.C_OURS:
        for ref in report.C_SHORT:
            tag = f"{ours}/vs_{ref}"
            s |= {f"cmp/{tag}/{key}": v for key, v in {
                "mean": 0.25, "ci_lo": 0.2, "ci_hi": 0.3, "structural": False,
                "n_cells": 40, "per_fold": [0.2, 0.3], "folds": ["f0", "f1"],
            }.items()}  # fmt: skip
            s |= {f"cap{c}/{tag}/mean": 0.1 for c in (5, 20)}
            s |= {f"energy_e{e}/{tag}/mean": 0.2 for e in (0.3, 1, 3)}
    return s


def test_slow_edge_section_is_empty_without_summaries_and_prints_stored_numbers(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    assert report.slow_edge_section() == []
    exp = tmp_path / "experiments"
    for k in (1.0, 10.0):
        (exp / f"e3_k{k:g}").mkdir(parents=True)
        (exp / f"e3_k{k:g}" / "summary.json").write_text(json.dumps(_c_summary(k)))
    star = {
        "k_grid": [1.0, 10.0],
        "kai_s1_v0": {
            "k_star_mean_curve": 3.7, "k_star_mean_curve_label": "crossing",
            "k_star_ci_lo": 2.5, "k_star_ci_hi": 4.0, "k_star_boot_share_beyond_grid": 0.125,
            "k_star_per_fold": [3.0, None], "k_star_per_fold_label": ["crossing", "> 10"],
            "k_star_fold_range": [3.0, 3.0], "sign_changes_mean_curve": [3.7],
            "delta_mean": [-0.1, 0.2], "delta_per_fold": [[-0.1, -0.1], [0.2, 0.2]],
            "folds": ["f0", "f1"],
        },
    }  # fmt: skip
    (exp / "e3_kstar").mkdir()
    (exp / "e3_kstar" / "summary.json").write_text(json.dumps(star))
    text = "\n".join(report.slow_edge_section())
    assert "exploratory" in text
    assert "k = 10 (CPU power ÷ 10)" in text
    assert "3.7" in text
    assert "> 10" in text
    assert "0.125" in text
    assert "Measured slow edge" not in text  # no slow400 summary yet
    assert "kai_s1_v0_classical" in text  # comparisons of both Kai variants


def test_budget_sweep_uses_peak_rss_for_b14_and_classical_when_present() -> None:
    s = {
        "X26/required_bytes/b14_edge_inference": 100.0,
        "X26/required_bytes/kai_s1_v0_classical": 10.0,
        "X26/required_bytes/b4_best_single": 50.0,
        "rss/f_raw": 5000.0, "rss/kai_classical": 700.0, "rss/baseline": 30000.0,
    }  # fmt: skip
    by = pd.DataFrame(
        {
            "policy": ["b14_edge_inference", "kai_s1_v0_classical", "b4_best_single"],
            "cost_total_per_fold": [1.0, 2.0, 3.0],
        }
    )
    text = "\n".join(report.budget_sweep(s, by))
    assert "| 5000 |" in text
    assert "| 700 |" in text
    assert "| 100 |" not in text
    assert "peak RSS above a numpy-only Python" in text
    no_rss = "\n".join(report.budget_sweep({k: v for k, v in s.items() if "rss" not in k}, by))
    assert "| 100 |" in no_rss
    assert "peak RSS" not in no_rss


def test_h6_numbers_come_from_the_first_e4_run_and_the_latest_is_separate() -> None:
    def row(exp: str, theory: float) -> dict:
        m = {"H6_pass": True, "n_boundaries_theory": theory, "n_boundaries_measured": 3.0,
             "n_boundaries_matched": 2.0, "max_rel_error": 0.1}  # fmt: skip
        return {"experiment": exp, "metrics": json.dumps(m)}

    reg = pd.DataFrame([row("e3", 99.0), row("e4", 4.0), row("e4", 6.0)])
    latest = dict.fromkeys(report.H6_KEYS, 6.0)
    first, again = report.h6_numbers(reg, latest)
    assert first["n_boundaries_theory"] == 4.0
    assert again == latest
    assert report.h6_numbers(None, latest) == (latest, None)


def test_comparison_tables_label_the_configured_cap() -> None:
    s = _c_summary(1.0) | {"late_cap_factor": 4.0}
    text = "\n".join(report._comparison_tables(s, "t"))
    assert "lateness cap 4·D" in text
    assert "configured 4·D" in text


def test_kstar_table_reports_folds_used_and_rss(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    star = {
        "k_grid": [2.0, 10.0], "rss/f_raw": 5000, "rss/kai_classical": 700, "rss/baseline": 30000,
        "kai_s1_v0": {
            "k_star_mean_curve": None, "k_star_mean_curve_label": "<= 2",
            "k_star_ci_lo": None, "k_star_ci_hi": None, "k_star_boot_share_beyond_grid": 0.0,
            "k_star_per_fold": [None], "k_star_per_fold_label": ["undefined"],
            "k_star_fold_range": None, "sign_changes_mean_curve": [],
            "delta_mean": [0.1, 0.2], "delta_per_fold": [[0.1, 0.1], [0.2, 0.2]],
            "folds": ["f0", "f1"], "n_folds_used": 1, "folds_excluded": ["f1"],
            "n_folds_defined_per_k": [1, 2],
        },
    }  # fmt: skip
    (tmp_path / "experiments" / "e3_kstar").mkdir(parents=True)
    (tmp_path / "experiments" / "e3_kstar" / "summary.json").write_text(json.dumps(star))
    text = "\n".join(report.slow_edge_section())
    assert "1/2" in text
    assert "f1" in text
    assert "<= 2" in text
    assert "5,000" in text
    assert "numpy-only baseline" in text


def test_stale_summaries_are_those_older_than_e1(tmp_path, monkeypatch) -> None:
    import os

    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    exp = tmp_path / "experiments"
    for name, t in (("e1", 200), ("e2", 300), ("e7d", 100), ("e3_wired_gateway", 199)):
        (exp / name).mkdir(parents=True)
        (exp / name / "summary.json").write_text("{}")
        os.utime(exp / name / "summary.json", (t, t))
    assert report.stale_summaries() == ["e3_wired_gateway", "e7d"]
    (exp / "e1" / "summary.json").unlink()
    assert report.stale_summaries() == []


def _conf_work(tmp_path, *, b4=0.30, b14=0.20, risks=None, h5=0.20, folds=5, e5=True) -> None:
    """Synthetic E3 k = 10, E2 and E5 outputs; per-fold values jitter around the given means."""
    jit = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    exp = tmp_path / "experiments"
    for name in ("e3_k10", "e2", "e5"):
        (exp / name).mkdir(parents=True, exist_ok=True)
    c = {
        f"cmp/kai_s1_v0/vs_{r}/per_fold": (m + jit)[:folds].tolist()
        for r, m in (("b4", b4), ("b14", b14))
    }
    (exp / "e3_k10" / "summary.json").write_text(json.dumps(c))
    (exp / "e2" / "summary.json").write_text("{}")
    risks = risks or {0.02: 0.025, 0.05: 0.055, 0.10: 0.105}
    rows = [
        {"variant": "v0", "alpha": a, "fold": f"f{i}", "risk": m + 0.2 * jit[i]}
        for a, m in risks.items()
        for i in range(5)
    ]
    pd.DataFrame(rows).to_parquet(exp / "e2" / "results.parquet")
    if e5:
        (exp / "e5" / "summary.json").write_text(
            json.dumps({"per_fold": {"H5": (h5 + jit).tolist()}})
        )


def _verdicts(text: str) -> dict[str, str]:
    rows = [r.split("|") for r in text.splitlines() if r.startswith("| S-H")]
    return {r[1].strip(): r[6].strip() for r in rows}


def test_confirmatory_all_supported_and_labelled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    assert report.confirmatory_section("X") == []
    _conf_work(tmp_path)
    text = "\n".join(report.confirmatory_section("SHHS", 0.25))
    assert "Confirmatory family — SHHS" in text
    assert set(_verdicts(text).values()) == {"**supported**"}


def test_confirmatory_h3_fails_when_only_b14_threshold_misses(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    _conf_work(tmp_path, b4=0.30, b14=0.05)  # (a) met, (b) < 0.10
    v = _verdicts("\n".join(report.confirmatory_section("SHHS", 0.25)))
    assert v["S-H3′"] == "not supported"
    assert v["S-H5"] == "**supported**"


def test_confirmatory_h2a_uses_max_over_alpha(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    # alpha = 0.10 sits above 0.11: mean risk 0.115 > alpha + 0.01 although 0.02 and 0.05 pass
    _conf_work(tmp_path, risks={0.02: 0.025, 0.05: 0.055, 0.10: 0.115})
    text = "\n".join(report.confirmatory_section("SHHS", 0.25))
    assert _verdicts(text)["S-H2a"] == "not supported"
    row = next(r for r in text.splitlines() if r.startswith("| S-H2a"))
    p = float(row.split("|")[3])
    assert p > 0.5  # the failing alpha dominates; passing alphas alone would give p << 0.05


def test_confirmatory_applies_holm(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    _conf_work(tmp_path, h5=0.105)  # H5 mean just above 0.10 with spread: raw p small, not tiny
    text = "\n".join(report.confirmatory_section("SHHS", 0.25))
    rows = {
        r.split("|")[1].strip(): r.split("|") for r in text.splitlines() if r.startswith("| S-H")
    }
    raw = {h: float(c[3]) for h, c in rows.items()}
    adj = report.stats.holm(raw)
    for h, c in rows.items():
        assert abs(float(c[4]) - adj[h][0]) < 1e-3 * max(1.0, adj[h][0]) + 1e-6
    assert any(float(rows[h][4]) > raw[h] for h in raw)  # at least one p was inflated


def test_confirmatory_incomplete_without_fold_or_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    _conf_work(tmp_path, folds=4)  # S-H3' has 4 folds
    text = "\n".join(report.confirmatory_section("SHHS", 0.25))
    assert "incomplete" in text
    assert "**supported**" not in text
    _conf_work(tmp_path, e5=False)
    (tmp_path / "experiments" / "e5" / "summary.json").unlink()
    text = "\n".join(report.confirmatory_section("SHHS", 0.25))
    assert _verdicts(text)["S-H5"] == "no verdict"
    assert "**supported**" not in text


def test_work_dataset_id_reads_the_single_split_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    splits = tmp_path / "data" / "splits"
    splits.mkdir(parents=True)
    (splits / "hmc-v1_L1_s0.json").write_text("{}")
    assert report.work_dataset_id() == "hmc-v1"


def test_everything_is_stale_when_e1_predates_the_folds(tmp_path, monkeypatch) -> None:
    import os

    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    for name in ("e1", "e3_kstar"):
        (tmp_path / "experiments" / name).mkdir(parents=True)
        (tmp_path / "experiments" / name / "summary.json").write_text("{}")
        os.utime(tmp_path / "experiments" / name / "summary.json", (300, 300))
    fold = tmp_path / "artifacts" / "folds" / "abc" / "s0_r0f0" / "metrics.json"
    fold.parent.mkdir(parents=True)
    fold.write_text("{}")
    os.utime(fold, (400, 400))
    assert report.stale_summaries() == ["e1", "e3_kstar"]
