"""Experiment-harness plumbing: replication unit, H7 on unmeasured rates, reuse provenance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from kai.bench import experiments as ex


def test_folds_stay_the_replication_unit_as_pre_registered() -> None:
    per_fold = pd.Series([0.1, 0.3, 0.5, 0.7], index=["s0_r0f0", "s0_r0f1", "s0_r1f0", "s0_r1f1"])
    units, unit = ex.replication_units(per_fold)
    assert unit == "folds"
    assert units.tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7])


def test_h7_fails_and_counts_a_rate_where_nothing_was_delivered() -> None:
    rows = [
        {"p50_rel_gap": 0.02, "selection_agreement": 1.0},
        {"p50_rel_gap": float("nan"), "selection_agreement": 1.0},
    ]
    summary = ex._h7_summary("loopback", rows)
    assert summary["n_rows_unmeasured"] == 1
    assert summary["H7_pass"] is False


def test_reuse_is_refused_when_the_config_or_split_changed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    monkeypatch.setattr(ex, "provenance", lambda cfg: {"config_hash": "new", "split_id": "s"})
    stored = ex.out_dir("e3") / "results.parquet"
    pd.DataFrame({"policy": ["a"], "cost": [1.0]}).to_parquet(stored)
    cfg = OmegaConf.create({"reuse": True})
    meta = stored.with_name("results.meta.json")
    meta.write_text(json.dumps({"config_hash": "old", "split_id": "s"}))
    with pytest.raises(ValueError, match="refusing to reuse"):
        ex.stored_or_run(cfg, "e3", lambda only: pd.DataFrame())
    meta.write_text(json.dumps({"config_hash": "new", "split_id": "s"}))
    assert ex.stored_or_run(cfg, "e3", lambda only: pd.DataFrame()).cost.tolist() == [1.0]
    assert np.isfinite(ex.load_stored(cfg, "e3").cost).all()


def test_e7_caps_real_time_per_rate() -> None:
    assert ex.e7_windows(64.0) == 300  # Paderborn
    assert ex.e7_windows(5120.0) == 300  # HAR: 351 fit in 30 min
    assert ex.e7_windows(30000.0) == 60  # Sleep-EDF epochs
    assert ex.e7_windows(1e6) == 60  # never below the floor


def test_gate_auc_separates_an_informative_predictor_from_noise(rng: np.random.Generator) -> None:
    """ADR 0022 gate: held-out AUC is high when φ predicts d, about 0.5 when it does not."""
    from kai.bench.calibrate import gate_auc

    n, groups = 600, np.repeat(np.arange(12), 50)
    phi = rng.standard_normal((n, 3))
    d = np.zeros((n, 3), bool)
    d[:, 1] = phi[:, 0] > 0.5  # learnable
    d[:, 2] = rng.random(n) < 0.3  # pure noise
    ids = ("r0_raw", "r1", "r2")
    informative = gate_auc(phi, d[:, :2], groups, ids[:2])
    noise = gate_auc(phi, d[:, [0, 2]], groups, ids[:2])
    assert informative > 0.9
    assert 0.35 < noise < 0.65
    assert np.isnan(gate_auc(phi, np.zeros((n, 2), bool), groups, ids[:2]))  # no outcome varies


def test_run_cell_keeps_the_default_probe_interval_and_passes_the_seed(monkeypatch) -> None:
    """ADR 0023: a positional `seed` once landed in `Scenario.probe_s`, so every grid cell
    forwarded cached raw back to back. The grid must use the same 1 s probe as the edge."""
    from types import SimpleNamespace

    from kai.bench import evaluate as ev
    from kai.link.des import Scenario

    seen: list[Scenario] = []
    monkeypatch.setattr(ev, "policies", lambda *a: {"p": lambda: (None, SimpleNamespace(aci=None))})
    monkeypatch.setattr(ev, "simulate", lambda sc, *a: seen.append(sc))
    monkeypatch.setattr(ev, "metrics", lambda *a: {})
    ctx = SimpleNamespace(
        workload=lambda n: None,
        window_period_ms=64.0,
        model=None,
        ghat_z=None,
        ghat_sel=None,
        ghat_d=None,
        model_edge=None,
        sizes={},
        rep_ids=(),
        committee=(),
    )
    cell = ev.Cell("static", 1e6, 100.0, 0.5, 8, 0.05, p_audit=0.02)
    ev.run_cell(ctx, cell, "fidelity_first", "main", 10, seed=7)
    (sc,) = seen
    assert sc.probe_s == Scenario.__dataclass_fields__["probe_s"].default == 1.0
    assert (sc.seed, sc.p_audit, sc.load, sc.n_sensors) == (7, 0.02, 0.5, 8)


def test_a_fold_missing_from_one_policy_is_an_error_not_a_zero_reduction() -> None:
    """ADR 0023: reuse + add_policies on a larger fold set once turned NaN/NaN into 0.0."""
    ours = pd.Series([1.0, 2.0, np.nan], index=["f0", "f1", "f2"], name="kai")
    theirs = pd.Series([2.0, 2.0, 3.0], index=["f0", "f1", "f2"], name="b4")
    with pytest.raises(ValueError, match="no result in some folds"):
        ex.fold_reduction(ours, theirs)
    got = ex.fold_reduction(pd.Series([1.0, 0.0, 1.0]), pd.Series([2.0, 0.0, 0.0]))
    assert got.tolist() == [0.5, 0.0, -np.inf]  # a zero reference is never dropped


def test_reuse_is_refused_without_a_provenance_sidecar(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    pd.DataFrame({"policy": ["a"], "cost": [1.0]}).to_parquet(ex.out_dir("e3") / "results.parquet")
    with pytest.raises(ValueError, match="no provenance sidecar"):
        ex.load_stored(OmegaConf.create({"reuse": True}), "e3")


def test_gate_auc_with_one_group_is_nan() -> None:
    import numpy as np

    from kai.bench.calibrate import gate_auc

    phi, d = np.zeros((6, 2)), np.zeros((6, 3))
    assert np.isnan(gate_auc(phi, d, np.array(["b"] * 6), ("r0", "r1", "r2")))


# --- ADR 0028 fixes 10 and 12 ---------------------------------------------------------------------


def _frame_cfg() -> object:
    return OmegaConf.create(
        {
            "seed": 0, "alphas": [0.05], "delta": 0.1, "reuse": True,
            "profile": {"mission": "m"}, "dataset": {"windows": "w"},
        }
    )  # fmt: skip


def test_provenance_hashes_the_fold_tables_and_the_simulator_code(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    fold = tmp_path / "s0_r0f0"
    fold.mkdir()
    (fold / "table.npz").write_bytes(b"one")
    monkeypatch.setattr(ex, "fold_dirs", lambda cfg, level="L1": [fold])
    monkeypatch.setattr(ex, "split_id", lambda cfg, level="L1": "sp")
    first = ex.provenance(_frame_cfg())  # type: ignore[arg-type]
    assert len(first["code_hash"]) == 64
    assert first["fold_tables"].startswith("s0_r0f0:")
    assert first["folds"] == "s0_r0f0"
    (fold / "table.npz").write_bytes(b"two!")  # a rebuilt table under the same fold name
    second = ex.provenance(_frame_cfg())  # type: ignore[arg-type]
    assert second["fold_tables"] != first["fold_tables"]
    assert second["code_hash"] == first["code_hash"]
    edge = tmp_path / "slow.json"
    edge.write_text("{}")
    assert "edge_profile_sha256" in ex.provenance(_frame_cfg(), edge)  # type: ignore[arg-type]


def test_add_policies_must_name_policies_the_simulator_builds(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAI_WORK", str(tmp_path))
    monkeypatch.setattr(ex, "provenance", lambda cfg: {"p": 1})
    monkeypatch.setattr(ex, "known_policies", lambda cfg, which: {"a", "b"})
    pd.DataFrame({"policy": ["a"], "cost": [1.0]}).to_parquet(ex.out_dir("e3") / "results.parquet")
    (ex.out_dir("e3") / "results.meta.json").write_text(json.dumps({"p": 1}))
    cfg = OmegaConf.create({"reuse": True, "add_policies": ["kai_s1_v0_typo"]})
    with pytest.raises(ValueError, match=r"unknown policies.*kai_s1_v0_typo"):
        ex.stored_or_run(cfg, "e3", lambda only: pd.DataFrame())


def _cell_rows(policies: list[str]) -> pd.DataFrame:
    from dataclasses import asdict

    from kai.bench import evaluate as ev

    cell = ev.Cell("static", 1e6, 250.0, 0.0, 1, 0.05)
    return pd.DataFrame(
        [
            {"fold": f, "policy": p, "cost": 1.0, **asdict(cell)}
            for f in ("f0", "f1")
            for p in policies
        ]
    )


def test_fill_required_bytes_creates_the_column_and_fills_the_first_fold(monkeypatch) -> None:
    from pathlib import Path

    from kai.bench import evaluate as ev

    monkeypatch.setattr(ex, "fold_dirs", lambda cfg, level="L1": [Path("f0"), Path("f1")])
    monkeypatch.setattr(ex, "load_profiles", lambda **kw: ([], None))
    monkeypatch.setattr(ex, "prices_of", lambda cfg: None)
    monkeypatch.setattr(ex, "energy_of", lambda cfg: None)
    monkeypatch.setattr(ex, "window_period_ms", lambda cfg: 64.0)
    monkeypatch.setattr(ex, "work", lambda: Path("."))
    monkeypatch.setattr(ev, "load_fold", lambda *a, **kw: "ctx")
    monkeypatch.setattr(
        ev, "policies", lambda *a: {"p": lambda: ("sel", None), "q": lambda: ("sel", None)}
    )
    monkeypatch.setattr(ev, "required_bytes", lambda ctx, policy, sel: {"p": 10, "q": 20}[policy])
    frame = _cell_rows(["p", "q", "kai_s1_auto"])
    assert "required_bytes" not in frame  # a run from before ADR 0026
    out = ex.fill_required_bytes(_frame_cfg(), frame)  # type: ignore[arg-type]
    first = out[out.fold == "f0"].set_index("policy").required_bytes
    assert first["p"] == 10
    assert first["q"] == 20
    assert np.isnan(first["kai_s1_auto"])
    assert out[out.fold == "f1"].required_bytes.isna().all()  # only the first fold is read
    again = ex.fill_required_bytes(_frame_cfg(), out)  # type: ignore[arg-type]
    assert again.required_bytes.equals(out.required_bytes)  # nothing left to fill: no simulation


def test_x26_summary_works_without_required_bytes_and_with_a_policy_on_fewer_folds(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ex, "fold_dirs", lambda cfg, level="L1": [])
    frame = _cell_rows(["b4_best_single", "kai_s1_v0", "b14_edge_inference", X26[0]])
    frame["cost"] = frame.policy.map(
        {"b4_best_single": 10.0, "kai_s1_v0": 8.0, "b14_edge_inference": 20.0, X26[0]: 5.0}
    )
    frame = frame[~((frame.policy == X26[0]) & (frame.fold == "f1"))]  # committee missing on f1
    out = ex.x26_summary(_frame_cfg(), frame, 0)  # type: ignore[arg-type]
    key = f"X26/{X26[0]}/cost_reduction_vs_b4"
    assert out[key] == pytest.approx(0.5)
    assert out[f"{key}/missing_folds"] == ["f1"]
    assert out["X26/b14_edge_inference/cost_reduction_vs_b4"] == pytest.approx(-1.0)
    assert out["X26/b14_edge_inference/cost_vs_best_committee"] == pytest.approx(4.0)
    assert not any(k.startswith("X26/required_bytes/") for k in out)
    frame["required_bytes"] = 7.0
    out = ex.x26_summary(_frame_cfg(), frame, 0)  # type: ignore[arg-type]
    assert out["X26/required_bytes/kai_s1_v0"] == 7.0


X26 = ex.X26_KAI


def test_warm_fold_caches_fills_each_fold_once_in_order(monkeypatch, tmp_path) -> None:
    """ADR 0028: the grid workers must find the caches filled (fresh folds OOM-killed E3)."""
    from types import SimpleNamespace

    from kai.bench import experiments as ex

    calls: list[tuple[str, str]] = []
    table = object()
    monkeypatch.setattr(ex, "work", lambda: tmp_path)
    monkeypatch.setattr(
        ex.ev, "scored_fold",
        lambda f, k, s: calls.append(("scored", f.name)) or SimpleNamespace(table=table),
    )  # fmt: skip
    monkeypatch.setattr(ex.ev, "latent_scores", lambda f, t, w, k, s: calls.append(("z", f.name)))
    monkeypatch.setattr(
        ex.ev, "committee_scores", lambda f, t, w, k, s: calls.append(("d", f.name))
    )
    cfg = SimpleNamespace(seed=0, dataset=SimpleNamespace(windows="w.npz"))
    ex.warm_fold_caches(cfg, [tmp_path / "f0", tmp_path / "f1"])  # type: ignore[arg-type]
    assert calls == [(c, f) for f in ("f0", "f1") for c in ("scored", "z", "d")]


def test_f_raw_choice_is_shared_by_all_folds_or_raises(tmp_path) -> None:
    """ADR 0028 §7: E1 profiles fold 0's f_raw, so mixed per-fold choices must not pass silently."""
    from kai.bench.experiments import f_raw_choice

    folds = []
    for i, meta in enumerate(({"f_raw_choice": "w2v"}, {"f_raw_choice": "w2v"}, {})):
        (d := tmp_path / f"f{i}").mkdir()
        (d / "metrics.json").write_text(json.dumps(meta))
        folds.append(d)
    assert f_raw_choice(folds) == "w2v"
    assert f_raw_choice(folds[2:]) is None  # folds built before ADR 0028 record no choice
    (folds[1] / "metrics.json").write_text(json.dumps({"f_raw_choice": "wave"}))
    with pytest.raises(ValueError, match="different f_raw"):
        f_raw_choice(folds)
