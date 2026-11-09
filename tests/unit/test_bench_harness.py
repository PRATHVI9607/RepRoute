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
