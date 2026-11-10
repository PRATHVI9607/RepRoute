"""scripts/report.py helpers (the script is not a package: load it by path)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

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
