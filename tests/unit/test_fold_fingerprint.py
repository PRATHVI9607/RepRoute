from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import typer

from kai import cli
from kai.tasks.pipeline import FoldConfig


def _ws(x: np.ndarray) -> SimpleNamespace:
    n = len(x)
    return SimpleNamespace(
        x=x, window_id=np.arange(n), label=np.zeros(n, dtype=np.int64), dataset_id="d"
    )


def _run(monkeypatch, tmp_path, ws) -> None:
    import kai.bench.config as bc
    import kai.contracts.windows as cw
    import kai.data.splits as sp

    (tmp_path / "splits").mkdir(exist_ok=True)
    (tmp_path / "splits/d_L1_x.json").write_text("{}")
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            windows="w.npz",
            splits_dir="splits",
            folds_dir="folds",
            f_raw="env",
            bottleneck="generic",
        )
    )
    manifest = SimpleNamespace(split_id="m", folds=[SimpleNamespace(repeat=0, fold=0)])
    monkeypatch.setattr(bc, "load_config", lambda o: cfg)
    monkeypatch.setattr(bc, "work", lambda: tmp_path)
    monkeypatch.setattr(cw.WindowSet, "load", staticmethod(lambda p: ws))
    monkeypatch.setattr(sp, "load_manifest", lambda p: manifest)
    cli.folds(
        repeats=1,
        folds=1,
        f_raw="env",
        bottleneck="generic",
        device="cpu",
        force=False,
        overrides=None,
    )


def test_rebuilt_windows_file_is_refused_and_missing_stamp_is_written(
    monkeypatch, tmp_path
) -> None:
    out = tmp_path / "folds/m/s0_r0f0"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}")
    (out / "fold_config.json").write_text(
        json.dumps(
            FoldConfig("env", 30, 15, head_trials=10, bottleneck="generic", device="cpu").recipe()
        )
    )
    x = np.arange(40, dtype=np.int16).reshape(10, 4)
    _run(monkeypatch, tmp_path, _ws(x))  # no windows.json yet: trust on first use
    stored = json.loads((out / "windows.json").read_text())
    assert stored == {"fingerprint": cli._windows_fingerprint(_ws(x))}
    _run(monkeypatch, tmp_path, _ws(x.copy()))  # same content: still skipped
    with pytest.raises(typer.BadParameter):
        _run(monkeypatch, tmp_path, _ws(x + 1))
