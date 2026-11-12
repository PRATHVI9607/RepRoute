"""Peak-RSS probe (ADR 0028 A): fresh subprocess, baseline subtracted."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from torch import nn

from kai.certify.predictor import Predictor
from kai.profile.memory import baseline_rss_bytes, peak_rss_bytes
from kai.reps.classical import Lossless, Raw, Summary
from kai.reps.registry import save_menu
from kai.tasks import train as tr

N = 64


@pytest.fixture
def tiny_fold(tmp_path: Path) -> Path:
    (tmp_path / "f_raw.onnx").write_bytes(
        tr.export_onnx(nn.Sequential(nn.Flatten(), nn.Linear(N, 3)), N)
    )
    reps = [Raw(N), Lossless(N), Summary(N)]
    reps[1].fit(np.zeros((2, N), np.int16))
    save_menu(reps, tmp_path / "menu")
    layers = ((np.zeros((20, 4)), np.zeros(4)), (np.zeros((4, 1)), np.zeros(1)))
    Predictor(tuple(r.rep_id for r in reps), np.zeros(20), np.ones(20), (layers, layers)).save(
        tmp_path / "predictor_mlp_s0.npz"
    )
    return tmp_path


@pytest.mark.parametrize("kind", ["f_raw", "kai_classical"])
def test_peak_rss_is_a_non_negative_int_from_a_fresh_process(kind: str, tiny_fold: Path) -> None:
    rss = peak_rss_bytes(kind, tiny_fold)
    assert isinstance(rss, int)
    assert rss >= 0


def test_unknown_kind_is_rejected(tiny_fold: Path) -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        peak_rss_bytes("kai_latent", tiny_fold)


def test_f_raw_includes_onnxruntime_above_the_numpy_only_baseline(tiny_fold: Path) -> None:
    assert baseline_rss_bytes() > 0
    assert peak_rss_bytes("f_raw", tiny_fold) > 10_000_000  # onnxruntime import + session
