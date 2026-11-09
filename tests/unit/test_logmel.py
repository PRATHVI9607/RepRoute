"""R10 log-mel (ADR 0018): opt-in, deterministic, fixed size, reloadable, ONNX head round-trips."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kai.cloud.heads import OnnxHead, load_head, save_head
from kai.reps.logmel import LogMel
from kai.reps.registry import classical_menu, load_menu, save_menu
from kai.tasks import train as tr


def test_logmel_is_opt_in_deterministic_fixed_size_and_reloads(
    rng: np.random.Generator, tmp_path
) -> None:
    n = 4096
    assert not any(isinstance(r, LogMel) for r in classical_menu(n))
    menu = classical_menu(n, logmel_hz=16000.0)
    reps = [r for r in menu if isinstance(r, LogMel)]
    assert [r.rep_id for r in reps] == ["r10a_logmel", "r10b_logmel"]
    x = (rng.standard_normal(n) * 3000).astype(np.int16)
    for rep in reps:
        p = rep.encode(x)
        assert p == rep.encode(x.copy())
        assert len(p) == rep.n_mels * rep.n_frames + 2 == rep.size_model().q95_bytes
        dec = rep.decode_for_head(p)
        assert dec.shape == (1 + rep.n_mels * rep.n_frames,)
        assert dec[1:].min() >= 0.0
        assert dec[1:].max() == 1.0  # the peak maps to the top level
    with pytest.raises(ValueError, match="payload"):
        reps[0].decode_for_head(p)  # the other grid's payload
    with pytest.raises(ValueError, match="audio"):
        classical_menu(n, logmel_hz=50.0)  # HAR: most mel bands would be empty
    other = LogMel("r10a_logmel", n, 8000.0, 16, 16)
    assert other.contract_hash() != reps[0].contract_hash()
    for r in menu:
        r.set_size(np.array([len(r.encode(x))]))
    save_menu(menu, tmp_path)
    assert [r.contract_hash() for r in load_menu(tmp_path)] == [r.contract_hash() for r in menu]


def test_logmel_cnn_head_learns_and_round_trips_through_onnx(
    rng: np.random.Generator, tmp_path
) -> None:
    rep = LogMel("r10b_logmel", 4096, 16000.0, 12, 12)
    t = np.arange(4096) / 16000.0
    y = np.arange(120) % 3
    x = np.stack(
        [
            3000 * np.sin(2 * np.pi * (300 + 1500 * k) * t) + rng.standard_normal(t.size) * 50
            for k in y
        ]
    ).astype(np.int16)
    feats = np.stack([rep.decode_for_head(rep.encode(w)) for w in x])
    net = tr.train_logmel_head(feats, y, 12, 12, seed=0, device=tr.resolve_device("cpu"), epochs=8)
    assert not torch.are_deterministic_algorithms_enabled()  # training restores the global flag
    head = OnnxHead(tr.export_onnx(net, feats.shape[1]))
    assert (head.predict(feats) == y).mean() > 0.9
    save_head(head, tmp_path / "r10b_logmel")
    assert np.array_equal(load_head(tmp_path / "r10b_logmel").predict(feats), head.predict(feats))
