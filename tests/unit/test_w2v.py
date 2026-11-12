"""train_w2v on a tiny stub backbone (no download): training, grouped early stop, ONNX parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchaudio
from torch import nn

from kai.tasks import train as tr


class _Stub(nn.Module):
    hidden_size = 8

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Conv1d(1, 4, 400, stride=400)
        self.proj = nn.Linear(4, 8)

    def extract_features(self, w):
        h = self.feature_extractor(w[:, None]).transpose(1, 2)
        return [self.proj(h)], None


class _Bundle:
    def get_model(self) -> nn.Module:
        return _Stub()


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(torchaudio.pipelines, "WAV2VEC2_BASE", _Bundle())


def _toy(n=48, w=1600):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, n)
    t = np.arange(w)
    x = np.stack([(8000 * np.sin(t * (0.05 + 0.1 * c)) + rng.normal(0, 500, w)) for c in y])
    return x.astype(np.int16), y, np.repeat(np.arange(n // 4), 4)


def test_holdout_is_grouped_and_seeded():
    g = np.repeat(np.arange(10), 3)
    m1 = tr.grouped_holdout(g, np.random.default_rng(1))
    m2 = tr.grouped_holdout(g, np.random.default_rng(1))
    assert (m1 == m2).all()
    assert 0 < m1.sum() < len(g)
    assert all(m1[g == i].all() or not m1[g == i].any() for i in range(10))


def test_train_export_parity(stub):
    x, y, g = _toy()
    model = tr.train_w2v(x, y, g, np.random.default_rng(0), torch.device("cpu"), epochs=2, batch=8)
    assert not model.training
    assert next(model.parameters()).device.type == "cpu"
    assert not any(p.requires_grad for p in model.backbone.feature_extractor.parameters())
    blob = tr.export_w2v_onnx(model, x.shape[1])
    assert len(blob) > 0
    med, mad = tr.time_gpu_ms(model, x.shape[1], torch.device("cpu"), warmup=1, repeats=3)
    assert med > 0
    assert mad >= 0


def test_parity_failure_raises(stub, monkeypatch):
    x, y, g = _toy(16)
    model = tr.train_w2v(x, y, g, np.random.default_rng(0), torch.device("cpu"), epochs=1, batch=8)
    real = tr.export_onnx

    def broken(m, n):  # an export whose weights differ from the torch model
        with torch.no_grad():
            m.head.bias.add_(1.0)
        blob = real(m, n)
        with torch.no_grad():
            m.head.bias.sub_(1.0)
        return blob

    monkeypatch.setattr(tr, "export_onnx", broken)
    with pytest.raises(RuntimeError, match="parity"):
        tr.export_w2v_onnx(model, x.shape[1])
