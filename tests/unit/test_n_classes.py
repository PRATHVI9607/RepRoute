"""Trainers size their output from the labels: the transfer datasets have 5, 6 and 10 classes."""

from __future__ import annotations

import numpy as np

from kai.tasks import train as tr


def test_trainers_emit_one_logit_per_class(rng: np.random.Generator) -> None:
    n, window_n, k = 60, 256, 5
    y = np.arange(n) % k
    x = (rng.standard_normal((n, window_n)) * 3000 * (1 + y[:, None])).astype(np.int16)
    groups = np.arange(n) % 6
    xf = x.astype(np.float32) / 32768.0
    spec = tr.train_specnet(x, y, groups, seed=0)
    assert tr.predict(spec, xf).shape == (n, k)
    raw = tr.train_rawnet(x, y, epochs=1, seed=0, device=tr.resolve_device("cpu"))
    assert tr.predict(raw, xf).shape == (n, k)


def test_linear_head_covers_every_class_it_was_trained_on(rng: np.random.Generator) -> None:
    from kai.tasks.heads import _linear

    y = np.arange(50) % 7
    x = rng.standard_normal((50, 4)) + y[:, None]
    head = _linear(x, y, c=1.0, seed=0)
    assert set(head.predict(x).tolist()) <= set(range(7))
    assert head.weight.shape[0] == 7
