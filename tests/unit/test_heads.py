"""Cloud heads: both families, degenerate targets, serialisation; the spectral reference model."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from kai.cloud.heads import LinearHead, TreeHead, load_head, save_head
from kai.contracts.windows import WindowSet
from kai.tasks import train as tr
from kai.tasks.heads import _linear, _tree, tune_and_fit


def blobs(rng: np.random.Generator, n: int = 600, classes: tuple[int, ...] = (0, 1, 2)):
    y = rng.choice(classes, n)
    x = rng.normal(size=(n, 5)) + np.eye(3)[y] @ rng.normal(0, 3, size=(3, 5))
    return x, y, np.arange(n) % 6


def test_both_families_learn_and_round_trip(rng: np.random.Generator, tmp_path) -> None:
    x, y, _ = blobs(rng)
    tree_params = {"n_estimators": 40, "num_leaves": 8}
    for head in (_linear(x, y, 1.0, 0), _tree(x, y, tree_params, 0, 2)):
        assert (head.predict(x) == y).mean() > 0.9
        assert head.predict(x[0]).shape == (1,)  # a single decoded payload works too
        save_head(head, tmp_path / "h")
        loaded = load_head(tmp_path / "h")
        assert type(loaded) is type(head)
        assert np.array_equal(loaded.predict(x), head.predict(x))
        for f in tmp_path.iterdir():
            f.unlink()


def test_search_gives_every_representation_the_same_budget(rng: np.random.Generator) -> None:
    x, y, groups = blobs(rng)
    head = tune_and_fit(x, y, groups, n_trials=4, seed=0, n_jobs=2)
    assert isinstance(head, LinearHead | TreeHead)
    assert (head.predict(x) == y).mean() > 0.9
    again = tune_and_fit(x, y, groups, n_trials=4, seed=0, n_jobs=2)
    assert np.array_equal(again.predict(x), head.predict(x))  # same seed → same head


def test_degenerate_targets_do_not_break_the_head(rng: np.random.Generator) -> None:
    x, _, groups = blobs(rng)
    constant = tune_and_fit(x, np.full(len(x), 2), groups, n_trials=2, seed=0)
    assert set(constant.predict(x)) == {2}  # the reference model said one thing everywhere
    x2, y2, _ = blobs(rng, classes=(0, 2))
    binary = _linear(x2, y2, 1.0, 0)
    assert set(binary.predict(x2)) <= {0, 2}  # the absent class is never predicted
    assert (binary.predict(x2) == y2).mean() > 0.9
    tree = _tree(x2, y2, {"n_estimators": 30}, 0, 2)
    assert set(tree.predict(x2)) <= {0, 2}  # booster columns are mapped back to class ids


def test_spectral_reference_model_ignores_gain_and_exports(ws: WindowSet) -> None:
    model = tr.train_specnet(ws.x, ws.label, ws.bearing_id, seed=0)
    x = ws.x[::7].astype(np.float32) / 32768.0
    logits = tr.predict(model, x)
    assert (logits.argmax(1) == ws.label[::7]).mean() > 0.8
    # the features are a spectral *shape*: rescaling the window cannot change the decision
    assert np.array_equal(tr.predict(model, x * 0.2).argmax(1), logits.argmax(1))
    onnx_bytes = tr.export_onnx(model, ws.window_n)
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    assert np.abs(sess.run(None, {"x": x[:, None, :]})[0] - logits).max() <= 1e-4
    assert sum(p.numel() for p in model.parameters()) < 100  # low capacity by design
