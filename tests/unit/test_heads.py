"""Cloud heads: both families, degenerate targets, serialisation; the spectral reference model."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

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


def test_envelope_reference_model_ignores_gain_and_exports(ws: WindowSet) -> None:
    model = tr.train_envnet(ws.x, ws.label, ws.bearing_id, seed=0)
    x = ws.x[::7].astype(np.float32) / 32768.0
    logits = tr.predict(model, x)
    assert (logits.argmax(1) == ws.label[::7]).mean() > 0.6  # impact rates separate the classes
    # peak-to-floor ratios of the envelope spectrum: the window's level cannot change the decision
    assert (tr.predict(model, x * 0.2).argmax(1) == logits.argmax(1)).mean() > 0.99
    sess = ort.InferenceSession(
        tr.export_onnx(model, ws.window_n), providers=["CPUExecutionProvider"]
    )
    assert np.abs(sess.run(None, {"x": x[:, None, :]})[0] - logits).max() <= 1e-3
    assert np.isfinite(tr.predict(model, np.zeros((2, ws.window_n), np.float32))).all()


def test_teacher_front_bottleneck_repeats_the_teacher_from_sixteen_bytes(ws: WindowSet) -> None:
    from kai.reps.latent import OnnxLatent
    from kai.tasks.pipeline import latent_features

    teacher = tr.train_envnet(ws.x, ws.label, ws.bearing_id, seed=0)
    encoder = tr.train_front_bottleneck(
        teacher, ws.x, 16, epochs=30, seed=0, device=tr.resolve_device("cpu")
    )
    rep = OnnxLatent(
        "r6_bottleneck", ws.window_n, tr.export_onnx(encoder, ws.window_n), np.full(16, 1 / 127.0)
    )
    assert rep.size_model().mean_bytes == 16
    z = latent_features(rep, ws.x)  # through int8 quantisation, as on the wire
    assert z.shape == (len(ws), 16)
    assert np.abs(z).max() <= 1.0
    decision = tr.predict(teacher, ws.x.astype(np.float32) / 32768.0).argmax(1)
    head = _linear(z[::2], decision[::2], 1.0, 0)
    assert (head.predict(z[1::2]) == decision[1::2]).mean() > 0.9


@pytest.mark.xdist_group("fold")
def test_headzoo_decide_timed_and_backwards_compatibility(fold_dir, ws: WindowSet) -> None:
    import math
    from unittest.mock import MagicMock

    from kai.cloud.heads import HeadZoo

    zoo = HeadZoo(fold_dir)
    # Test one waveform rep (r0_raw) and one feature rep (r4_summary)
    for rid in ("r0_raw", "r4_summary"):
        payload = zoo.reps[rid].encode(ws.x[0])
        decision_timed, t_dec_ms, t_head_ms = zoo.decide_timed(rid, payload)
        decision_plain = zoo.decide(rid, payload)

        assert isinstance(decision_timed, int)
        assert isinstance(decision_plain, int)
        assert decision_timed == decision_plain

        assert isinstance(t_dec_ms, float)
        assert isinstance(t_head_ms, float)
        assert math.isfinite(t_dec_ms) and t_dec_ms >= 0.0
        assert math.isfinite(t_head_ms) and t_head_ms >= 0.0

    # Verify that decode_for_head and head are each invoked exactly once
    spy_rep = zoo.reps["r4_summary"]
    orig_decode = spy_rep.decode_for_head
    orig_head = zoo.head
    spy_rep.decode_for_head = MagicMock(side_effect=orig_decode)
    zoo.head = MagicMock(side_effect=orig_head)
    try:
        sample_payload = spy_rep.encode(ws.x[0])
        zoo.decide_timed("r4_summary", sample_payload)
        spy_rep.decode_for_head.assert_called_once_with(sample_payload)
        assert zoo.head.call_count == 1
    finally:
        spy_rep.decode_for_head = orig_decode
        zoo.head = orig_head


@pytest.mark.xdist_group("fold")
def test_headzoo_decide_timed_exception_propagation(fold_dir) -> None:
    import pytest
    import zstandard

    from kai.cloud.heads import HeadZoo

    zoo = HeadZoo(fold_dir)
    # Undecodable corrupt payload must propagate specific decoding exception (ZstdError on r1_lossless)
    with pytest.raises(zstandard.ZstdError):
        zoo.decide_timed("r1_lossless", b"bad_payload_too_short")
    with pytest.raises(zstandard.ZstdError):
        zoo.decide("r1_lossless", b"bad_payload_too_short")

    # Inference-head failure must propagate unchanged
    class CustomInferenceError(RuntimeError):
        pass

    orig_head = zoo.head

    def failing_head(rep_id, decoded):
        raise CustomInferenceError("head inference crashed")

    zoo.head = failing_head
    try:
        payload = zoo.reps["r0_raw"].encode(np.zeros(zoo.reps["r0_raw"].window_n, dtype=np.int16))
        with pytest.raises(CustomInferenceError, match="head inference crashed"):
            zoo.decide_timed("r0_raw", payload)
        with pytest.raises(CustomInferenceError, match="head inference crashed"):
            zoo.decide("r0_raw", payload)
    finally:
        zoo.head = orig_head


@pytest.mark.xdist_group("fold")
def test_cloud_server_handle_and_telemetry_callback(fold_dir, ws: WindowSet) -> None:
    import math

    from kai.cloud.heads import HeadZoo
    from kai.cloud.server import CloudServer
    from kai.contracts import frame as fr
    from kai.contracts.representation import REP_CODES

    telemetry_events = []

    def on_decision(wid, rep_id, decision, t_dec, t_head, cloud_ms):
        telemetry_events.append({
            "wid": wid,
            "rep_id": rep_id,
            "decision": decision,
            "t_dec": t_dec,
            "t_head": t_head,
            "cloud_ms": cloud_ms,
        })

    zoo = HeadZoo(fold_dir)
    server = CloudServer(zoo, on_decision=on_decision)
    pending: dict[int, dict[int, int]] = {}

    rep = zoo.reps["r4_summary"]
    rep_code = REP_CODES[rep.rep_id]
    chash = bytes.fromhex(rep.contract_hash())
    payload = rep.encode(ws.x[0])
    frame = fr.Frame(rep_code, chash, 42, 1000, payload)

    reply = server.handle(frame, pending)

    assert reply.status == "OK"
    assert reply.window_id == 42
    assert reply.rep_code == rep_code
    assert isinstance(reply.decision, int)
    assert isinstance(reply.cloud_ms, float)
    assert math.isfinite(reply.cloud_ms) and reply.cloud_ms >= 0.0

    assert len(telemetry_events) == 1
    ev = telemetry_events[0]
    assert ev["wid"] == 42
    assert ev["rep_id"] == "r4_summary"
    assert ev["decision"] == reply.decision
    assert isinstance(ev["t_dec"], float) and math.isfinite(ev["t_dec"]) and ev["t_dec"] >= 0.0
    assert isinstance(ev["t_head"], float) and math.isfinite(ev["t_head"]) and ev["t_head"] >= 0.0
    assert isinstance(ev["cloud_ms"], float) and math.isfinite(ev["cloud_ms"]) and ev["cloud_ms"] >= 0.0


@pytest.mark.xdist_group("fold")
def test_cloud_server_callback_failure_does_not_break_reply(fold_dir, ws: WindowSet) -> None:
    from kai.cloud.heads import HeadZoo
    from kai.cloud.server import CloudServer
    from kai.contracts import frame as fr
    from kai.contracts.representation import REP_CODES

    def crashing_callback(wid, rep_id, decision, t_dec, t_head, cloud_ms):
        raise RuntimeError("telemetry sink crashed")

    zoo = HeadZoo(fold_dir)
    server = CloudServer(zoo, on_decision=crashing_callback)
    pending: dict[int, dict[int, int]] = {}

    rep = zoo.reps["r0_raw"]
    rep_code = REP_CODES[rep.rep_id]
    chash = bytes.fromhex(rep.contract_hash())
    payload = rep.encode(ws.x[0])
    frame = fr.Frame(rep_code, chash, 99, 1000, payload)

    # Even though callback crashes, reply must still be constructed and sent successfully
    reply = server.handle(frame, pending)
    assert reply.status == "OK"
    assert reply.window_id == 99
    assert isinstance(reply.decision, int)
    assert server.frames == 1
