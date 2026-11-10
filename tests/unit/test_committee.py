"""ADR 0026: head probabilities, committee score ĝ_D, decision payloads, b14, memory gate."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pytest

from kai.bench import evaluate as ev
from kai.bench.calibrate import Scored
from kai.certify.fit import fit_predictor
from kai.cloud.heads import LinearHead, OnnxHead, TreeHead
from kai.contracts.schemas import PARTITIONS, EnergyModel, Prices, RepProfile
from kai.select.cost import Overheads
from kai.tasks.pipeline import Table

REPS = ("r0_raw", "r1_lossless", "r3_spectral", "r4_summary")
COMMITTEE = (2, 3)
N, F, C = 400, 4, 3


def _linear(rng: np.random.Generator) -> LinearHead:
    return LinearHead(np.zeros(F), np.ones(F), rng.normal(size=(C, F)), rng.normal(size=C))


def test_linear_proba_matches_predict(rng: np.random.Generator) -> None:
    x, head = rng.normal(size=(50, F)), _linear(rng)
    p = head.proba(x)
    assert p.shape == (50, C)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.array_equal(p.argmax(axis=1), head.predict(x))


@pytest.mark.parametrize("classes", [[0, 1], [0, 1, 2], [0, 2]])
def test_tree_proba_matches_predict_and_aligns_columns(
    rng: np.random.Generator, classes: list[int]
) -> None:
    x = rng.normal(size=(300, F))
    col = np.digitize(x[:, 0], np.quantile(x[:, 0], np.linspace(0, 1, len(classes) + 1)[1:-1]))
    params = {"verbose": -1, "num_threads": 1, "min_data_in_leaf": 5}
    if len(classes) > 2:
        params |= {"objective": "multiclass", "num_class": len(classes)}
    else:
        params["objective"] = "binary"
    booster = lgb.train(params, lgb.Dataset(x, col), num_boost_round=10)
    head = TreeHead(booster, np.array(classes))
    p = head.proba(x, n_classes=C)
    assert p.shape == (300, C)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.array_equal(p.argmax(axis=1), head.predict(x))
    absent = [c for c in range(C) if c not in classes]
    assert (p[:, absent] == 0).all()


def test_onnx_proba_matches_predict(rng: np.random.Generator) -> None:
    torch = pytest.importorskip("torch")
    from kai.tasks import train as tr

    feats = rng.normal(size=(60, 145)).astype(np.float32)
    y = rng.integers(0, C, 60)
    net = tr.train_logmel_head(feats, y, 12, 12, seed=0, device=tr.resolve_device("cpu"), epochs=2)
    head = OnnxHead(tr.export_onnx(net, feats.shape[1]))
    p = head.proba(feats)
    assert torch is not None
    assert p.shape == (60, C)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(p.argmax(axis=1), head.predict(feats))


@pytest.fixture
def fold(monkeypatch, tmp_path, rng):
    x = rng.normal(size=(N, F))
    heads = {r: _linear(rng) for r in ("r3_spectral", "r4_summary")}
    yhat = np.zeros((N, len(REPS)), int)
    for k in COMMITTEE:
        yhat[:, k] = heads[REPS[k]].predict(x)
    yhat[:, 0] = yhat[:, 2]  # f_raw's decision; the lossless copy agrees with it
    yhat[:, 1] = yhat[:, 0]
    flip = rng.random(N) < 0.15
    yhat[flip, 3] = (
        yhat[flip, 3] + 1
    ) % C  # r4 sometimes disagrees with f_raw, so d is non-trivial
    heads["r4_summary"] = LinearHead(
        np.zeros(F), np.ones(F), np.zeros((C, F)), np.eye(C)[0]
    )  # placeholder, replaced below once yhat is final
    part = np.array([PARTITIONS.index(p) for p in ("ptrain", "calib", "test", "train")])
    table = Table(
        REPS, np.arange(N), np.array([f"b{i % 20}" for i in range(N)]), part[np.arange(N) % 4],
        yhat[:, 0], yhat, np.tile([2000, 900, 40, 20], (N, 1)), rng.normal(size=(N, 3)),
    )  # fmt: skip
    # r4's stored decision is its head's own: rebuild yhat[:, 3] from a real head
    heads["r4_summary"] = _linear(rng)
    table.yhat[:, 3] = heads["r4_summary"].predict(x)
    reps = [
        SimpleNamespace(
            rep_id=r, head_input="features", encode=lambda w: w, decode_for_head=lambda b: b
        )
        for r in ("r3_spectral", "r4_summary")
    ]
    monkeypatch.setattr(
        ev.WindowSet,
        "load",
        staticmethod(lambda p: SimpleNamespace(x=x, window_id=table.window_id)),
    )
    monkeypatch.setattr(
        ev, "load_menu", lambda d: [SimpleNamespace(rep_id="r0_raw", head_input="waveform"), *reps]
    )
    monkeypatch.setattr(ev, "load_head", lambda stem: heads[stem.name])
    return table, tmp_path, heads, x


def test_committee_features_shape_and_votes(fold) -> None:
    table, path, heads, x = fold
    feats, idx = ev.committee_features(path, table, path / "w.npz")
    assert idx == COMMITTEE
    assert feats.shape == (N, len(COMMITTEE) * C + C)
    p3, p4 = feats[:, :C], feats[:, C : 2 * C]
    assert np.allclose(p3, heads["r3_spectral"].proba(x), atol=1e-6)
    votes = feats[:, 2 * C :]
    assert (votes.sum(axis=1) == len(COMMITTEE)).all()
    expect = np.eye(C)[p3.argmax(1)] + np.eye(C)[p4.argmax(1)]
    assert np.array_equal(votes, expect)


def test_agreement_guard_raises_with_rates(fold) -> None:
    table, path, _, _ = fold
    table.yhat[: N // 10, 3] = (table.yhat[: N // 10, 3] + 1) % C  # 10 % of r4 differ
    with pytest.raises(ValueError, match=r"r4_summary.*0\.9"):
        ev.committee_features(path, table, path / "w.npz")


def test_ghat_d_is_fitted_on_ptrain_only_and_cached(monkeypatch, fold) -> None:
    table, path, _, _ = fold
    seen: list[int] = []

    def spy(phi, d, rep_ids, kind, seed):
        seen.append(len(phi))
        return fit_predictor(phi, d, rep_ids, kind, seed)

    monkeypatch.setattr(ev, "fit_predictor", spy)
    (path / "table.npz").write_bytes(b"x")  # the cache key reads it
    g, idx, d_dec = ev.committee_scores(path, table, path / "w.npz", "logistic", 0)
    assert d_dec is not None
    assert g is not None
    assert g.shape == (N, len(REPS))
    assert idx == COMMITTEE
    assert seen == [int(table.of("ptrain").sum())]
    g2, _, _ = ev.committee_scores(path, table, path / "w.npz", "logistic", 0)
    assert seen == [int(table.of("ptrain").sum())]  # cached: no refit
    assert np.array_equal(g, g2)
    assert ev.committee_scores(path, table, None) == (None, (), None)


def _profiles() -> list[RepProfile]:
    return [
        RepProfile(
            rep_id=r, size_mean_bytes=s, size_q95_bytes=s, t_enc_ms=e, t_enc_mad_ms=0.0,
            t_enc_q95_ms=e, t_dec_ms=0.1, t_head_ms=h, load_exponent=1.0,
        )
        for r, s, e, h in zip(
            REPS, (2000, 900, 40, 20), (0.01, 0.3, 0.5, 0.2), (0.7, 0.1, 0.2, 0.3), strict=True
        )
    ]  # fmt: skip


@pytest.fixture
def ctx(monkeypatch, fold):
    table, path, _, _ = fold
    (path / "table.npz").write_bytes(b"x")
    pred = fit_predictor(
        table.phi[table.of("ptrain")], table.d[table.of("ptrain")], REPS, "logistic", 0
    )
    monkeypatch.setattr(ev, "scored_fold", lambda *a: Scored(table, pred, pred.predict(table.phi)))
    prices = Prices(name="t", story="t", p_byte=1.0, p_j=0.0, p_cpu_ms=5.0, p_late_ms=1000.0)
    return ev.load_fold(
        path, _profiles(), Overheads(0.05, 0.02, 0.01), prices,
        EnergyModel(p_cpu_w=2.0, e_bit_j=1e-7), (0.1,), 0.05, 64.0, "logistic",
        windows_path=path / "w.npz",
    )  # fmt: skip


def test_model_d_and_model_dec_accounting(ctx) -> None:
    p, c = ctx.model.profiles, list(COMMITTEE)
    assert ctx.committee == COMMITTEE
    assert set(ctx.lam_d) == {0.1}
    probe = ctx.model.overheads.t_probe_ms + (p.t_enc_ms + p.t_head_ms)[c].sum()
    assert ctx.model_d.overheads.t_probe_ms == pytest.approx(probe)
    assert (ctx.model_d.profiles.t_enc_ms[c] == 0).all()
    assert np.array_equal(ctx.model_d.profiles.size_bytes, p.size_bytes)
    m = ctx.model_dec.profiles
    assert (m.size_bytes[c] == 1).all()
    assert (m.t_dec_ms[c] == 0).all()
    assert (m.t_head_ms[c] == 0).all()
    assert (m.t_enc_ms[c] == 0).all()
    assert m.size_bytes[1] == p.size_bytes[1]  # non-members untouched
    assert (ctx.model.profiles.t_enc_ms[c] > 0).all()  # the plain model is untouched
    assert (ctx.workload(10, scores="dec").payload_bytes[:, c] == 1).all()
    assert (ctx.workload(10, scores="committee").payload_bytes[:, c] > 1).all()
    assert ctx.workload(10, scores="committee").ghat.shape == (10, len(REPS))


def test_b14_sends_one_byte_with_zero_disagreement(ctx) -> None:
    e = ctx.model_edge.profiles
    assert e.size_bytes[0] == 1
    assert e.t_enc_ms[0] == pytest.approx(0.7)  # raw's measured head time
    assert e.t_dec_ms[0] == 0
    assert e.t_head_ms[0] == 0
    w = ctx.workload(10, scores="edge")
    assert (w.payload_bytes[:, 0] == 1).all()
    assert not w.d[:, 0].any()
    cell = ev.Cell("static", 1e6, 250.0, 0.0, 1, 0.1)
    rows = ev.run_cell(ctx, cell, "fidelity_first", "main", 50, 0, only=(ev.EDGE_POLICY,))
    assert len(rows) == 1
    assert rows[0]["bytes"] == pytest.approx(1 + 28)  # decision + frame overhead
    assert rows[0]["risk_sent"] == 0.0


def test_edge_slowdown_scales_the_head_times_run_on_the_edge(ctx) -> None:
    """ADR 0027: f_raw (b14) and the committee heads run on the edge, so they slow down too."""
    p, c, k = ctx.model.profiles, list(COMMITTEE), 10.0
    model_d, _ = ev.committee_models(ctx.model, ctx.committee, k)
    probe = ctx.model.overheads.t_probe_ms + (p.t_enc_ms + k * p.t_head_ms)[c].sum()
    assert model_d.overheads.t_probe_ms == pytest.approx(probe)
    assert ev.edge_model(ctx.model, k).profiles.t_enc_ms[0] == pytest.approx(7.0)
    assert ctx.model_edge.profiles.t_enc_ms[0] == pytest.approx(0.7)  # default: no slowdown


def test_committee_policies_run(ctx) -> None:
    cell = ev.Cell("static", 1e6, 250.0, 0.0, 1, 0.1)
    names = (*ev.COMMITTEE_POLICIES, ev.DEC_POLICY)
    rows = ev.run_cell(ctx, cell, "fidelity_first", "main", 50, 0, only=names)
    assert {r["policy"] for r in rows} == set(names)


def test_required_bytes_uses_the_right_files(ctx, tmp_path) -> None:
    fold = tmp_path / "f"
    (fold / "menu").mkdir(parents=True)
    (fold / "heads").mkdir()
    (fold / "f_raw.onnx").write_bytes(b"x" * 900)
    (fold / "menu" / "reps.json").write_text(json.dumps([{"rep_id": r} for r in REPS]))
    (fold / "menu" / "r3_spectral.onnx").write_bytes(b"x" * 100)
    (fold / "menu" / "r3_spectral.npy").write_bytes(b"x" * 7)
    (fold / "menu" / "r10_other.onnx").write_bytes(b"x" * 999)  # not a menu rep: never counted
    (fold / "heads" / "r3_spectral.npz").write_bytes(b"x" * 20)
    (fold / "heads" / "r4_summary.txt").write_bytes(b"x" * 30)
    (fold / "heads" / "r4_summary.classes.npy").write_bytes(b"x" * 5)
    (fold / "predictor_logistic_ab.npz").write_bytes(b"x" * 50)
    (fold / "predictor_d_logistic_ab.npz").write_bytes(b"x" * 10)
    (fold / "predictor_z_logistic_ab.npz").write_bytes(b"x" * 777)  # the latent one: never counted
    sz = ev.model_bytes(fold, "logistic")
    assert sz["f_raw"] == 900
    assert sz["enc/r3_spectral"] == 107
    assert sz["enc/r4_summary"] == 0  # classical codec: no stored file
    assert sz["head/r3_spectral"] == 20
    assert sz["head/r4_summary"] == 35
    assert (sz["predictor"], sz["predictor_d"]) == (50, 10)
    ctx = replace(ctx, sizes=sz)
    cell = ev.Cell("static", 1e6, 250.0, 0.0, 1, 0.1)
    names = (ev.EDGE_POLICY, "kai_s1_v0_late_committee", "kai_s1_v0", "b3_features", "oracle")
    rows = ev.run_cell(ctx, cell, "fidelity_first", "main", 20, 0, names)
    need = {r["policy"]: r["required_bytes"] for r in rows}
    assert need[ev.EDGE_POLICY] == 900  # f_raw only
    assert need["kai_s1_v0_late_committee"] == 20 + 35 + 107 + 0 + 10  # members' enc + head + ĝ_D
    assert need["kai_s1_v0"] == 107 + 50  # every menu encoder + the plain predictor
    assert need["b3_features"] == 107  # its own encoder
    assert need["oracle"] == 0


def test_dec_loss_is_what_the_edge_sends(ctx, fold) -> None:
    table = ctx.scored.table
    assert ctx.d_dec is not None
    # a member's decision is its own argmax, so it is compared with f_raw's decision directly
    for k in COMMITTEE:
        assert np.array_equal(ctx.d_dec[:, k], table.yhat[:, k] != table.yhat[:, 0])
    assert np.array_equal(ctx.d_dec[:, [0, 1]], table.d[:, [0, 1]])
    assert set(ctx.lam_dec) == {0.1}
    assert np.array_equal(
        ctx.workload(20, scores="dec").d, ctx.d_dec[ev.spread(np.flatnonzero(table.of("test")), 20)]
    )


def test_certified_on_time_rate_is_reported(ctx) -> None:
    cell = ev.Cell("static", 1e6, 250.0, 0.0, 1, 0.1)
    (row,) = ev.run_cell(ctx, cell, "fidelity_first", "main", 30, 0, ("kai_s1_v0",))
    assert 0.0 <= row["rate_certified_on_time"] <= row["rate_certified"] + 1e-12
