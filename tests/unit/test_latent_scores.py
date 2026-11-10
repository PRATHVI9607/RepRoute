"""ADR 0025 C: latent sufficiency score ĝ_z and its cost model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kai.bench import evaluate as ev
from kai.bench.calibrate import Scored
from kai.certify.fit import fit_predictor
from kai.contracts.schemas import PARTITIONS, EnergyModel, Prices, RepProfile
from kai.select.cost import Overheads
from kai.tasks.pipeline import Table

REPS = ("r0_raw", "r1_lossless", "r6_bottleneck")
N = 400


def _table(rng: np.random.Generator) -> Table:
    part = np.array([PARTITIONS.index(p) for p in ("ptrain", "calib", "test", "train")])
    partition = part[np.arange(N) % 4]
    yhat = rng.integers(0, 2, (N, 3))
    yhat[:, :2] = yhat[:, :1]  # raw and lossless agree; r6 sometimes differs
    return Table(
        REPS, np.arange(N), np.array([f"b{i % 20}" for i in range(N)]), partition,
        yhat[:, 0], yhat, np.tile([2000, 900, 16], (N, 1)), rng.normal(size=(N, 3)),
    )  # fmt: skip


@pytest.fixture
def patched(monkeypatch, tmp_path, rng):
    table = _table(rng)
    z = rng.normal(size=(N, 2))
    monkeypatch.setattr(
        ev.WindowSet,
        "load",
        staticmethod(lambda p: SimpleNamespace(x=z, window_id=table.window_id)),
    )
    monkeypatch.setattr(ev, "load_menu", lambda d: [SimpleNamespace(rep_id="r6_bottleneck")])
    monkeypatch.setattr(ev, "latent_features", lambda rep, x: x)
    monkeypatch.setattr(ev, "committee_scores", lambda *a: (None, (), None))  # ADR 0026: not here
    return table, tmp_path


def test_ghat_z_is_fitted_on_ptrain_only_and_has_one_column_per_rep(monkeypatch, patched) -> None:
    table, fold = patched
    seen: list[int] = []

    def spy(phi, d, rep_ids, kind, seed):
        seen.append(len(phi))
        assert phi.shape[1] in (table.phi.shape[1], table.phi.shape[1] + 2)  # φ or [φ, z]
        return fit_predictor(phi, d, rep_ids, kind, seed)

    monkeypatch.setattr(ev, "fit_predictor", spy)
    out = ev.latent_scores(fold, table, fold / "w.npz", "logistic", 0)
    assert out is not None
    g, use_z = out
    assert g.shape == (N, len(REPS))
    assert use_z.shape == (len(REPS),)
    n_ptrain = int(table.of("ptrain").sum())
    assert seen[0] == n_ptrain  # the final fit
    assert max(seen) == n_ptrain  # and every CV fit saw a subset of ptrain, never calib / test
    n_calls = len(seen)
    ev.latent_scores(fold, table, fold / "w.npz", "logistic", 0)  # second call: cached, no refit
    assert len(seen) == n_calls


def test_no_latent_without_r6_or_windows(patched) -> None:
    table, fold = patched
    assert ev.latent_scores(fold, table, None) is None
    no_r6 = SimpleNamespace(rep_ids=("r0_raw", "r1_lossless"))
    assert ev.latent_scores(fold, no_r6, fold / "w.npz") is None  # type: ignore[arg-type]


def test_window_ids_must_match(monkeypatch, patched) -> None:
    table, fold = patched
    monkeypatch.setattr(
        ev.WindowSet,
        "load",
        staticmethod(
            lambda p: SimpleNamespace(x=np.zeros((N, 2)), window_id=table.window_id[::-1])
        ),
    )
    with pytest.raises(ValueError, match="window ids"):
        ev.latent_scores(fold, table, fold / "w.npz")


def test_model_z_charges_the_r6_encode_in_the_overhead(monkeypatch, patched) -> None:
    table, fold = patched
    profiles = [
        RepProfile(
            rep_id=r, size_mean_bytes=s, size_q95_bytes=s, t_enc_ms=e, t_enc_mad_ms=0.0,
            t_enc_q95_ms=e, t_dec_ms=0.1, t_head_ms=0.1, load_exponent=1.0,
        )
        for r, s, e in (("r0_raw", 2000, 0.01), ("r1_lossless", 900, 0.3), (REPS[2], 16, 0.4))
    ]  # fmt: skip
    pred = fit_predictor(
        table.phi[table.of("ptrain")], table.d[table.of("ptrain")], REPS, "logistic", 0
    )
    monkeypatch.setattr(ev, "scored_fold", lambda *a: Scored(table, pred, pred.predict(table.phi)))
    prices = Prices(name="t", story="t", p_byte=1.0, p_j=0.0, p_cpu_ms=5.0, p_late_ms=1000.0)
    ctx = ev.load_fold(
        fold, profiles, Overheads(0.05, 0.02, 0.01), prices,
        EnergyModel(p_cpu_w=2.0, e_bit_j=1e-7), (0.1,), 0.05, 64.0, "logistic",
        windows_path=fold / "w.npz",
    )  # fmt: skip
    assert ctx.ghat_z is not None
    assert ctx.model_z is not None
    assert set(ctx.lam_z) == {0.1}
    assert ctx.model_z.overheads.t_probe_ms == pytest.approx(ctx.model.overheads.t_probe_ms + 0.4)
    assert ctx.model_z.profiles.t_enc_ms[2] == 0.0
    assert ctx.model.profiles.t_enc_ms[2] == 0.4  # the plain model is untouched
    assert ctx.workload(10, scores="latent").ghat.shape == (10, 3)


def test_choose_latent_picks_z_where_it_carries_the_signal(monkeypatch, rng) -> None:
    n = 600
    groups = np.array([f"u{i % 30}" for i in range(n)])
    phi = rng.normal(size=(n, 2))
    z = rng.normal(size=(n, 2))
    d = np.zeros((n, 3), bool)
    d[:, 1] = z[:, 0] + 0.1 * rng.normal(size=n) > 0  # only z tells
    d[:, 2] = phi[:, 0] + 0.1 * rng.normal(size=n) > 0  # only phi tells
    ptrain = np.arange(n) < 500
    rows: list[int] = []

    def spy(x, dd, rep_ids, kind, seed):
        rows.append(len(x))
        return fit_predictor(x, dd, rep_ids, kind, seed)

    monkeypatch.setattr(ev, "fit_predictor", spy)
    use = ev.choose_latent(phi, z, d, groups, ptrain, REPS, "logistic", 0)
    assert use.tolist() == [False, True, False]  # raw is never z; rep 2 has no gain from z
    assert max(rows) < 500  # ptrain rows only, and each fit is on a part of them


def test_ghat_sel_mixes_columns_and_zsel_policies_use_the_sel_workload_and_model(
    monkeypatch, patched
) -> None:
    table, fold = patched
    prices = Prices(name="t", story="t", p_byte=1.0, p_j=0.0, p_cpu_ms=5.0, p_late_ms=1000.0)
    profiles = [
        RepProfile(
            rep_id=r, size_mean_bytes=s, size_q95_bytes=s, t_enc_ms=e, t_enc_mad_ms=0.0,
            t_enc_q95_ms=e, t_dec_ms=0.1, t_head_ms=0.1, load_exponent=1.0,
        )
        for r, s, e in (("r0_raw", 2000, 0.01), ("r1_lossless", 900, 0.3), (REPS[2], 16, 0.4))
    ]  # fmt: skip
    pred = fit_predictor(
        table.phi[table.of("ptrain")], table.d[table.of("ptrain")], REPS, "logistic", 0
    )
    monkeypatch.setattr(ev, "scored_fold", lambda *a: Scored(table, pred, pred.predict(table.phi)))
    monkeypatch.setattr(ev, "choose_latent", lambda *a, **k: np.array([False, False, True]))
    ctx = ev.load_fold(
        fold, profiles, Overheads(0.05, 0.02, 0.01), prices,
        EnergyModel(p_cpu_w=2.0, e_bit_j=1e-7), (0.1,), 0.05, 64.0, "logistic",
        windows_path=fold / "w.npz",
    )  # fmt: skip
    assert ctx.use_z.tolist() == [False, False, True]
    assert np.array_equal(ctx.ghat_sel[:, :2], ctx.scored.ghat[:, :2])
    assert np.array_equal(ctx.ghat_sel[:, 2], ctx.ghat_z[:, 2])
    assert ctx.model_sel is ctx.model_z  # some rep uses z: the r6 encode is paid
    assert set(ctx.lam_sel) == {0.1}

    seen: list[tuple] = []
    monkeypatch.setattr(ev, "simulate", lambda sc, w, sel, cert, truth: seen.append((w, truth)))
    monkeypatch.setattr(ev, "metrics", lambda *a: {})
    cell = ev.Cell("static", 1e6, 100.0, 0.0, 1, 0.1)
    ev.run_cell(
        ctx,
        cell,
        "fidelity_first",
        "main",
        10,
        0,
        only=("kai_s1_v0_zsel", "kai_s1_v0_ab_zsel", "kai_s1_v0"),
    )
    sel_work = ctx.workload(10, scores="sel")
    assert len(seen) == 3
    zsel = [v for v in seen if v[1] is ctx.model_sel]
    assert len(zsel) == 2
    assert all(np.array_equal(w.ghat, sel_work.ghat) for w, _ in zsel)
    assert any(v[1] is ctx.model for v in seen)

    for f in fold.glob("use_z_*.npy"):
        f.unlink()  # forget the cached choice
    monkeypatch.setattr(ev, "choose_latent", lambda *a, **k: np.zeros(3, bool))
    ctx0 = ev.load_fold(
        fold, profiles, Overheads(0.05, 0.02, 0.01), prices,
        EnergyModel(p_cpu_w=2.0, e_bit_j=1e-7), (0.1,), 0.05, 64.0, "logistic",
        windows_path=fold / "w2.npz",
    )  # fmt: skip
    assert ctx0.model_sel is ctx0.model  # nobody uses z: no r6 encode charged
