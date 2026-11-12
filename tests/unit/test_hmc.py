"""HMC loader on tiny EDF/EDF+ files (no network), plus the ADR 0028 §1 fixes 15-17 and 19."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy.signal import resample_poly
from test_sleepedf import _edf, _hypnogram

from kai.data import hmc, sleepedf
from kai.data.pu import quantise

m = hmc.MARGIN_EPOCHS


def _eeg(digital: np.ndarray, n_records: int, spr: int = 256) -> dict:
    return {
        "label": "EEG C4-M1",
        "unit": "uV",
        "spr": spr,
        "pmin": -2048,
        "pmax": 2047,
        "dmin": -2048,
        "dmax": 2047,
        "data": digital[: n_records * spr],
    }


def _night(tmp_path, sn: str, n_epochs: int, stages, rng) -> np.ndarray:
    (tmp_path / "recordings").mkdir(exist_ok=True)
    n_rec = n_epochs * 30
    d = rng.integers(-100, 100, size=n_rec * 256).astype(np.int16)
    (tmp_path / f"recordings/{sn}.edf").write_bytes(_edf([_eeg(d, n_rec)], n_rec, 1.0))
    (tmp_path / f"recordings/{sn}_sleepscoring.edf").write_bytes(_hypnogram(stages))
    return d.astype(float)  # physical = digital here (gain 1, offset 0)


def _sums(tmp_path, names) -> None:
    def h(n):
        f = tmp_path / "recordings" / n
        return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "0" * 64

    lines = [f"{h(n)} recordings/{n}" for n in names]
    (tmp_path / sleepedf.SUMS).write_text("\n".join(lines) + "\n")


def test_hmc_build_resamples_whole_night_and_skips_lights(tmp_path, rng) -> None:
    stages = [(0, 30 * 70, "Sleep stage W"), (30 * 70, 30 * 3, "Sleep stage N2")]
    stages += [(30 * 73, 30, "Sleep stage R"), (30 * 74, 30, "Sleep stage N3")]
    stages += [(30 * 75, 30 * 80, "Sleep stage W")]
    stages = [(0, 0, "Lights off"), *stages, (12345, 0, "Lights on")]
    phys = _night(tmp_path, "SN001", 160, stages, rng)
    _sums(tmp_path, ["SN001.edf", "SN001_sleepscoring.edf", "SN002.edf", "../x/SN003.edf"])
    ws, report = hmc.build_windowset(tmp_path)
    assert report["n_catalogued"] == 1
    assert report["skipped_nights"]["missing"] == []  # SN002 has no hypnogram: not catalogued
    assert ws.window_n == 3072
    assert ws.sample_rate_hz == pytest.approx(102.4)
    assert set(ws.bearing_id.tolist()) == {"SN001"}
    assert ws.dataset_id == "hmc-v1"
    first, last = 70, 74
    assert len(ws) == (last + m) - (first - m) + 1
    assert ws.label[m : m + 5].tolist() == [2, 2, 2, 4, 3]
    # epoch 40 is the whole-night resample cut at the epoch edge (not a per-epoch resample)
    whole = quantise(resample_poly(phys, 2, 5), hmc.FULL_SCALE_UV)[0]
    assert np.array_equal(ws.x[40 - (first - m)], whole[40 * 3072 : 41 * 3072])
    per_epoch = quantise(resample_poly(phys[40 * 7680 : 41 * 7680], 2, 5), 250.0)[0]
    assert not np.array_equal(ws.x[40 - (first - m)], per_epoch)


def test_hmc_unknown_annotation_is_an_error_and_txt_fallback_works() -> None:
    with pytest.raises(ValueError, match="unknown annotation"):
        hmc.stage_epochs([(0, 30, "Sleep stage ?")])
    with pytest.raises(ValueError, match="30-s grid"):
        hmc.stage_epochs([(10, 30, "Sleep stage W")])
    ann = hmc.read_scoring_csv(
        "Onset,Duration,Annotation\n0,30,Sleep stage W\n30,30,Sleep stage R\n"
    )
    assert hmc.stage_epochs(ann) == {0: 0, 1: 4}
    cat = hmc.catalogue(
        [
            "recordings/SN001.edf",
            "recordings/SN001_sleepscoring.txt",
            "recordings/SN001_sleepscoring.edf",
        ]
    )
    assert cat == {"SN001": ("recordings/SN001.edf", "recordings/SN001_sleepscoring.edf")}


def test_hmc_wrong_unit_or_rate_is_unreadable(tmp_path, rng) -> None:
    (tmp_path / "recordings").mkdir()
    d = rng.integers(-100, 100, size=60 * 128).astype(np.int16)
    bad = _eeg(d, 60, spr=128)  # 128 Hz, not 256
    (tmp_path / "recordings/SN001.edf").write_bytes(_edf([bad], 60, 1.0))
    (tmp_path / "recordings/SN001_sleepscoring.edf").write_bytes(
        _hypnogram([(0, 30, "Sleep stage W"), (30, 30, "Sleep stage N1")])
    )
    psg = (tmp_path / "recordings/SN001.edf").read_bytes()
    hyp = (tmp_path / "recordings/SN001_sleepscoring.edf").read_bytes()
    with pytest.raises(ValueError, match="128"):
        hmc.night_epochs(psg, "SN001_sleepscoring.edf", hyp)


# --- ADR 0028 §1 -------------------------------------------------------------------------------


def test_fix15_r5_scale_is_in_the_recipe_and_not_loader_order() -> None:
    from kai.tasks.pipeline import R5_SCALE_SAMPLE, FoldConfig

    assert FoldConfig().recipe()["r5_scale"] == "random_train"
    assert R5_SCALE_SAMPLE >= 4096


def test_fix16_classical_cache_key_sees_every_window() -> None:
    import inspect

    from kai.tasks import pipeline

    assert "[::97]" not in inspect.getsource(pipeline.classical_outputs)
    assert "ascontiguousarray(ws.x)" in inspect.getsource(pipeline.classical_outputs)


def test_fix17_deterministic_context_is_strict(monkeypatch) -> None:
    import os

    import torch

    from kai.tasks import train

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"]
    with train._deterministic():
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()


def test_fix19_range_and_finite_checks() -> None:
    from kai.contracts import frame as fr
    from kai.contracts.windows import WindowSet

    with pytest.raises(ValueError, match="non-finite"):
        quantise(np.array([0.0, np.nan]), 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        quantise(np.array([np.inf]), 1.0)
    one = np.zeros((1, 4), np.int16)
    args = (np.zeros(1, np.int64), np.array(["a"]), np.array(["c"]))
    with pytest.raises(ValueError, match="label"):
        WindowSet(one, *args, np.array([-1], np.int64), 1.0, "t")
    WindowSet(one, *args, np.array([7], np.int64), 1.0, "t")  # classes beyond CLASS_NAMES are fine
    for bad in ({"window_id": 2**32}, {"window_id": -1}, {"rep_code": 256}):
        f = fr.Frame(
            **{
                "rep_code": 1,
                "contract_hash": b"abcd",
                "window_id": 0,
                "ts_ns": 0,
                "payload": b"",
                **bad,
            }
        )
        with pytest.raises(fr.FrameError):
            fr.pack(f)


def test_hmc_hash_mismatch_is_corrupt_and_skipped(tmp_path, rng) -> None:
    stages = [(0, 30 * 70, "Sleep stage W"), (30 * 70, 30 * 3, "Sleep stage N2")]
    _night(tmp_path, "SN001", 160, stages, rng)
    _night(tmp_path, "SN002", 160, stages, rng)
    names = ["SN001.edf", "SN001_sleepscoring.edf", "SN002.edf", "SN002_sleepscoring.edf"]
    _sums(tmp_path, names)
    with (tmp_path / "recordings/SN002.edf").open("ab") as f:
        f.write(b"x")  # flipped after the sums were taken
    ws, report = hmc.build_windowset(tmp_path)
    assert report["skipped_nights"]["corrupt"] == ["SN002"]
    assert set(ws.bearing_id.tolist()) == {"SN001"}
