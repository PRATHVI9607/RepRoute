"""UORED loader on tiny fake recordings in the published MAT layout (no network, no real data)."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.io import savemat

from kai.data import uored
from kai.data.splits import make_l1_splits


def _write(path, acc: np.ndarray, load_n: float = 400.0) -> None:
    # the published layout: one variable named like the file stem, [n, 5] float64 with columns
    # accelerometer, microphone, speed, load, temperature; speed and load are logged in row 0 only
    table = np.zeros((len(acc), 5))
    table[:, 0] = acc
    table[:, 1] = 0.08
    table[0, 2:4] = 1819.0, load_n
    savemat(path, {path.stem: table})


def test_catalogue_matches_the_published_file_names() -> None:
    assert len(uored.ALL_FILES) == 60  # 20 bearings × healthy / developing fault / faulty
    assert len(uored.RECORDINGS) == 30  # 20 healthy + 5 faulty inner + 5 faulty outer
    assert uored.RECORDINGS["H_1_0.mat"] == ("I01", 0)
    assert uored.RECORDINGS["H_20_0.mat"] == ("C20", 0)
    assert uored.RECORDINGS["I_3_2.mat"] == ("I03", 1)
    assert uored.RECORDINGS["O_10_2.mat"] == ("O10", 2)
    assert set(uored.RECORDINGS) <= set(uored.ALL_FILES)
    assert not any(n.endswith("_1.mat") or n[0] in "BC" for n in uored.RECORDINGS)
    assert Counter(uored.STRATA.values()) == {0: 5, 1: 5, 2: 5, 3: 5}


def test_windowset_from_fake_recordings(tmp_path, rng: np.random.Generator) -> None:
    signals = {
        "H_1_0.mat": rng.normal(0, 3.0, 10_000),  # 2 windows, the 1808-sample tail is dropped
        "I_1_2.mat": rng.normal(0, 20.0, 3 * 4096),  # same physical bearing, faulty
        "H_12_0.mat": rng.normal(0, 3.0, 4096),  # ball bearing, healthy state: class 0
        "O_7_2.mat": rng.normal(0, 15.0, 5000),
        "I_1_1.mat": rng.normal(0, 20.0, 4096),  # developing fault: dropped
        "B_12_2.mat": rng.normal(0, 20.0, 4096),  # faulty ball bearing: dropped
    }
    for name, signal in signals.items():
        _write(tmp_path / name, signal)
    savemat(tmp_path / "C_16_0.mat", {"other": np.zeros(3)})  # unknown name: dropped
    savemat(tmp_path / "H_2_0.mat", {"wrong_name": np.zeros((4096, 5))})  # unreadable: skipped

    ws, report = uored.build_windowset(tmp_path)

    assert (len(ws), ws.window_n) == (7, 4096)
    assert (ws.sample_rate_hz, ws.dataset_id) == (42_000.0, "uored-v1")
    rows = Counter(zip(ws.bearing_id.tolist(), ws.label.tolist(), strict=True))
    assert rows == {("I01", 0): 2, ("I01", 1): 3, ("B12", 0): 1, ("O07", 2): 1}
    assert set(ws.condition.tolist()) == {"1750rpm_400N"}
    first = ws.x[ws.bearing_id == "I01"][0]
    expected = np.rint(signals["H_1_0.mat"][:4096] / uored.FULL_SCALE * 32767)
    assert np.array_equal(first, expected)  # the accelerometer column, not the microphone

    assert report["dropped_files"] == ["B_12_2.mat", "C_16_0.mat", "I_1_1.mat"]
    assert report["skipped_recordings"] == ["H_2_0.mat"]
    assert (report["n_windows"], report["n_bearings"], report["n_recordings"]) == (7, 3, 4)
    assert report["clipped_fraction"] == 0.0

    _, clipped = uored.build_windowset(tmp_path, full_scale=1.0)
    assert clipped["clipped_fraction"] > 0  # a too-small full scale is reported, never silent


def test_fault_type_strata_split_every_bearing_whole() -> None:
    manifest = make_l1_splits(uored.STRATA, uored.DATASET_ID, seed=1, n_folds=5, n_repeats=2)
    for fold in manifest.folds:
        parts = [fold.train, fold.ptrain, fold.calib, fold.test]
        assert sorted(b for part in parts for b in part) == sorted(uored.STRATA)  # once each
        for part in parts:  # every partition holds healthy windows; inner and outer faults
            assert {b[0] for b in part} == set(uored.FAULT_TYPES)  # appear in each one
