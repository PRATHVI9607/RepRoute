"""HUST loader on tiny fake recordings in the published MAT layout (no network, no real data)."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.io import savemat

from kai.data import hust


def _write(path, signal: np.ndarray) -> None:
    # the published layout: `data` is [n, 1] float64, `fs` is the shaft frequency (not the rate)
    savemat(path, {"data": signal.reshape(-1, 1), "fs": 24.93, "ru_raw": np.zeros((8, 1))})


def test_catalogue_matches_the_published_file_names() -> None:
    assert len(hust.RECORDINGS) == 45  # 3 classes × 5 bearing types × 3 loads
    assert hust.RECORDINGS["I402.mat"] == ("I6204", "200W", 1)
    assert hust.RECORDINGS["O804.mat"] == ("O6208", "400W", 2)
    bearings = {b: label for b, _, label in hust.RECORDINGS.values()}
    assert Counter(bearings.values()) == {0: 5, 1: 5, 2: 5}  # healthy / inner race / outer race
    assert not any(name.startswith(("B", "IB", "IO", "OB")) for name in hust.RECORDINGS)


def test_windowset_from_fake_recordings(tmp_path, rng: np.random.Generator) -> None:
    signals = {
        "N400.mat": rng.normal(0, 0.2, 10_000),  # 2 windows, the 1808-sample tail is dropped
        "N402.mat": rng.normal(0, 0.2, 4096),  # same bearing, another load
        "I502.mat": rng.normal(0, 2.0, 3 * 4096),
        "O804.mat": rng.normal(0, 1.0, 5000),
        "B500.mat": rng.normal(0, 1.0, 4096),  # ball defect: no 3-class label
        "IO600.mat": rng.normal(0, 1.0, 4096),  # combined defect
    }
    for name, signal in signals.items():
        _write(tmp_path / name, signal)
    savemat(tmp_path / "N600.mat", {"fs": 24.9})  # in the catalogue but without `data`

    ws, report = hust.build_windowset(tmp_path)

    assert (len(ws), ws.window_n) == (7, 4096)
    assert (ws.sample_rate_hz, ws.dataset_id) == (51_200.0, "hust-v1")
    assert ws.bearing_labels() == {"N6204": 0, "I6205": 1, "O6208": 2}
    meta = zip(ws.bearing_id.tolist(), ws.condition.tolist(), ws.label.tolist(), strict=True)
    rows = Counter(meta)
    assert rows == {
        ("N6204", "0W", 0): 2,
        ("N6204", "200W", 0): 1,
        ("I6205", "200W", 1): 3,
        ("O6208", "400W", 2): 1,
    }
    first_n = ws.x[ws.bearing_id == "N6204"][0]
    expected = np.rint(signals["N400.mat"][:4096] / hust.FULL_SCALE * 32767)
    assert np.array_equal(first_n, expected)

    assert report["dropped_files"] == ["B500.mat", "IO600.mat"]
    assert report["skipped_recordings"] == ["N600.mat"]
    assert (report["n_windows"], report["n_bearings"], report["n_recordings"]) == (7, 3, 4)
    assert report["clipped_fraction"] == 0.0
    used = ("N400.mat", "N402.mat", "I502.mat", "O804.mat")
    assert report["peak_abs"] == max(np.abs(signals[n]).max() for n in used)

    _, clipped = hust.build_windowset(tmp_path, full_scale=1.0)
    assert clipped["clipped_fraction"] > 0  # a too-small full scale is reported, never silent
