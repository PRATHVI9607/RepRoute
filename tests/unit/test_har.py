"""HAR loader on tiny fake HAPT RawData text files in the published layout (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from kai.data import har


def _write_acc(raw, exp: int, user: int, acc: np.ndarray) -> None:
    path = har.acc_path(raw, exp, user)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, acc, fmt="%.17g")


def _write_labels(raw, rows: list[tuple[int, int, int, int, int]]) -> None:
    (raw / har.RAW).mkdir(parents=True, exist_ok=True)
    text = "".join(" ".join(map(str, r)) + "\n" for r in rows)
    (raw / har.RAW / "labels.txt").write_text(text)


def test_quantise_full_scale_and_clipping() -> None:
    q = har.quantise(np.array([0.0, 1.0, 4.0, 5.0]))
    assert q.tolist() == [0, round(32767 / 4), 32767, 32767]


def test_windowset_from_fake_recordings(tmp_path, rng: np.random.Generator) -> None:
    acc1 = rng.uniform(-1.5, 1.5, size=(1000, 3))
    acc1[700:710] = [4.0, 4.0, 4.0]  # magnitude ≈ 6.9 g: clipped at the 4 g rail
    acc2 = rng.uniform(-1.0, 1.0, size=(300, 3))
    _write_acc(tmp_path, 1, 1, acc1)
    _write_acc(tmp_path, 2, 7, acc2)
    _write_acc(tmp_path, 4, 9, np.ones((300, 2)))  # two columns: unreadable
    _write_labels(
        tmp_path,
        [
            (1, 1, 5, 11, 530),  # 520 samples → 2 windows (STANDING), 8 samples left over
            (1, 1, 7, 531, 600),  # transition: dropped
            (1, 1, 1, 601, 856),  # exactly one window (WALKING), includes the clipped samples
            (1, 1, 6, 857, 1000),  # 144 samples < 256: no window
            (2, 7, 3, 1, 256),  # one window (DOWNSTAIRS) of user 7
            (2, 7, 4, 200, 400),  # end past the recording
            (3, 8, 1, 1, 256),  # recording not downloaded
            (4, 9, 1, 1, 256),  # unreadable recording
        ],
    )

    ws, report = har.build_windowset(tmp_path, window_n=256)

    assert (len(ws), ws.window_n) == (4, 256)
    assert (ws.sample_rate_hz, ws.dataset_id) == (50.0, "har-v1")
    assert ws.bearing_id.tolist() == ["U1", "U1", "U1", "U7"]
    assert ws.label.tolist() == [4, 4, 0, 2]
    assert set(ws.condition.tolist()) == {"lab"}
    mag1 = np.sqrt((acc1**2).sum(axis=1))
    # 1-based inclusive start 11 → index 10; windows stay inside their interval
    expected = [mag1[10:266], mag1[266:522], mag1[600:856], np.sqrt((acc2[:256] ** 2).sum(axis=1))]
    for got, want in zip(ws.x, expected, strict=True):
        assert np.array_equal(got, har.quantise(want))
    assert report["windows_per_class"] == {
        "WALKING": 1,
        "WALKING_UPSTAIRS": 0,
        "WALKING_DOWNSTAIRS": 1,
        "SITTING": 0,
        "STANDING": 2,
        "LAYING": 0,
    }
    assert report["dropped_intervals"] == {
        "activity_7": 1,
        "interval_shorter_than_window": 1,
        "interval_out_of_range": 1,
        "recording_unavailable": 2,
    }
    assert report["skipped_recordings"] == {
        "missing": ["acc_exp03_user08.txt"],
        "unreadable": ["acc_exp04_user09.txt"],
    }
    assert (report["n_users"], report["clipped_samples"]) == (2, 10)
    assert report["windows_with_clipping"] == 1


def test_labels_must_have_five_columns(tmp_path) -> None:
    (tmp_path / har.RAW).mkdir()
    (tmp_path / har.RAW / "labels.txt").write_text("1 1 5 1\n")
    with pytest.raises(ValueError, match="columns"):
        har.read_labels(tmp_path)
