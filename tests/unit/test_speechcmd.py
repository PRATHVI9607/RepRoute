"""Speech Commands loader on tiny synthetic WAVs in the published layout (no network)."""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from kai.data import speechcmd


def _wav(raw, rel: str, x: np.ndarray, rate_hz: int = 16000) -> None:
    path = raw / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, rate_hz, x)


def test_speaker_of() -> None:
    assert speechcmd.speaker_of("0a7c2a8d_nohash_12.wav") == "SPK0a7c2a8d"
    for bad in ("0a7c2a8d_nohash_1.wav.txt", "../0a7c2a8d_nohash_1.wav", "zz7c2a8d_nohash_1.wav"):
        assert speechcmd.speaker_of(bad) is None


def test_windowset_from_fake_clips(tmp_path, rng: np.random.Generator) -> None:
    clips = {
        "yes/aaaaaaaa_nohash_0.wav": rng.integers(-32768, 32768, 16000).astype(np.int16),
        "go/aaaaaaaa_nohash_0.wav": rng.integers(-500, 500, 9000).astype(np.int16),  # padded
        "no/bbbbbbbb_nohash_0.wav": rng.integers(-500, 500, 16000).astype(np.int16),
    }
    for rel, x in clips.items():
        _wav(tmp_path, rel, x)
    capped = [f"up/cccccccc_nohash_{i}.wav" for i in range(6)]  # 6 clips, the cap keeps 4
    for i, rel in enumerate(capped):
        _wav(tmp_path, rel, np.full(100, i, dtype=np.int16))
    _wav(tmp_path, "yes/dddddddd_nohash_0.wav", clips["no/bbbbbbbb_nohash_0.wav"], rate_hz=8000)
    _wav(tmp_path, "yes/eeeeeeee_nohash_0.wav", np.zeros(100, np.float32))  # not int16
    _wav(tmp_path, "yes/ffffffff_nohash_0.wav", np.zeros((100, 2), np.int16))  # stereo
    _wav(tmp_path, "yes/abc.wav", np.zeros(100, np.int16))  # outside the naming pattern
    _wav(tmp_path, "bed/aaaaaaaa_nohash_0.wav", np.zeros(100, np.int16))  # not a keyword: ignored
    (tmp_path / "no/99999999_nohash_0.wav").write_bytes(b"RIFF not a wav")

    ws, report, files = speechcmd.build_windowset(tmp_path, np.random.default_rng(1))

    assert (len(ws), ws.window_n, ws.x.dtype) == (7, 16384, np.int16)
    assert (ws.sample_rate_hz, ws.dataset_id) == (16000.0, "speechcmd-v1")
    assert set(ws.condition.tolist()) == {"clip"}
    for rel, x in clips.items():
        k = files.index(rel)
        assert np.array_equal(ws.x[k, : len(x)], x)  # stored samples, no rescaling
        assert not ws.x[k, len(x) :].any()  # zero padding at the end
        assert ws.label[k] == speechcmd.WORDS.index(rel.split("/")[0])
        assert ws.bearing_id[k] == "SPK" + rel.split("/")[1][:8]
    kept = [f for f in files if f.startswith("up/")]
    assert len(kept) == 4
    assert set(kept) <= set(capped)
    assert report["n_dropped_by_cap"] == 2
    _, _, again = speechcmd.build_windowset(tmp_path, np.random.default_rng(1))
    assert again == files  # the selection depends only on data and seed
    assert report["windows_per_class"] == {
        **dict.fromkeys(speechcmd.WORDS, 0),
        **{"yes": 1, "no": 1, "up": 4, "go": 1},
    }
    assert report["rejected_files"] == {
        "bad_filename": ["yes/abc.wav"],
        "wrong_rate": ["yes/dddddddd_nohash_0.wav"],
        "wrong_format": ["yes/eeeeeeee_nohash_0.wav", "yes/ffffffff_nohash_0.wav"],
        "unreadable": ["no/99999999_nohash_0.wav"],
    }
    assert (report["n_speakers"], report["n_files_found"]) == (3, 14)
