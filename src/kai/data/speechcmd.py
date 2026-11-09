"""Speech Commands v0.02 (transfer dataset, other sensor domain): 1-s WAV clips → int16 WindowSet.

Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition", arXiv:1804.03209
(2018), CC BY 4.0. 16-bit PCM mono WAV at 16 kHz, one directory per word, files named
<speaker hash>_nohash_<n>.wav. Here the "bearing" is the speaker, so splits leave speakers out;
the dataset's own validation/testing lists are not used. Only the 10 classic keywords are kept.
Each clip (≤ 1 s) is zero-padded at the end to `window_n` samples, int16 as stored. At most
`max_per_word` clips per (speaker, word) are kept, chosen with the caller's Generator so that one
prolific speaker does not dominate. WAV files are untrusted data: parsed by scipy, never executed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

from kai.contracts.windows import WindowSet

SAMPLE_RATE_HZ = 16000.0
DATASET_ID = "speechcmd-v1"
WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
LABELS = {w: k for k, w in enumerate(WORDS)}
CONDITION = "clip"
_NAME = re.compile(r"([0-9a-f]{8})_nohash_\d+\.wav")


def speaker_of(name: str) -> str | None:
    """'0a7c2a8d_nohash_1.wav' → 'SPK0a7c2a8d'; None for a name outside the published pattern."""
    m = _NAME.fullmatch(name)
    return f"SPK{m.group(1)}" if m else None


def read_clip(path: Path, window_n: int = 16384) -> npt.NDArray[np.int16]:
    """The clip as int16, zero-padded at the end to `window_n`. ValueError if it is not 16 kHz
    ("wrong_rate") or not mono int16 with 1..window_n samples ("wrong_format")."""
    rate_hz, x = wavfile.read(path)
    if rate_hz != SAMPLE_RATE_HZ:
        raise ValueError(f"wrong_rate: {path.name} is {rate_hz} Hz")
    if x.dtype != np.int16 or x.ndim != 1 or not 0 < len(x) <= window_n:
        raise ValueError(f"wrong_format: {path.name} is {x.dtype} {x.shape}")
    out = np.zeros(window_n, dtype=np.int16)
    out[: len(x)] = x
    return out


def build_windowset(
    raw_dir: Path, rng: np.random.Generator, window_n: int = 16384, max_per_word: int = 4
) -> tuple[WindowSet, dict[str, object], list[str]]:
    """WindowSet of the kept clips, the report, and the kept files (raw_dir-relative, window order).

    Groups are visited in (word, speaker) order and every file of a group is validated before
    `rng` picks `max_per_word` of the valid ones, so the selection is a function of data and seed.
    """
    groups: dict[tuple[str, str], list[Path]] = {}
    rejected: dict[str, list[str]] = {
        k: [] for k in ("bad_filename", "wrong_rate", "wrong_format", "unreadable")
    }
    for word in WORDS:
        for p in sorted((raw_dir / word).glob("*.wav")):
            spk = speaker_of(p.name)
            if spk is None:
                rejected["bad_filename"].append(f"{word}/{p.name}")
            else:
                groups.setdefault((word, spk), []).append(p)
    xs, bids, labels, files = [], [], [], []
    n_capped = 0
    for (word, spk), paths in groups.items():
        valid = []
        for p in paths:
            try:
                valid.append((p, read_clip(p, window_n)))
            except ValueError as e:  # wrong rate / format, or a malformed WAV: counted, not hidden
                reason = str(e).split(":")[0]
                rejected[reason if reason in rejected else "unreadable"].append(f"{word}/{p.name}")
        if len(valid) > max_per_word:
            n_capped += len(valid) - max_per_word
            keep = np.sort(rng.choice(len(valid), max_per_word, replace=False))
            valid = [valid[i] for i in keep]
        for p, x in valid:
            xs.append(x)
            bids.append(spk)
            labels.append(LABELS[word])
            files.append(f"{word}/{p.name}")
    x = np.stack(xs)
    y = np.asarray(labels, dtype=np.int64)
    ws = WindowSet(
        x=x,
        window_id=np.arange(len(x), dtype=np.int64),
        bearing_id=np.asarray(bids),
        condition=np.full(len(x), CONDITION),
        label=y,
        sample_rate_hz=SAMPLE_RATE_HZ,
        dataset_id=DATASET_ID,
    )
    report: dict[str, object] = {
        "n_windows": len(x),
        "n_speakers": len(set(bids)),
        "windows_per_class": {w: int((y == k).sum()) for w, k in LABELS.items()},
        "n_files_found": sum(len(v) for v in groups.values()) + len(rejected["bad_filename"]),
        "n_dropped_by_cap": n_capped,
        "rejected_files": rejected,
        "min": int(x.min()),
        "max": int(x.max()),
        "window_n": window_n,
        "max_per_word": max_per_word,
    }
    return ws, report, files
