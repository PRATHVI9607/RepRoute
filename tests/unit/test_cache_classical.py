"""Encode cache of the classical codecs: one key per codec, atomic files, same values."""

from __future__ import annotations

import numpy as np

from kai.certify.probe import probe
from kai.contracts.windows import WindowSet
from kai.reps.registry import classical_menu
from kai.tasks.pipeline import classical_outputs


def test_an_opt_in_codec_encodes_only_itself_and_the_files_are_complete(
    ws: WindowSet, tmp_path
) -> None:
    sub = ws.take(np.arange(600))  # two chunks of rows
    base = classical_outputs(sub, tmp_path, workers=2)
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    with_logmel = classical_outputs(sub, tmp_path, workers=2, logmel=True)
    after = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    assert {k: after[k] for k in before} == before  # the other codecs were not encoded again
    new = sorted(set(after) - set(before))
    assert len(new) == 4
    assert all("logmel" in n for n in new)
    assert not any(n.endswith(".tmp") for n in after)
    assert set(base) < set(with_logmel)
    for rep in classical_menu(sub.window_n, logmel_hz=sub.sample_rate_hz):
        sizes = [len(rep.encode(w)) for w in sub.x]
        assert np.array_equal(with_logmel[f"{rep.rep_id}.bytes"], sizes)
        if rep.rep_id in ("r4_summary", "r10b_logmel"):
            decoded = np.stack([rep.decode_for_head(rep.encode(w)) for w in sub.x])
            assert np.array_equal(with_logmel[f"{rep.rep_id}.out"], decoded)
    phi = np.stack([probe(w) for w in sub.x]).astype(np.float32)
    assert np.array_equal(with_logmel["phi.out"], phi)
    assert with_logmel["phi.bytes"].shape == (0,)
    assert with_logmel["r0_raw.out"].shape == (600, 0)
