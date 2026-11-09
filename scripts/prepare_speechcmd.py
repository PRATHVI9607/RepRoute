"""Build transfer/speechcmd/data/processed/speechcmd.npz and the L1 split manifest from raw
Speech Commands v0.02."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from kai.data import speechcmd
from kai.data.splits import make_l1_splits, save_manifest

WORK = Path(__file__).resolve().parents[1] / "transfer" / "speechcmd"


def main(
    window_n: int = 16384,
    max_per_word: int = 4,
    seed: int = 20261012,
    n_folds: int = 5,
    n_repeats: int = 3,
) -> None:
    raw = WORK / "data/raw"
    if not (raw / ".extracted").exists():  # a partial extraction must not pass silently
        raise typer.BadParameter("archive not extracted, run scripts/download_speechcmd.py")
    rng = np.random.default_rng(seed)
    ws, report, files = speechcmd.build_windowset(raw, rng, window_n, max_per_word)
    out = WORK / "data/processed/speechcmd.npz"
    ws.save(out)
    # the kept clips, in window_id order: the cap's random selection is recorded, not re-derived
    out.with_suffix(".files.txt").write_text("\n".join(files) + "\n")
    # every speaker says several words, so a speaker has no single class (ws.bearing_labels()
    # raises): the splits are leave-speakers-out with all speakers in one stratum
    strata = dict.fromkeys(sorted(set(ws.bearing_id.tolist())), 0)
    l1 = make_l1_splits(strata, ws.dataset_id, seed, n_folds, n_repeats)
    path = save_manifest(l1, WORK / "data/splits")  # one condition, so no L2 manifest
    (WORK / "data/processed/speechcmd.report.json").write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report))
    typer.echo(f"wrote {out} and {path.name}")


if __name__ == "__main__":
    typer.run(main)
