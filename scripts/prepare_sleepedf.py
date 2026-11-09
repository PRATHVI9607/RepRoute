"""Build transfer/sleepedf/data/processed/sleepedf.npz and the L1 split manifest from raw
Sleep-EDF Sleep Cassette."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kai.data import sleepedf
from kai.data.splits import make_l1_splits, save_manifest

WORK = Path(__file__).resolve().parents[1] / "transfer" / "sleepedf"


def main(
    full_scale_uv: float = sleepedf.FULL_SCALE_UV,
    seed: int = 20261012,
    n_folds: int = 5,
    n_repeats: int = 3,
) -> None:
    ws, report = sleepedf.build_windowset(WORK / "data/raw", full_scale_uv)
    if report["n_nights"] != report["n_catalogued"]:  # a partial download must not pass silently
        raise typer.BadParameter(f"nights missing, re-run scripts/download_sleepedf.py: {report}")
    out = WORK / "data/processed/sleepedf.npz"
    ws.save(out)
    # a subject has epochs of every stage, so no single class (ws.bearing_labels() raises):
    # the splits are leave-subjects-out with all subjects in one stratum
    strata = dict.fromkeys(sorted(set(ws.bearing_id.tolist())), 0)
    l1 = make_l1_splits(strata, ws.dataset_id, seed, n_folds, n_repeats)
    path = save_manifest(l1, WORK / "data/splits")  # one condition, so no L2 manifest
    (WORK / "data/processed/sleepedf.report.json").write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report))
    typer.echo(f"wrote {out} and {path.name}")


if __name__ == "__main__":
    typer.run(main)
