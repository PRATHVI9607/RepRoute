"""Build transfer/ptbxl/data/processed/ptbxl.npz and the L1 split manifest from raw PTB-XL."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kai.data import ptbxl
from kai.data.splits import make_l1_splits, save_manifest

WORK = Path(__file__).resolve().parents[1] / "transfer" / "ptbxl"


def main(window_n: int = 4096, seed: int = 20261012, n_folds: int = 5, n_repeats: int = 3) -> None:
    ws, report = ptbxl.build_windowset(WORK / "data/raw", window_n)
    if report["n_windows"] != report["n_catalogued"]:  # a partial download must not pass silently
        raise typer.BadParameter(f"records missing, re-run scripts/download_ptbxl.py: {report}")
    out = WORK / "data/processed/ptbxl.npz"
    ws.save(out)
    # PTB-XL's own strat_fold is not used: the splits are leave-patients-out like every dataset here
    l1 = make_l1_splits(ws.bearing_labels(), ws.dataset_id, seed, n_folds, n_repeats)
    path = save_manifest(l1, WORK / "data/splits")  # one condition, so no L2 manifest
    (WORK / "data/processed/ptbxl.report.json").write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report))
    typer.echo(f"wrote {out} and {path.name}")


if __name__ == "__main__":
    typer.run(main)
