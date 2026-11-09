"""Build transfer/har/data/processed/har.npz and the L1 split manifest from raw HAPT recordings."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kai.data import har
from kai.data.splits import make_l1_splits, save_manifest

WORK = Path(__file__).resolve().parents[1] / "transfer" / "har"


def main(window_n: int = 256, seed: int = 20261012, n_folds: int = 5, n_repeats: int = 3) -> None:
    ws, report = har.build_windowset(WORK / "data/raw", window_n)
    if report["skipped_recordings"] != {"missing": [], "unreadable": []}:  # partial download
        raise typer.BadParameter(f"recordings missing, re-run scripts/download_har.py: {report}")
    out = WORK / "data/processed/har.npz"
    ws.save(out)
    # every user performs all six activities, so (as for UORED) one stratum holds all users
    strata = dict.fromkeys(sorted(set(ws.bearing_id.tolist())), 0)
    l1 = make_l1_splits(strata, ws.dataset_id, seed, n_folds, n_repeats)
    path = save_manifest(l1, WORK / "data/splits")  # one condition, so no L2 manifest
    (WORK / "data/processed/har.report.json").write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report))
    typer.echo(f"wrote {out} and {path.name}")


if __name__ == "__main__":
    typer.run(main)
