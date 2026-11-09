"""Build transfer/icentia/data/processed/icentia.npz and the L1 split manifest from raw data."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kai.data import icentia
from kai.data.splits import make_l1_splits, save_manifest

WORK = Path(__file__).resolve().parents[1] / "transfer" / "icentia"


def main(
    window_n: int = 4096,
    max_per_class: int = 6,
    seed: int = 20261012,
    n_folds: int = 5,
    n_repeats: int = 3,
) -> None:
    ws, report = icentia.build_windowset(WORK / "data/raw", window_n, max_per_class, seed)
    skipped = report["skipped_records"]
    assert isinstance(skipped, dict)
    if skipped["missing"]:  # a partial download must not pass silently
        raise typer.BadParameter(f"records missing, re-run scripts/download_icentia.py: {report}")
    out = WORK / "data/processed/icentia.npz"
    ws.save(out)
    # a patient holds windows of several classes, so splits stratify by its highest class
    l1 = make_l1_splits(icentia.split_labels(ws), ws.dataset_id, seed, n_folds, n_repeats)
    path = save_manifest(l1, WORK / "data/splits")  # one condition, so no L2 manifest
    (WORK / "data/processed/icentia.report.json").write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report))
    typer.echo(f"wrote {out} and {path.name}")


if __name__ == "__main__":
    typer.run(main)
