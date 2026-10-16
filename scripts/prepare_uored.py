"""Build transfer/uored/data/processed/uored.npz and the split manifests from the raw UORED data."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kai.data import uored
from kai.data.splits import make_l1_splits, make_l2_splits, save_manifest

WORK = Path(__file__).resolve().parents[1] / "transfer" / "uored"


def main(
    window_n: int = 4096,
    full_scale: float = uored.FULL_SCALE,
    seed: int = 20261015,
    n_folds: int = 5,
    n_repeats: int = 20,
) -> None:
    ws, report = uored.build_windowset(WORK / "data/raw", window_n, full_scale)
    if float(report["clipped_fraction"]) > 1e-5:  # type: ignore[arg-type]
        raise typer.BadParameter(f"full_scale too small: {report}")
    if report["n_recordings"] != len(uored.RECORDINGS):  # a partial download must not pass silently
        raise typer.BadParameter(f"recordings missing, re-run scripts/download_uored.py: {report}")
    out = WORK / "data/processed/uored.npz"
    ws.save(out)
    # a bearing is healthy and faulty, so stratify by its fault type (inner/outer/ball/cage)
    l1 = make_l1_splits(uored.STRATA, ws.dataset_id, seed, n_folds, n_repeats)
    conditions = sorted(set(ws.condition.tolist()))
    # leave-condition-out (L2) needs ≥ 2 conditions; the published 3-class subset has one
    manifests = [l1, make_l2_splits(l1, conditions)] if len(conditions) > 1 else [l1]
    paths = [save_manifest(m, WORK / "data/splits") for m in manifests]
    (WORK / "data/processed/uored.report.json").write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report))
    typer.echo(f"wrote {out} and {[p.name for p in paths]}; conditions {conditions}")


if __name__ == "__main__":
    typer.run(main)
