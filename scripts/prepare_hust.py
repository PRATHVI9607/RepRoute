"""Build transfer/hust/data/processed/hust.npz and the split manifests from the raw HUST files."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kai.data import hust
from kai.data.splits import make_l1_splits, make_l2_splits, save_manifest

WORK = Path(__file__).resolve().parents[1] / "transfer" / "hust"


def main(
    window_n: int = 4096,
    full_scale: float = hust.FULL_SCALE,
    seed: int = 20261012,
    n_folds: int = 5,
    n_repeats: int = 20,
) -> None:
    ws, report = hust.build_windowset(WORK / "data/raw", window_n, full_scale)
    if float(report["clipped_fraction"]) > 1e-5:  # type: ignore[arg-type]
        raise typer.BadParameter(f"full_scale too small: {report}")
    if report["n_recordings"] != len(hust.RECORDINGS):  # a partial download must not pass silently
        raise typer.BadParameter(f"recordings missing, re-run scripts/download_hust.py: {report}")
    out = WORK / "data/processed/hust.npz"
    ws.save(out)
    l1 = make_l1_splits(ws.bearing_labels(), ws.dataset_id, seed, n_folds, n_repeats)
    l2 = make_l2_splits(l1, list(hust.CONDITIONS))
    paths = [save_manifest(m, WORK / "data/splits") for m in (l1, l2)]
    (WORK / "data/processed/hust.report.json").write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report))
    typer.echo(f"wrote {out} and {[p.name for p in paths]}")


if __name__ == "__main__":
    typer.run(main)
