"""Build data/processed/pu.npz and the committed split manifests from the raw PU archives."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kai.bench.config import work
from kai.data import pu
from kai.data.splits import make_l1_splits, make_l2_splits, save_manifest

ROOT = Path(__file__).resolve().parents[1]


def main(
    window_n: int = 4096,
    n_measurements: int = 5,
    full_scale: float = 16.0,
    seed: int = 20261012,
    n_folds: int = 5,
    n_repeats: int = 20,
) -> None:
    ws, report = pu.build_windowset(
        ROOT / "data/raw/pu", ROOT / "data/interim/pu", window_n, n_measurements, full_scale
    )
    if float(report["clipped_fraction"]) > 1e-5:  # type: ignore[arg-type]
        raise typer.BadParameter(f"full_scale too small: {report}")
    out = work() / "data/processed/pu.npz"  # KAI_WORK=… builds e.g. a long-window variant elsewhere
    ws.save(out)
    l1 = make_l1_splits(ws.bearing_labels(), ws.dataset_id, seed, n_folds, n_repeats)
    l2 = make_l2_splits(l1, list(pu.CONDITIONS))
    paths = [save_manifest(m, work() / "data/splits") for m in (l1, l2)]
    (work() / "data/processed/pu.report.json").write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report))
    typer.echo(f"wrote {out} and {[p.name for p in paths]}")


if __name__ == "__main__":
    typer.run(main)
