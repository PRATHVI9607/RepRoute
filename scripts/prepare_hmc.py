"""Build transfer/hmc/data/processed/hmc.npz and the L1 split manifest from raw
HMC sleep staging v1.1."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kai.data import hmc
from kai.data.splits import make_l1_splits, save_manifest

WORK = Path(__file__).resolve().parents[1] / "transfer" / "hmc"


def main(
    full_scale_uv: float = hmc.FULL_SCALE_UV,
    seed: int = 20261110,
    n_folds: int = 5,
    n_repeats: int = 3,
) -> None:
    ws, report = hmc.build_windowset(WORK / "data/raw", full_scale_uv)
    skipped = report["skipped_nights"]
    n_skipped = sum(len(v) for v in skipped.values())
    typer.echo(f"skipped nights per reason: { {k: len(v) for k, v in skipped.items()} }")
    if n_skipped > 0.05 * report["n_catalogued"]:  # a partial or corrupt download must not pass
        raise typer.BadParameter(
            f"{n_skipped}/{report['n_catalogued']} nights skipped (> 5 %): {skipped}"
        )
    out = WORK / "data/processed/hmc.npz"
    ws.save(out)
    # a subject has epochs of every stage, so no single class (ws.bearing_labels() raises):
    # the splits are leave-subjects-out with all subjects in one stratum
    strata = dict.fromkeys(sorted(set(ws.bearing_id.tolist())), 0)
    l1 = make_l1_splits(strata, ws.dataset_id, seed, n_folds, n_repeats)
    path = save_manifest(l1, WORK / "data/splits")  # one condition, so no L2 manifest
    (WORK / "data/processed/hmc.report.json").write_text(json.dumps(report, indent=1))
    typer.echo(json.dumps(report))
    typer.echo(f"wrote {out} and {path.name}")


if __name__ == "__main__":
    typer.run(main)
