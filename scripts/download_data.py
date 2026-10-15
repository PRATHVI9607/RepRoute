"""Download the Paderborn (PU) bearing archives into data/raw/pu. Resumable and size-verified.

Source: https://groups.uni-paderborn.de/kat/BearingDataCenter/ (Lessmeier et al., PHME 2016).
Licence (https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/): Creative Commons
Attribution-NonCommercial 4.0. Non-commercial academic use with a citation; never commit the data.
"""

from __future__ import annotations

import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from kai.data.pu import BEARINGS

BASE = "https://groups.uni-paderborn.de/kat/BearingDataCenter"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "pu"


def _remote_size(name: str) -> int:
    req = urllib.request.Request(f"{BASE}/{name}.rar", method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def _fetch(name: str) -> tuple[str, bool]:
    dst = RAW / f"{name}.rar"
    want = _remote_size(name)
    if dst.exists() and dst.stat().st_size == want:
        return name, True
    subprocess.run(
        ["curl", "-sSL", "--retry", "8", "--retry-delay", "5", "-C", "-", "-o", str(dst),
         f"{BASE}/{name}.rar"],
        check=False,
    )  # fmt: skip
    return name, dst.exists() and dst.stat().st_size == want


def main(jobs: int = 6, include_combined: bool = False) -> None:
    """Fetch every bearing archive needed for the 3-class task (add --include-combined for KB*)."""
    RAW.mkdir(parents=True, exist_ok=True)
    names = [b for b, meta in BEARINGS.items() if include_combined or meta.label is not None]
    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(_fetch, names))
    bad = [n for n, ok in results if not ok]
    typer.echo(f"{len(results) - len(bad)}/{len(results)} archives complete in {RAW}")
    if bad:
        typer.echo(f"INCOMPLETE: {bad} — re-run to resume")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
