"""Download the Sleep-EDF Sleep Cassette nights into transfer/sleepedf/data/raw. Resumable.

Source: https://physionet.org/content/sleep-edfx/1.0.0/ (Kemp et al., IEEE TBME 47(9), 2000;
Goldberger et al., Circulation 101(23), 2000). Licence: Open Data Commons Attribution License
v1.0; cite the dataset and never commit the data. Only sleep-cassette/*-PSG.edf and
*-Hypnogram.edf (153 nights, ~7.4 GB), not the telemetry study. Everything, SHA256SUMS.txt
included, comes from PhysioNet's own AWS Open Data mirror (physionet.org serves ~3 KB/s); every
file is checked against that SHA256SUMS.txt. A broken transfer leaves a `.part` file that the next
run resumes; a `.part` that completes with the wrong bytes is deleted.
"""

from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from kai.data import sleepedf

MIRROR = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/"
RAW = Path(__file__).resolve().parents[1] / "transfer" / "sleepedf" / "data" / "raw"
CURL = ["curl", "-sSL", "--proto", "=https", "--retry", "8", "--retry-delay", "5"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _fetch(name: str, sha256: str) -> bool:
    """Fetch one file unless already verified; True if it ends up with the published bytes."""
    dst = RAW / name
    if dst.exists():
        return _sha256(dst) == sha256
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    subprocess.run([*CURL, "-C", "-", "-o", str(part), MIRROR + name], check=False)
    if part.exists() and _sha256(part) == sha256:
        part.rename(dst)
        return True
    part.unlink(missing_ok=True)  # wrong bytes cannot be fixed by resuming: start over next run
    return False


def main(jobs: int = 12) -> None:
    """Fetch the checksums, then the PSG and hypnogram of every Sleep Cassette night."""
    RAW.mkdir(parents=True, exist_ok=True)
    sums_path = RAW / sleepedf.SUMS
    if not sums_path.exists():
        part = sums_path.with_suffix(".part")
        subprocess.run([*CURL, "-C", "-", "-o", str(part), MIRROR + sleepedf.SUMS], check=True)
        part.rename(sums_path)
    sums = sleepedf.read_sums(RAW)
    nights = sleepedf.catalogue(sums)  # names are validated there before they become paths
    names = [n for _, psg, hyp in nights.values() for n in (psg, hyp)]
    with ThreadPoolExecutor(jobs) as pool:
        ok = list(pool.map(lambda n: _fetch(n, sums[n]), names))
    bad = [n for n, good in zip(names, ok, strict=True) if not good]
    total_bytes = sum((RAW / n).stat().st_size for n in names if (RAW / n).exists())
    typer.echo(f"{len(nights)} nights, {total_bytes} bytes of EDF in {RAW}")
    if bad:
        typer.echo(f"INCOMPLETE: {len(bad)} files, e.g. {bad[:5]} — re-run to resume")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
