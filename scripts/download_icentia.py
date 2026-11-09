"""Download the first segment of patients p00000..p00999 of Icentia11k into
transfer/icentia/data/raw. Resumable.

Source: https://physionet.org/content/icentia11k-continuous-ecg/1.0/ (Tan et al., PhysioNet 2022,
doi:10.13026/kk0v-r952). Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International; research use only, cite the dataset and never commit the data. Only the .hea /
.dat / .atr of one ~70-min segment per patient (~2.2 GB), not the 188 GB dataset. Everything,
SHA256SUMS.txt (149 MB) included, comes from PhysioNet's AWS Open Data mirror (physionet.org
throttles), and every file is checked against that SHA256SUMS.txt.
"""

from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from kai.data import icentia

MIRROR = "https://physionet-open.s3.amazonaws.com/icentia11k-continuous-ecg/1.0/"
RAW = Path(__file__).resolve().parents[1] / "transfer" / "icentia" / "data" / "raw"
CURL = ["curl", "-sSL", "--proto", "=https", "--retry", "8", "--retry-delay", "5"]


def _complete(dst: Path, sha256: str) -> bool:
    return dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == sha256


def _fetch(names: list[str], sums: dict[str, str]) -> list[str]:
    """Fetch the named files from the mirror unless verified; return those failing SHA-256."""
    todo = [n for n in names if not _complete(RAW / n, sums[n])]
    for n in todo:
        dst = RAW / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():  # wrong bytes cannot be fixed by resuming: start over
            dst.unlink()
    if todo:
        args = [a for n in todo for a in ("-o", str(RAW / n), MIRROR + n)]
        subprocess.run([*CURL, *args], check=False)
    return [n for n in todo if not _complete(RAW / n, sums[n])]


def main(jobs: int = 8, n_patients: int = icentia.N_PATIENTS) -> None:
    """Fetch the checksums and the licence, then the first segment of each kept patient."""
    RAW.mkdir(parents=True, exist_ok=True)
    sums_path = RAW / icentia.SUMS
    if not sums_path.exists():
        part = sums_path.with_suffix(".part")
        subprocess.run([*CURL, "-C", "-", "-o", str(part), MIRROR + icentia.SUMS], check=True)
        part.rename(sums_path)
    sums = icentia.read_sums(RAW)
    bad = _fetch(["LICENSE.txt"], sums)
    if bad:
        raise typer.BadParameter(f"licence failed its checksum: {bad}")
    records = icentia.catalogue(list(sums), n_patients)  # names are validated there
    names = [f"{r}{ext}" for r in records for ext in (".hea", ".dat", ".atr")]
    batches = [names[i : i + 60] for i in range(0, len(names), 60)]  # one curl call per batch
    with ThreadPoolExecutor(jobs) as pool:
        bad = [n for failed in pool.map(lambda b: _fetch(b, sums), batches) for n in failed]
    total_bytes = sum((RAW / n).stat().st_size for n in names if (RAW / n).exists())
    typer.echo(f"{len(records)} records, {total_bytes} bytes of .hea/.dat/.atr in {RAW}")
    if bad:
        typer.echo(f"INCOMPLETE: {len(bad)} files, e.g. {bad[:5]} — re-run to resume")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
