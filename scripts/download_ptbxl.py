"""Download the PTB-XL files the 3-class task uses into transfer/ptbxl/data/raw. Resumable.

Source: https://physionet.org/content/ptb-xl/1.0.3/ (Wagner et al., Sci Data 7, 154, 2020,
doi:10.13026/kfzx-aw45). Licence: Creative Commons Attribution 4.0 International Public License;
cite the dataset and never commit the data. Not the 1.7 GB zip: only the metadata CSVs and the
500 Hz .hea/.dat of the 10,266 kept records (~1.25 GB). Everything, SHA256SUMS.txt included, comes
from PhysioNet's own AWS Open Data mirror (physionet.org serves ~3 KB/s); every file is checked
against that SHA256SUMS.txt, whose first 3.5 MB were found byte-identical to physionet.org's copy.
"""

from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from kai.data import ptbxl

MIRROR = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
RAW = Path(__file__).resolve().parents[1] / "transfer" / "ptbxl" / "data" / "raw"
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


def main(jobs: int = 8) -> None:
    """Fetch the checksums, the metadata, then the 500 Hz records of the kept patients."""
    RAW.mkdir(parents=True, exist_ok=True)
    sums_path = RAW / "SHA256SUMS.txt"
    if not sums_path.exists():
        part = sums_path.with_suffix(".part")
        subprocess.run([*CURL, "-C", "-", "-o", str(part), MIRROR + "SHA256SUMS.txt"], check=True)
        part.rename(sums_path)
    sums = {
        name: digest
        for digest, name in (s.split(maxsplit=1) for s in sums_path.read_text().split("\n") if s)
    }
    bad = _fetch([ptbxl.DATABASE, ptbxl.STATEMENTS, "LICENSE.txt"], sums)
    if bad:
        raise typer.BadParameter(f"metadata failed its checksum: {bad}")
    records, _ = ptbxl.catalogue(RAW)  # names are validated there before they become paths
    names = [f"{r}{ext}" for r in records for ext in (".hea", ".dat")]
    batches = [names[i : i + 100] for i in range(0, len(names), 100)]  # one curl call per batch
    with ThreadPoolExecutor(jobs) as pool:
        bad = [n for failed in pool.map(lambda b: _fetch(b, sums), batches) for n in failed]
    total_bytes = sum((RAW / n).stat().st_size for n in names if (RAW / n).exists())
    typer.echo(f"{len(records)} records, {total_bytes} bytes of .hea/.dat in {RAW}")
    if bad:
        typer.echo(f"INCOMPLETE: {len(bad)} files, e.g. {bad[:5]} — re-run to resume")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
