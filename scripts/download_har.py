"""Download HAPT (the raw recordings behind UCI HAR) into transfer/har/data/raw. Resumable.

Source: UCI ML Repository #341, "Smartphone-Based Recognition of Human Activities and Postural
Transitions" (Reyes-Ortiz et al., Neurocomputing 171, 2016), CC BY 4.0; cite it and never commit
the data. Same 30 subjects and lab as UCI HAR #240 (Anguita et al., ESANN 2013), whose zip holds
only pre-cut 128-sample windows, so it is not fetched. The zip (~80 MB) is checked against the
SHA-256 pinned below; only RawData/labels.txt, RawData/acc_*.txt and README.txt are extracted.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

import typer

URL = (
    "https://archive.ics.uci.edu/static/public/341/"
    "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
)
SHA256 = "4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09"  # fetched 2026-10-30
RAW = Path(__file__).resolve().parents[1] / "transfer" / "har" / "data" / "raw"
CURL = ["curl", "-sSL", "--proto", "=https", "--retry", "8", "--retry-delay", "5"]
KEEP = re.compile(r"README\.txt|RawData/labels\.txt|RawData/acc_exp\d{2}_user\d{2}\.txt")


def main() -> None:
    """Fetch the zip unless verified, check its SHA-256, extract the kept members safely."""
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "hapt.zip"
    if not zpath.exists():
        part = zpath.with_suffix(".part")
        subprocess.run([*CURL, "-C", "-", "-o", str(part), URL], check=True)
        part.rename(zpath)
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if digest != SHA256:
        zpath.unlink()  # wrong bytes cannot be fixed by resuming: start over on the next run
        raise typer.BadParameter(f"hapt.zip SHA-256 {digest} != pinned {SHA256}; re-run")
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename:
                raise typer.BadParameter(f"unsafe member in hapt.zip: {info.filename!r}")
            if KEEP.fullmatch(info.filename):
                z.extract(info, RAW)
                n += 1
    typer.echo(f"sha256 {digest}; extracted {n} files into {RAW}")


if __name__ == "__main__":
    typer.run(main)
