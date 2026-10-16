"""Download the UORED-VAFCLS MAT files into transfer/uored/data/raw. Resumable, SHA-256-verified.

Source: https://data.mendeley.com/datasets/y2px5tg92h/5 (Sehri & Dumond, doi:10.17632/y2px5tg92h.5;
Sehri, Dumond & Bouchard, Data in Brief 49 (2023) 109327). Licence: Creative Commons Attribution
4.0 International (CC BY 4.0); cite the dataset and never commit the data. All 60 MAT files are
fetched (~303 MB) so that the unused ones are counted as dropped; the CSV / XLSX copies are not.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from kai.data.uored import ALL_FILES

HOST = "https://data.mendeley.com/"
# the five sub-folders (healthy, inner, outer, ball, cage) of "3_MatLab_Raw_Data_Files (.mat)"
FOLDERS = (
    "efa2641a-b57b-435a-95aa-d1864683c261",
    "408aeca8-7683-4e17-964b-265c69c50484",
    "b6553543-188f-4e46-9b98-186be4e670e0",
    "3345eb37-7aad-496c-8b4d-90928d406714",
    "8dbda6ab-1551-45a5-8ec3-30459cd7af98",
)
LISTING = f"{HOST}public-api/datasets/y2px5tg92h/files?version=5&folder_id="
RAW = Path(__file__).resolve().parents[1] / "transfer" / "uored" / "data" / "raw"
CURL = ["curl", "-sSL", "--proto", "=https", "--retry", "8", "--retry-delay", "5"]


def _complete(dst: Path, size: int, sha256: str) -> bool:
    if not dst.exists() or dst.stat().st_size != size:
        return False
    return hashlib.sha256(dst.read_bytes()).hexdigest() == sha256


def _fetch(item: tuple[str, str, int, str]) -> tuple[str, bool]:
    name, url, size, sha256 = item
    dst = RAW / name
    if _complete(dst, size, sha256):
        return name, True
    if dst.exists() and dst.stat().st_size >= size:  # full-length but wrong: resuming cannot fix
        dst.unlink()
    subprocess.run([*CURL, "-C", "-", "-o", str(dst), url], check=False)
    return name, _complete(dst, size, sha256)


def main(jobs: int = 4) -> None:
    """Fetch the 60 MAT recordings (20 bearings × healthy / developing fault / faulty)."""
    RAW.mkdir(parents=True, exist_ok=True)
    remote = {}
    for folder in FOLDERS:  # the site rejects urllib's user agent, so listings use curl as well
        out = subprocess.run([*CURL, LISTING + folder], check=True, capture_output=True).stdout
        remote |= {f["filename"]: f["content_details"] for f in json.loads(out)}
    # file names come from our catalogue, never from the listing; URLs must stay on the data host
    items = [
        (n, remote[n]["download_url"], int(remote[n]["size"]), remote[n]["sha256_hash"])
        for n in ALL_FILES
    ]
    if not all(url.startswith(HOST) for _, url, _, _ in items):
        raise typer.BadParameter(f"unexpected download host in the listing of {LISTING}")
    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(_fetch, items))
    bad = [n for n, ok in results if not ok]
    total_bytes = sum(size for _, _, size, _ in items)
    typer.echo(f"{len(results) - len(bad)}/{len(results)} files ({total_bytes} bytes) in {RAW}")
    if bad:
        typer.echo(f"INCOMPLETE: {bad} — re-run to resume")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
