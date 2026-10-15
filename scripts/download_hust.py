"""Download the HUST bearing recordings into transfer/hust/data/raw. Resumable, SHA-256-verified.

Source: https://data.mendeley.com/datasets/cbv7jyx4p9/3 (Hong & Thuan, BMC Res Notes 16, 2023,
doi:10.17632/cbv7jyx4p9.3). Licence: Creative Commons Attribution 4.0 International (CC BY 4.0);
cite the dataset and never commit the data. Only the normal / inner / outer files are fetched.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from kai.data.hust import RECORDINGS

HOST = "https://data.mendeley.com/"
LISTING = (
    f"{HOST}public-api/datasets/cbv7jyx4p9/files"
    "?folder_id=72793f17-cac2-4cc6-ac18-2b047e0d45f8&version=3"
)
RAW = Path(__file__).resolve().parents[1] / "transfer" / "hust" / "data" / "raw"
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
    """Fetch every recording of the 3-class task (the ball / combined-defect files are not used)."""
    RAW.mkdir(parents=True, exist_ok=True)
    # the site rejects urllib's user agent, so the listing is fetched with curl as well
    listing = json.loads(subprocess.run([*CURL, LISTING], check=True, capture_output=True).stdout)
    remote = {f["filename"]: f["content_details"] for f in listing}
    # file names come from our catalogue, never from the listing; URLs must stay on the data host
    items = [
        (n, remote[n]["download_url"], int(remote[n]["size"]), remote[n]["sha256_hash"])
        for n in RECORDINGS
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
