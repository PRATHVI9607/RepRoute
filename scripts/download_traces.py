"""Download the UCC 4G LTE dataset and write its throughput traces as one-column CSVs.

Source: Raca, Quinlan, Zahran, Sreenan, "Beyond Throughput: a 4G LTE Dataset with Channel and
Context Metrics", ACM MMSys 2018 — https://zenodo.org/records/1219679, Creative Commons
Attribution 4.0 International. Measured with G-NetTrack Pro, one sample per second.

    uv run --no-sync python scripts/download_traces.py

Writes `data/raw/traces/ucc4g/<pattern>__<file>.csv` (bits per second, one sample per line) and
`index.json`, where pattern is the mobility class (bus, car, pedestrian, static, train).

What the numbers are: the logs were recorded while the phone downloaded continuously, so
`DL_bitrate` is the throughput the downlink delivered. `UL_bitrate` in the same logs is about 2 %
of it — acknowledgements, not uplink capacity — and is not used. The simulator takes these traces
for their *shape over time* and rescales the median to each scenario's uplink rate; no measured
uplink capacity is claimed.

The archive is verified against the checksum Zenodo publishes; it is data, never code: members
are read from the zip in memory and only one numeric column is parsed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import typer

RECORD = "https://zenodo.org/api/records/1219679"
ARCHIVE = "LTE_Dataset.zip"  # the measured logs (the other archive is an ns-3 replay)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "traces" / "ucc4g"
COLUMN = "DL_bitrate"  # kbit/s in the G-NetTrack export
MIN_SAMPLES = 120  # shorter logs cannot fill a simulated run


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "kai-research/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return bytes(r.read())


def main() -> None:
    """Fetch, verify and convert. Safe to re-run."""
    record = json.loads(_get(RECORD))
    (entry,) = [f for f in record["files"] if f["key"] == ARCHIVE]
    blob = _get(entry["links"]["self"])
    algo, _, want = entry["checksum"].partition(":")
    if hashlib.new(algo, blob).hexdigest() != want:
        raise typer.BadParameter(f"{entry['key']}: checksum mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in sorted(n for n in archive.namelist() if n.lower().endswith(".csv")):
            rows = list(csv.DictReader(io.TextIOWrapper(archive.open(name), "utf-8", "replace")))
            if not rows or COLUMN not in rows[0]:
                continue
            kbps = np.array([float(r[COLUMN]) if r[COLUMN].strip("-") else 0.0 for r in rows])
            if len(kbps) < MIN_SAMPLES or np.median(kbps) <= 0:
                continue  # too short, or an idle log
            parts = Path(name).parts
            stem = "__".join(p.replace(" ", "_") for p in (*parts[-2:-1], Path(parts[-1]).stem))
            np.savetxt(OUT / f"{stem}.csv", kbps * 1000.0, fmt="%.0f")
            index.append(
                {
                    "file": f"{stem}.csv", "source_member": name, "n_samples": len(kbps),
                    "median_bps": float(np.median(kbps) * 1000.0),
                    "zero_share": float(np.mean(kbps == 0)),
                }
            )  # fmt: skip
    meta = {
        "title": record["metadata"]["title"],
        "doi": record["metadata"].get("doi"),
        "licence": record["metadata"].get("license", {}).get("id"),
        "archive": entry["key"],
        "checksum": entry["checksum"],
        "step_s": 1.0,
        "column": COLUMN,
        "traces": index,
    }
    (OUT / "index.json").write_text(json.dumps(meta, indent=1) + "\n")
    typer.echo(f"{len(index)} throughput traces in {OUT} ({meta['licence']})")


if __name__ == "__main__":
    typer.run(main)
