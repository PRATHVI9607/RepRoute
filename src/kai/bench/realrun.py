"""Real-socket backend: a cloud server subprocess plus the edge runtime in this process.

Used by E4 (measured phase boundaries) and E7 (sim-to-real). The uplink rate is enforced by the
edge's token-bucket pacer; the Docker + `tc` backend in `docker/` shapes in the kernel instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from kai.contracts.representation import REP_CODES
from kai.contracts.schemas import EnergyModel, Prices
from kai.edge.bundle import Bundle
from kai.edge.runtime import EdgeConfig, run_edge
from kai.profile.profiler import co_tenants
from kai.select.base import Selector

logger = logging.getLogger(__name__)


def parse_telemetry_jsonl(path: Path) -> dict[tuple[int, str], dict]:
    """Parse cloud telemetry JSONL file.

    Returns a mapping keyed by (window_id, rep_id).
    Raises ValueError on malformed JSONL or duplicate correlation keys.
    """
    events: dict[tuple[int, str], dict] = {}
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line_str = raw_line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
            except Exception as e:
                raise ValueError(f"Malformed JSONL line {line_num} in {path}: {e}") from e
            if not isinstance(rec, dict):
                raise ValueError(f"Malformed JSONL line {line_num} in {path}: expected dict")
            for field in ("window_id", "rep_id", "decision"):
                if field not in rec:
                    raise ValueError(
                        f"Malformed JSONL line {line_num} in {path}: missing field '{field}'"
                    )
            key = (int(rec["window_id"]), str(rec["rep_id"]))
            if key in events:
                msg = (
                    f"Duplicate telemetry event for (window_id={key[0]}, rep_id='{key[1]}') "
                    f"at line {line_num}"
                )
                raise ValueError(msg)
            events[key] = rec
    return events


def correlate_telemetry(
    records: list[dict],
    telemetry_events: dict[tuple[int, str], dict],
    code_to_rep_id: dict[int, str],
) -> list[dict]:
    """Correlate telemetry events into edge records in place by (window_id, rep_id)."""
    for rec in records:
        wid = rec.get("window_id")
        rep_code = rec.get("rep_code")
        if wid is None or rep_code is None or rep_code not in code_to_rep_id:
            continue
        rep_id = code_to_rep_id[rep_code]
        ev = telemetry_events.get((wid, rep_id))
        if ev is not None:
            if "measured_cloud_dec_ms" in ev:
                rec["measured_cloud_dec_ms"] = ev["measured_cloud_dec_ms"]
            if "measured_cloud_head_ms" in ev:
                rec["measured_cloud_head_ms"] = ev["measured_cloud_head_ms"]
    return records


@contextlib.contextmanager
def cloud_process(
    bundle_dir: Path,
    telemetry_out: Path | None = None,
) -> Iterator[int]:
    """Start `kai.cloud.server` on a free local port; yield the port once it is listening."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [
        sys.executable,
        "-m",
        "kai.cloud.server",
        str(bundle_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if telemetry_out is not None:
        cmd.extend(["--telemetry-out", str(telemetry_out)])
    proc = subprocess.Popen(cmd)
    try:
        for _ in range(200):
            if proc.poll() is not None:
                raise RuntimeError(f"cloud server exited with code {proc.returncode}")
            with (
                contextlib.suppress(OSError),
                socket.create_connection(("127.0.0.1", port), timeout=0.2),
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("cloud server did not start listening")
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def real_run(
    bundle: Bundle,
    port: int,
    cfg: EdgeConfig,
    prices: Prices,
    energy: EnergyModel,
    windows: np.ndarray,
    window_ids: np.ndarray,
    selector: Selector | None = None,
    tenants: int = 0,
    cloud_telemetry: Path | None = None,
) -> list[dict]:
    """Run the edge against the cloud at `port`. `tenants` busy co-tenants share the edge's core."""
    with co_tenants(tenants):
        records = asyncio.run(
            run_edge(
                bundle,
                cfg,
                prices,
                energy,
                windows,
                window_ids,
                "127.0.0.1",
                port,
                selector=selector,
            )
        )
    if cloud_telemetry is not None:
        code_to_rep_id = {
            REP_CODES[r.rep_id]: r.rep_id for r in bundle.reps if r.rep_id in REP_CODES
        }
        events = parse_telemetry_jsonl(cloud_telemetry)
        correlate_telemetry(records, events, code_to_rep_id)
    return records
