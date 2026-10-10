"""Real-socket backend: a cloud server subprocess plus the edge runtime in this process.

Used by E4 (measured phase boundaries) and E7 (sim-to-real). The uplink rate is enforced by the
edge's token-bucket pacer; the Docker + `tc` backend in `docker/` shapes in the kernel instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from kai.contracts.schemas import EnergyModel, Prices
from kai.edge.bundle import Bundle
from kai.edge.runtime import EdgeConfig, run_edge
from kai.profile.profiler import co_tenants
from kai.select.base import Selector


@contextlib.contextmanager
def cloud_process(bundle_dir: Path) -> Iterator[int]:
    """Start `kai.cloud.server` on a free local port; yield the port once it is listening."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kai.cloud.server",
            str(bundle_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    )
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
        proc.wait(timeout=10)


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
) -> list[dict]:
    """Run the edge against the cloud at `port`. `tenants` busy co-tenants share the edge's core."""
    with co_tenants(tenants):
        return asyncio.run(
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
