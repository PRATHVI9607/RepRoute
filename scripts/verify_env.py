"""Environment check (PRD §16.5). Prints one line per check and exits non-zero unless ALL GREEN."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDGE = ["numpy", "onnxruntime", "pywt", "zstandard", "psutil", "msgpack", "pydantic"]
LAB = [
    "torch",
    "onnx",
    "sklearn",
    "lightgbm",
    "scipy",
    "pandas",
    "pyarrow",
    "statsmodels",
    "optuna",
    "hydra",
    "simpy",
    "matplotlib",
]
FORBIDDEN_ON_EDGE = ["torch", "sklearn", "pandas", "scipy", "lightgbm"]
results: list[tuple[bool, str]] = []


def check(ok: bool, message: str) -> None:
    results.append((ok, message))
    print(("  ok    " if ok else "  FAIL  ") + message)


def locked_versions() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for pkg in tomllib.loads((ROOT / "uv.lock").read_text())["package"]:
        out.setdefault(pkg["name"].lower().replace("_", "-"), set()).add(pkg["version"])
    return out


def main(edge_only: bool) -> int:
    check(sys.version_info[:2] == (3, 12), f"Python {sys.version.split()[0]} (need 3.12)")
    locked = locked_versions()
    drift = []
    for dist in md.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name in locked and dist.version not in locked[name]:
            drift.append(f"{name} {dist.version} (lock: {sorted(locked[name])})")
    check(not drift, "installed versions match uv.lock" + (f": {drift[:5]}" if drift else ""))
    for mod in EDGE:
        check(importlib.util.find_spec(mod) is not None, f"edge import: {mod}")

    code = (
        "import sys, kai.edge.runtime, kai.contracts.frame; "
        f"sys.exit(any(m in sys.modules for m in {FORBIDDEN_ON_EDGE!r}))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)
    check(
        proc.returncode == 0, "kai.edge imports without torch / sklearn / pandas / scipy / lightgbm"
    )

    if not edge_only:
        for mod in LAB:
            check(importlib.util.find_spec(mod) is not None, f"lab import: {mod}")
        import numpy as np
        import onnxruntime as ort
        import torch

        from kai.tasks.models import LatentEncoder
        from kai.tasks.train import export_onnx

        torch.manual_seed(0)
        model = LatentEncoder(1024, 8).eval()
        x = np.random.default_rng(0).normal(size=(4, 1, 1024)).astype(np.float32)
        ref = model(torch.from_numpy(x)).detach().numpy()
        got = ort.InferenceSession(
            export_onnx(model, 1024), providers=["CPUExecutionProvider"]
        ).run(None, {"x": x})[0]
        check(float(np.abs(ref - got).max()) <= 1e-4, "torch → ONNX → onnxruntime parity ≤ 1e-4")
        check(
            True, f"torch build: {torch.__version__} (cuda available: {torch.cuda.is_available()})"
        )
        a = np.random.default_rng(123).random(5)
        check(bool((a == np.random.default_rng(123).random(5)).all()), "deterministic seed test")

    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    governor = gov.read_text().strip() if gov.exists() else "unknown"
    check(
        True,
        f"cpu: {os.cpu_count()} logical cores, governor {governor} "
        "(use `performance` for timing runs)",
    )
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    check(free_gb > 10, f"free disk {free_gb:.0f} GB (need > 10)")
    check(
        shutil.which("unrar") is not None or shutil.which("7z") is not None,
        "archive tool (unrar or 7z) on PATH",
    )
    tc = (
        subprocess.run(["tc", "qdisc", "show", "dev", "lo"], capture_output=True, check=False)
        if shutil.which("tc")
        else None
    )
    check(
        True,
        "tc available"
        if tc and tc.returncode == 0
        else "tc not usable here — netem runs need the Docker backend with NET_ADMIN",
    )

    failed = [m for ok, m in results if not ok]
    print("\nALL GREEN" if not failed else f"\n{len(failed)} CHECK(S) FAILED")
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main("--edge-only" in sys.argv))
