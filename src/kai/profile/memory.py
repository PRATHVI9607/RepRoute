"""Peak resident memory of the edge-side pieces (ADR 0028 A), measured in a fresh process.

Each measurement runs in its own `python -c` child so that the parent's heap, torch and test
fixtures do not count. The child's `ru_maxrss` is a high-water mark, hence one child per kind;
the interpreter + numpy baseline is measured the same way and subtracted.
Linux only (VmHWM is in KiB).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# VmHWM, not ru_maxrss: a child's ru_maxrss survives exec and starts at the parent's peak, which
# under pytest or `kai exp` (torch loaded) would swamp both the child and the baseline.
_PEAK = (
    "print(next(int(l.split()[1]) * 1024 for l in open('/proc/self/status')"
    " if l.startswith('VmHWM')))"
)
_BASELINE = f"import numpy;{_PEAK}"

_F_RAW = f"""
import sys, numpy as np, onnxruntime as ort
sess = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
shape = [1 if not isinstance(d, int) else d for d in inp.shape]  # dynamic batch axis → 1
sess.run(None, {{inp.name: np.zeros(shape, np.float32)}})
{_PEAK}
"""

# probe + predictor + the classical codecs r1–r4 on one zero window. `load_menu` builds every entry
# of a menu, latent ONNX models included, so only the classical entries go into a scratch menu.
_KAI_CLASSICAL = f"""
import json, re, sys, tempfile, numpy as np
from pathlib import Path
from kai.certify.predictor import Predictor
from kai.certify.probe import probe
from kai.reps.registry import load_menu
fold = Path(sys.argv[1])
predictor = Predictor.load(max(fold.glob("predictor_mlp_*.npz"), key=lambda p: p.stat().st_mtime))
entries = [e for e in json.loads((fold / "menu" / "reps.json").read_text())
           if re.match(r"r[1-4]", e["rep_id"])]
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "reps.json").write_text(json.dumps(entries))
    reps = load_menu(Path(tmp))
x = np.zeros(reps[0].window_n, np.int16)
predictor.predict(probe(x))
for rep in reps:
    rep.encode(x)
{_PEAK}
"""


def _run(code: str, *args: str) -> int:
    out = subprocess.run(
        [sys.executable, "-c", code, *args], check=True, capture_output=True, text=True
    )
    return int(out.stdout.strip().splitlines()[-1])


def baseline_rss_bytes() -> int:
    """Peak RSS of a fresh interpreter that imports numpy only: the common baseline."""
    return _run(_BASELINE)


def peak_rss_bytes(kind: str, fold_dir: Path) -> int:
    """Peak RSS in bytes of `kind` ("f_raw" or "kai_classical") above the numpy-only baseline,
    measured in a fresh subprocess; f_raw includes onnxruntime's own footprint. Clamped at 0.
    Report `baseline_rss_bytes()` next to it."""
    if kind == "f_raw":
        code, arg = _F_RAW, str(fold_dir / "f_raw.onnx")
    elif kind == "kai_classical":
        code, arg = _KAI_CLASSICAL, str(fold_dir)
    else:
        raise ValueError(f"unknown kind {kind!r}; expected 'f_raw' or 'kai_classical'")
    return max(0, _run(code, arg) - baseline_rss_bytes())
