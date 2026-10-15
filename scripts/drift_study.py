"""Can a probe-feature monitor tell a new operating condition from a new bearing? (ADR 0008)

Offline study on the trained L1 / L2 folds; writes `experiments/drift_study/summary.json`.

    uv run --no-sync python scripts/drift_study.py

Three designs, each scored by how often its alarm is up:

* `ks_pooled`     — the shipped monitor: per-feature KS test against the calibration bearings.
* `hull_pooled`   — share of test windows outside the calibration bearings' feature range.
* `envelope_own`  — per machine: reference = the first half of that test bearing's windows in the
                    conditions it was calibrated for; alarm = too many recent windows outside the
                    reference's [q, 1 - q] range (binomial test, Bonferroni over features).
                    False alarms are measured on the second half of the same conditions, detection
                    on the held-out condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from kai.bench.config import load_config, work
from kai.bench.experiments import fold_dirs
from kai.certify.drift import DriftMonitor
from kai.contracts.windows import WindowSet
from kai.data.splits import load_manifest
from kai.tasks.pipeline import Table

OUT = Path(__file__).resolve().parents[1] / "experiments/drift_study/summary.json"


def envelope_alarm(ref: np.ndarray, stream: np.ndarray, q: float, window: int) -> float:
    """Share of the stream (after the first full window) during which the alarm is up."""
    lo, hi = np.quantile(ref, [q, 1 - q], axis=0)
    outside = (stream < lo) | (stream > hi)
    up, flags = False, []
    for i in range(window - 1, len(stream)):
        if (i + 1) % 25 == 0:
            k = outside[i + 1 - window : i + 1].sum(axis=0)
            up = bool(stats.binom.sf(k - 1, window, 2 * q).min() < 0.01 / ref.shape[1])
        flags.append(up)
    return float(np.mean(flags)) if flags else float("nan")


def main() -> None:
    cfg = load_config()
    ws = WindowSet.load(work() / cfg.dataset.windows)
    out: dict[str, dict[str, float]] = {}
    for level in ("L1", "L2"):
        ks, hull = [], []
        for fold in fold_dirs(cfg, level)[:5]:
            t = Table.load(fold / "table.npz")
            ref, test = (t.phi[t.of(p)].astype(np.float64) for p in ("calib", "test"))
            monitor = DriftMonitor(ref)
            ks.append(float(np.mean([monitor.update(p) for p in test])))
            hull.append(float(((test < ref.min(0)) | (test > ref.max(0))).any(axis=1).mean()))
        out[level] = {"ks_pooled": float(np.mean(ks)), "hull_pooled": float(np.mean(hull))}
    (path,) = sorted((work() / cfg.dataset.splits_dir).glob(f"{cfg.dataset.id}_L2_*.json"))
    manifest = {f"s{cfg.seed}_r{f.repeat}f{f.fold}": f for f in load_manifest(path).folds}
    for q, window in ((0.005, 100), (0.02, 100), (0.02, 50), (0.05, 100)):
        false_alarm, detection = [], []
        for fold_dir in fold_dirs(cfg, "L2")[:5]:
            fold = manifest[fold_dir.name]
            phi = Table.load(fold_dir / "table.npz").phi.astype(np.float64)
            for bearing in fold.test:
                own = ws.bearing_id == bearing
                new = own & (ws.condition == fold.heldout_condition)
                ref_idx: list[int] = []
                later_idx: list[int] = []
                for cond in np.unique(ws.condition[own & ~new]):
                    idx = np.flatnonzero(own & (ws.condition == cond))  # recording order
                    ref_idx += idx[: len(idx) // 2].tolist()
                    later_idx += idx[len(idx) // 2 :].tolist()
                false_alarm.append(envelope_alarm(phi[ref_idx], phi[later_idx], q, window))
                detection.append(envelope_alarm(phi[ref_idx], phi[np.flatnonzero(new)], q, window))
        out[f"envelope_own/q{q:g}_w{window}"] = {
            "false_alarm": float(np.nanmean(false_alarm)),
            "detection": float(np.nanmean(detection)),
            "n_machines": len(false_alarm),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
