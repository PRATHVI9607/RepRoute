"""Build an immutable calibration bundle from one trained fold (PRD §5.4). Lab plane."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kai.bench.calibrate import ALPHAS, calibrate, score
from kai.certify import crc
from kai.contracts.schemas import BundleManifest, EnergyModel, Prices, RepProfile
from kai.edge.bundle import bundle_id_of, sha256_file
from kai.reps.registry import load_menu
from kai.select.base import Mission
from kai.select.cost import Overheads
from kai.tasks.pipeline import Table
from kai.tasks.train import ONNX_OPSET

ROOT = Path(__file__).resolve().parents[3]


def git_state(root: Path = ROOT) -> tuple[str, bool]:
    """(HEAD sha, dirty). A repository without commits reports ("uncommitted", True)."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return "uncommitted", True
    return head.stdout.strip(), bool(git("status", "--porcelain").stdout.strip())


def lock_hash(root: Path = ROOT) -> str:
    lock = root / "uv.lock"
    return sha256_file(lock) if lock.exists() else "no-lockfile"


@dataclass(frozen=True)
class BundleMeta:
    dataset_id: str
    split_id: str
    repeat: int
    fold: int
    seed: int
    sample_rate_hz: float

    @staticmethod
    def from_fold(fold_dir: Path, dataset_id: str, sample_rate_hz: float) -> BundleMeta:
        """Read split id, seed, repeat and fold from `<split_id>/s<seed>_r<repeat>f<fold>`."""
        seed, rest = fold_dir.name.split("_")
        repeat, fold = rest[1:].split("f")
        return BundleMeta(
            dataset_id, fold_dir.parent.name, int(repeat), int(fold), int(seed[1:]), sample_rate_hz
        )


def build_bundle(
    fold_dir: Path,
    out_root: Path,
    profiles: list[RepProfile],
    overheads: Overheads,
    meta: BundleMeta,
    prices: Prices,
    energy: EnergyModel,
    mission: Mission = "fidelity_first",
    cost_cal: np.ndarray | None = None,
    alphas: tuple[float, ...] = ALPHAS,
    delta: float = 0.1,
) -> Path:
    """Fit the predictor on ptrain, calibrate on calib, and freeze everything with hashes."""
    table = Table.load(fold_dir / "table.npz")
    scored = score(table, seed=meta.seed)
    ghat_cal, d_cal = scored.part("calib")
    tmp = out_root / "_building"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(fold_dir / "menu", tmp / "menu")
    shutil.copytree(fold_dir / "heads", tmp / "heads")
    shutil.copy(fold_dir / "f_raw.onnx", tmp / "f_raw.onnx")
    scored.predictor.save(tmp / "predictor.npz")
    lambdas = calibrate(ghat_cal, d_cal, table.rep_ids, alphas, delta, cost_cal)
    (tmp / "lambda.json").write_text(lambdas.model_dump_json(indent=1))
    by_id = {p.rep_id: p for p in profiles}
    ordered = [by_id[r] for r in table.rep_ids]
    (tmp / "profiles.json").write_text(
        json.dumps(
            {
                "profiles": [p.model_dump() for p in ordered],
                "overheads": asdict(overheads),
                # the deployment preset travels with the bundle: the edge has no config system
                "prices": prices.model_dump(),
                "energy": energy.model_dump(),
                "mission": mission,
            },
            indent=1,
        )
    )
    pd.DataFrame([p.model_dump() for p in ordered]).to_parquet(
        tmp / "profiles.parquet", index=False
    )
    np.save(tmp / "calib_m.npy", crc.first_loss_threshold(ghat_cal, d_cal))
    phi_cal = table.phi[table.of("calib")]
    np.save(tmp / "phi_ref.npy", phi_cal[:: max(1, len(phi_cal) // 2000)])

    reps = load_menu(tmp / "menu")
    files = {str(p.relative_to(tmp)): sha256_file(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
    sha, dirty = git_state()
    manifest = BundleManifest(
        bundle_id=bundle_id_of(files),
        created_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        git_sha=sha,
        git_dirty=dirty,
        lock_hash=lock_hash(),
        dataset_id=meta.dataset_id,
        split_id=meta.split_id,
        repeat=meta.repeat,
        fold=meta.fold,
        seed=meta.seed,
        window_n=reps[0].window_n,
        sample_rate_hz=meta.sample_rate_hz,
        onnx_opset=ONNX_OPSET,
        rep_ids=list(table.rep_ids),
        contract_hashes={r.rep_id: r.contract_hash() for r in reps},
        files=files,
    )
    (tmp / "manifest.json").write_text(manifest.model_dump_json(indent=1))
    final = out_root / manifest.bundle_id
    if final.exists():  # same content → same id: bundles are immutable, nothing to do
        shutil.rmtree(tmp)
    else:
        tmp.rename(final)
    return final
