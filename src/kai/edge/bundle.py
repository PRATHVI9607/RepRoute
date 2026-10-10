"""Load and verify a calibration bundle (PRD §5.4). numpy / json only — edge-safe.

The edge refuses to start when any file hash, the bundle id, a contract hash or (if a lockfile is
supplied) the environment lock hash does not match the manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kai.certify.predictor import Predictor
from kai.contracts.schemas import BundleManifest, EnergyModel, LambdaTable, Prices, RepProfile
from kai.reps.base import BaseRep
from kai.reps.registry import load_menu
from kai.select.base import Mission
from kai.select.cost import Overheads


class BundleError(RuntimeError):
    """The bundle on disk is not the bundle that was calibrated."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_id_of(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Bundle:
    path: Path
    manifest: BundleManifest
    reps: list[BaseRep]
    predictor: Predictor
    lambdas: LambdaTable
    profiles: list[RepProfile]
    overheads: Overheads
    prices: Prices  # deployment preset the bundle was built for
    energy: EnergyModel
    mission: Mission
    calib_m: np.ndarray  # V0 first-loss thresholds of the calibration windows (for ACI)
    phi_ref: np.ndarray  # probe features of calibration windows (drift reference)


def load_bundle(path: Path, lock_file: Path | None = None) -> Bundle:
    manifest = BundleManifest.model_validate_json((path / "manifest.json").read_text())
    for rel, digest in manifest.files.items():
        if not (path / rel).is_file():
            raise BundleError(f"bundle file missing: {rel}")
        if sha256_file(path / rel) != digest:
            raise BundleError(f"hash mismatch for {rel}: the bundle was modified after calibration")
    if bundle_id_of(manifest.files) != manifest.bundle_id:
        raise BundleError("bundle id does not match its file list")
    if lock_file is not None and sha256_file(lock_file) != manifest.lock_hash:
        raise BundleError("environment lockfile differs from the one the bundle was built with")
    reps = load_menu(path / "menu")  # raises ContractMismatchError on a codec / encoder change
    hashes = {r.rep_id: r.contract_hash() for r in reps}
    if hashes != manifest.contract_hashes or [r.rep_id for r in reps] != manifest.rep_ids:
        raise BundleError("representation menu does not match the manifest")
    prof = json.loads((path / "profiles.json").read_text())
    return Bundle(
        path=path,
        manifest=manifest,
        reps=reps,
        predictor=Predictor.load(path / "predictor.npz"),
        lambdas=LambdaTable.model_validate_json((path / "lambda.json").read_text()),
        profiles=[RepProfile(**p) for p in prof["profiles"]],
        overheads=Overheads(**prof["overheads"]),
        prices=Prices(**prof["prices"]),
        energy=EnergyModel(**prof["energy"]),
        mission=prof["mission"],
        calib_m=np.load(path / "calib_m.npy"),
        phi_ref=np.load(path / "phi_ref.npy"),
    )
