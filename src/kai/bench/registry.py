"""Run registry (PRD §14.1): one Parquet row per run, with the provenance needed to reproduce it."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd

from kai.bench.bundle import git_state, lock_hash
from kai.bench.config import work
from kai.contracts.schemas import RunRecord, config_hash


def registry_path() -> Path:
    return work() / "experiments" / "registry.parquet"


def record(
    experiment: str,
    config: dict,
    seed: int,
    metrics: dict[str, float],
    params: dict[str, str | int | float | bool | None] | None = None,
    path: Path | None = None,
    final: bool = False,
) -> RunRecord:
    """Append one run. A `final` run from a dirty tree is refused (PRD §12.4)."""
    sha, dirty = git_state()
    if final and dirty:
        raise RuntimeError("final runs must come from a clean tree at a tagged commit")
    rec = RunRecord(
        run_id=uuid.uuid4().hex[:12],
        experiment=experiment,
        created_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        git_sha=sha,
        git_dirty=dirty,
        lock_hash=lock_hash(),
        config_hash=config_hash(config),
        seed=seed,
        params=params or {},
        metrics=metrics,
    )
    row = rec.model_dump()
    row["params"], row["metrics"] = (
        json.dumps(row["params"], sort_keys=True),
        json.dumps(row["metrics"], sort_keys=True),
    )
    path = path or registry_path()
    # shortcut: read-modify-write without a lock, fine for one person running experiments in
    # sequence; switch to one file per run if several processes ever record at once
    frame = pd.DataFrame([row])
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return rec


def content_hash(path: Path | None = None) -> str:
    """Hash of what was computed (experiment, config, seed, metrics), not of when or by whom."""
    frame = pd.read_parquet(path or registry_path())[
        ["experiment", "config_hash", "seed", "params", "metrics"]
    ]
    return hashlib.sha256(frame.to_json(orient="records").encode()).hexdigest()
