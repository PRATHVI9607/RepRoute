"""Leakage-safe splitters (PRD §8.3). Only bearing-wise splits exist; there is no L0 splitter."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from kai.contracts.schemas import PARTITIONS, Partition, SplitFold, SplitManifest
from kai.contracts.windows import WindowSet


def _allocate(ids: list[str], f_calib: float, f_ptrain: float) -> dict[Partition, list[str]]:
    """Split the non-test bearings of one class into calib / ptrain / train (each ≥ 1)."""
    m = len(ids)
    if m < 3:
        raise ValueError(f"need ≥ 3 non-test bearings per class, got {m}: {ids}")
    n_calib = max(1, int(m * f_calib))  # round down: scarce bearings go to training first
    n_ptrain = max(1, int(m * f_ptrain))
    while n_calib + n_ptrain > m - 1:  # always leave a training bearing
        if n_calib >= n_ptrain:
            n_calib -= 1
        else:
            n_ptrain -= 1
    return {
        "calib": ids[:n_calib],
        "ptrain": ids[n_calib : n_calib + n_ptrain],
        "train": ids[n_calib + n_ptrain :],
    }


def make_l1_splits(
    bearing_labels: dict[str, int],
    dataset_id: str,
    seed: int,
    n_folds: int = 5,
    n_repeats: int = 20,
    f_calib: float = 0.25,
    f_ptrain: float = 0.25,
) -> SplitManifest:
    """Leave-bearings-out, stratified by class.

    Within a repeat every bearing is a test bearing in exactly one fold; the rest are divided into
    train / ptrain (sufficiency-predictor training) / calib so that no bearing is shared.
    """
    by_class: dict[int, list[str]] = defaultdict(list)
    for b in sorted(bearing_labels):
        by_class[bearing_labels[b]].append(b)
    smallest = min(len(v) for v in by_class.values())
    if smallest < n_folds:
        raise ValueError(f"smallest class has {smallest} bearings, cannot make {n_folds} folds")

    folds: list[SplitFold] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng([seed, repeat])
        shuffled = {c: [ids[i] for i in rng.permutation(len(ids))] for c, ids in by_class.items()}
        for fold in range(n_folds):
            parts: dict[Partition, list[str]] = {p: [] for p in PARTITIONS}
            for c in sorted(shuffled):
                ids = shuffled[c]
                test = ids[fold::n_folds]
                rest = [b for b in ids if b not in test]
                # rotate so calib/ptrain membership also varies across folds
                k = fold % len(rest)
                parts["test"] += test
                for part, chosen in _allocate(rest[k:] + rest[:k], f_calib, f_ptrain).items():
                    parts[part] += chosen
            folds.append(
                SplitFold(
                    repeat=repeat,
                    fold=fold,
                    train=sorted(parts["train"]),
                    ptrain=sorted(parts["ptrain"]),
                    calib=sorted(parts["calib"]),
                    test=sorted(parts["test"]),
                )
            )
    return SplitManifest(dataset_id=dataset_id, level="L1", seed=seed, folds=folds)


def make_l2_splits(l1: SplitManifest, conditions: list[str]) -> SplitManifest:
    """Leave-operating-condition-out on top of L1: bearing-disjoint *and* condition-disjoint."""
    folds = [
        f.model_copy(update={"heldout_condition": conditions[i % len(conditions)]})
        for i, f in enumerate(l1.folds)
    ]
    return SplitManifest(dataset_id=l1.dataset_id, level="L2", seed=l1.seed, folds=folds)


def partition_mask(ws: WindowSet, fold: SplitFold, partition: Partition) -> npt.NDArray[np.bool_]:
    mask = np.isin(ws.bearing_id, fold.bearings(partition))
    if fold.heldout_condition is not None:
        is_heldout = ws.condition == fold.heldout_condition
        mask &= is_heldout if partition == "test" else ~is_heldout
    return mask


def partition_of(ws: WindowSet, fold: SplitFold, partition: Partition) -> WindowSet:
    return ws.take(partition_mask(ws, fold, partition))


def assert_leakage_safe(ws: WindowSet, fold: SplitFold) -> None:
    """Raise if any bearing or window is shared between two partitions."""
    masks = {p: partition_mask(ws, fold, p) for p in PARTITIONS}
    for i, p in enumerate(PARTITIONS):
        if not masks[p].any():
            raise ValueError(f"partition {p} has no windows")
        for q in PARTITIONS[i + 1 :]:
            if (masks[p] & masks[q]).any():
                raise ValueError(f"windows shared between {p} and {q}")
            shared = set(ws.bearing_id[masks[p]]) & set(ws.bearing_id[masks[q]])
            if shared:
                raise ValueError(f"bearings {sorted(shared)} shared between {p} and {q}")


def save_manifest(manifest: SplitManifest, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{manifest.dataset_id}_{manifest.level}_{manifest.split_id}.json"
    path.write_text(manifest.model_dump_json(indent=1))
    return path


def load_manifest(path: Path) -> SplitManifest:
    return SplitManifest.model_validate_json(path.read_text())
