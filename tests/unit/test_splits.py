from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from kai.contracts.schemas import PARTITIONS, SplitFold, SplitManifest
from kai.contracts.windows import WindowSet
from kai.data.splits import (
    assert_leakage_safe,
    load_manifest,
    make_l1_splits,
    make_l2_splits,
    partition_mask,
    partition_of,
    save_manifest,
)


def test_no_bearing_in_two_partitions_and_every_class_everywhere(
    ws: WindowSet, splits: SplitManifest
) -> None:
    labels = ws.bearing_labels()
    for fold in splits.folds:
        assert_leakage_safe(ws, fold)
        all_ids = [b for p in PARTITIONS for b in fold.bearings(p)]
        assert sorted(all_ids) == sorted(labels)  # a partition of the bearings, no loss
        for p in PARTITIONS:
            assert {labels[b] for b in fold.bearings(p)} == {0, 1, 2}


def test_each_bearing_is_tested_exactly_once_per_repeat(splits: SplitManifest) -> None:
    for repeat in {f.repeat for f in splits.folds}:
        counts = Counter(b for f in splits.folds if f.repeat == repeat for b in f.test)
        assert set(counts.values()) == {1}


def test_windows_are_unique_and_non_overlapping(ws: WindowSet, splits: SplitManifest) -> None:
    fold = splits.folds[0]
    total = sum(int(partition_mask(ws, fold, p).sum()) for p in PARTITIONS)
    assert total == len(ws) == len(np.unique(ws.window_id))


def test_l2_is_bearing_and_condition_disjoint(ws: WindowSet, splits: SplitManifest) -> None:
    conditions = sorted(set(ws.condition.tolist()))
    l2 = make_l2_splits(splits, conditions)
    assert l2.level == "L2"
    assert l2.split_id != splits.split_id
    for fold in l2.folds:
        assert_leakage_safe(ws, fold)
        assert set(partition_of(ws, fold, "test").condition) == {fold.heldout_condition}
        for p in ("train", "ptrain", "calib"):
            assert fold.heldout_condition not in set(partition_of(ws, fold, p).condition)


def test_manifest_is_deterministic_and_round_trips(ws: WindowSet, tmp_path) -> None:
    a = make_l1_splits(ws.bearing_labels(), ws.dataset_id, seed=3, n_folds=3, n_repeats=2)
    b = make_l1_splits(ws.bearing_labels(), ws.dataset_id, seed=3, n_folds=3, n_repeats=2)
    c = make_l1_splits(ws.bearing_labels(), ws.dataset_id, seed=4, n_folds=3, n_repeats=2)
    assert a.split_id == b.split_id != c.split_id
    assert load_manifest(save_manifest(a, tmp_path)) == a


def test_schema_rejects_overlap_and_empty_partitions() -> None:
    with pytest.raises(ValidationError, match="both"):
        SplitFold(repeat=0, fold=0, train=["a"], ptrain=["b"], calib=["c"], test=["a"])
    with pytest.raises(ValidationError, match="empty"):
        SplitFold(repeat=0, fold=0, train=["a"], ptrain=[], calib=["c"], test=["d"])


def test_too_few_bearings_is_an_error() -> None:
    labels = {f"b{i}": i % 3 for i in range(6)}  # 2 per class
    with pytest.raises(ValueError, match="cannot make"):
        make_l1_splits(labels, "x", seed=0, n_folds=5, n_repeats=1)
    with pytest.raises(ValueError, match="≥ 3 non-test"):
        make_l1_splits({f"b{i}": i % 3 for i in range(9)}, "x", seed=0, n_folds=3, n_repeats=1)


def test_windowset_rejects_bearing_with_two_labels(ws: WindowSet) -> None:
    bad = WindowSet(
        ws.x[:2],
        ws.window_id[:2],
        np.array(["b", "b"]),
        ws.condition[:2],
        np.array([0, 1]),
        1.0,
        "x",
    )
    with pytest.raises(ValueError, match="more than one label"):
        bad.bearing_labels()
