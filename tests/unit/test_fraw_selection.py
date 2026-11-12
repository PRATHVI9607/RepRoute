"""Per-fold f_raw choice (ADR 0028 §7): rules, determinism, and which windows are touched."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from kai.tasks.pipeline import Candidate, FoldConfig, choose_f_raw, select_f_raw

NAMES = ("wave", "w2v")


def _data(n_groups: int = 20, per: int = 15, k: int = 3):
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(n_groups), per)
    y = rng.integers(0, k, len(g))
    return y, g, rng


def _pred(y, acc, rng, k=3):
    wrong = rng.random(len(y)) > acc
    return np.where(wrong, (y + 1 + rng.integers(0, k - 1, len(y))) % k, y)


def test_bootstrap_keeps_incumbent_when_equal():
    y, g, rng = _data()
    p = _pred(y, 0.7, rng)
    pick, rec = select_f_raw(
        "ptrain_bootstrap", NAMES, y, g, {"wave": p, "w2v": p}, {"wave": 1, "w2v": 9}, 0
    )
    assert pick == "wave"
    assert rec["boot_lb/w2v"] == 0.0


def test_bootstrap_switches_on_clear_win_and_is_deterministic():
    y, g, rng = _data()
    preds = {"wave": _pred(y, 0.5, rng), "w2v": _pred(y, 0.95, rng)}
    args = ("ptrain_bootstrap", NAMES, y, g, preds, {"wave": 1, "w2v": 9}, 3)
    a, b = select_f_raw(*args), select_f_raw(*args)
    assert a == b
    assert a[0] == "w2v"
    assert a[1]["boot_lb/w2v"] > 0
    assert select_f_raw(*args[:-1], 4)[1]["boot_lb/w2v"] != a[1]["boot_lb/w2v"]


def test_bootstrap_small_win_not_enough():
    y, g, rng = _data(n_groups=4, per=10)
    preds = {"wave": _pred(y, 0.6, rng), "w2v": _pred(y, 0.62, rng)}
    pick, rec = select_f_raw("ptrain_bootstrap", NAMES, y, g, preds, {"wave": 1, "w2v": 9}, 0)
    assert pick == "wave"
    assert rec["boot_lb/w2v"] <= 0


def test_best_rule_and_tie_goes_to_smaller_model():
    y, g, rng = _data()
    hi, lo = _pred(y, 0.9, rng), _pred(y, 0.5, rng)
    assert (
        select_f_raw("ptrain_best", NAMES, y, g, {"wave": lo, "w2v": hi}, {"wave": 1, "w2v": 9}, 0)[
            0
        ]
        == "w2v"
    )
    same = {"wave": hi, "w2v": hi}
    assert select_f_raw("ptrain_best", NAMES, y, g, same, {"wave": 9, "w2v": 1}, 0)[0] == "w2v"
    assert select_f_raw("ptrain_best", NAMES, y, g, same, {"wave": 1, "w2v": 9}, 0)[0] == "wave"


class _Fixed(nn.Module):
    def __init__(self, cls: int) -> None:
        super().__init__()
        self.cls = cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(len(x), 3)
        out[:, self.cls] = 1.0
        return out


def test_choose_touches_only_train_and_ptrain_arrays():
    seen: list[np.ndarray] = []

    def build(name, x, y, g, cfg, device):
        seen.append(x)
        return Candidate(name, _Fixed(0 if name == "wave" else 1), b"x" * (1 + (name == "w2v")))

    x_tr = np.full((6, 4), 1, np.int16)  # train windows are all 1, ptrain all 2
    x_pt = np.full((6, 4), 2, np.int16)
    y_pt = np.array([1, 1, 1, 1, 1, 0])
    g_pt = np.array([0, 0, 1, 1, 2, 2])
    cfg = FoldConfig(f_raw="wave", f_raw_candidates=NAMES, f_raw_rule="ptrain_best", device="cpu")
    cand, rec = choose_f_raw(
        cfg, (x_tr, np.zeros(6, int), np.arange(6)), (x_pt, y_pt, g_pt), torch.device("cpu"), build
    )
    assert cand.name == "w2v"
    assert rec["ptrain_f1/w2v"] > rec["ptrain_f1/wave"]
    assert all((x == 1).all() for x in seen)
    assert len(seen) == 2


def test_recipe_records_choice_and_old_recipe_unchanged():
    old = FoldConfig(f_raw="wave").recipe()
    assert "f_raw_candidates" not in old
    assert "f_raw_rule" not in old
    new = FoldConfig(f_raw="wave", f_raw_candidates=NAMES, f_raw_rule="ptrain_bootstrap").recipe()
    assert new["f_raw_candidates"] == ["wave", "w2v"]
    assert new["f_raw_rule"] == "ptrain_bootstrap"
    with pytest.raises(ValueError, match="f_raw"):
        FoldConfig(f_raw="spec", f_raw_candidates=NAMES, f_raw_rule="ptrain_best")
    with pytest.raises(ValueError, match="f_raw"):
        FoldConfig(f_raw="wave", f_raw_candidates=NAMES)
