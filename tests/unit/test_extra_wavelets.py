"""Opt-in mid-size wavelets (ADR 0016): absent by default, smaller as eps grows, reloadable."""

from __future__ import annotations

import numpy as np

from kai.reps.registry import classical_menu, load_menu, save_menu
from kai.tasks.pipeline import FoldConfig


def test_extra_wavelets_are_opt_in_ordered_by_size_and_reload(
    rng: np.random.Generator, tmp_path
) -> None:
    n = 1024
    assert "r2c_wavelet" not in [r.rep_id for r in classical_menu(n)]
    menu = {r.rep_id: r for r in classical_menu(n, extra_wavelets=True)}
    x = (rng.standard_normal(n) * 4000).astype(np.int16)
    sizes = {k: len(menu[k].encode(x)) for k in ("r2b_wavelet", "r2c_wavelet", "r2d_wavelet")}
    assert sizes["r2d_wavelet"] < sizes["r2c_wavelet"] < sizes["r2b_wavelet"]
    for rep in menu.values():
        rep.set_size(np.array([len(rep.encode(x))]))
    save_menu(list(menu.values()), tmp_path)
    assert [r.rep_id for r in load_menu(tmp_path)] == list(menu)


def test_folds_trained_before_the_option_keep_their_recipe() -> None:
    assert "extra_wavelets" not in FoldConfig().recipe()
    assert FoldConfig(extra_wavelets=True).recipe()["extra_wavelets"] is True
