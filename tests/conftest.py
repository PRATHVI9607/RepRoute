from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

from kai.contracts.schemas import EnergyModel, Prices, SplitManifest
from kai.contracts.windows import WindowSet
from kai.data.splits import make_l1_splits
from kai.data.synthetic import make_synthetic

settings.register_profile(
    "kai", deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("kai")

WINDOW_N = 1024


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture(scope="session")
def ws() -> WindowSet:
    """Tiny synthetic dataset: 30 bearings × 4 conditions × 8 windows of 1024 samples."""
    return make_synthetic(np.random.default_rng(0), 10, 8, window_n=WINDOW_N)


@pytest.fixture(scope="session")
def splits(ws: WindowSet) -> SplitManifest:
    return make_l1_splits(ws.bearing_labels(), ws.dataset_id, seed=1, n_folds=3, n_repeats=2)


# --- a fully trained tiny fold and its bundle, shared by the integration / e2e tests ------------
# Tests that use these carry `@pytest.mark.xdist_group("fold")` so one worker builds them once.


@pytest.fixture(scope="session")
def fold_dir(ws: WindowSet, splits: SplitManifest, tmp_path_factory: pytest.TempPathFactory):
    from kai.tasks.pipeline import FoldConfig, run_fold

    out = tmp_path_factory.mktemp("fold")
    cfg = FoldConfig(epochs_latent=12, head_trials=3, device="cpu")
    run_fold(ws, splits.folds[0], cfg, out / "f0", out / "cache")
    return out / "f0"


@pytest.fixture(scope="session")
def bundle_dir(
    ws: WindowSet, splits: SplitManifest, fold_dir, tmp_path_factory: pytest.TempPathFactory
):
    from kai.bench.bundle import BundleMeta, build_bundle
    from kai.cloud.heads import HeadZoo
    from kai.profile.profiler import profile_menu
    from kai.select.cost import Overheads

    zoo = HeadZoo(fold_dir)
    profiles, _ = profile_menu(list(zoo.reps.values()), zoo, ws.x[:8], repeats=5, with_load=False)
    meta = BundleMeta(ws.dataset_id, splits.split_id, 0, 0, 0, ws.sample_rate_hz)
    prices = Prices(name="t", story="t", p_byte=1.0, p_j=0.0, p_cpu_ms=0.0, p_late_ms=100.0)
    return build_bundle(
        fold_dir, tmp_path_factory.mktemp("bundles"), profiles, Overheads(0.05, 0.02, 0.01), meta,
        prices, EnergyModel(p_cpu_w=1.0, e_bit_j=1e-7),
    )  # fmt: skip
