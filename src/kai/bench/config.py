"""Compose the Hydra configuration and validate the parts that carry physical meaning."""

from __future__ import annotations

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from kai.contracts.schemas import EnergyModel, Prices

ROOT = Path(__file__).resolve().parents[3]  # the repository: configs, lockfile, git


def work() -> Path:
    """Where data/, artifacts/ and experiments/ live. `KAI_WORK` redirects them (tests, CI)."""
    return Path(os.environ.get("KAI_WORK", ROOT))


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """`configs/config.yaml` with Hydra overrides such as `scenario=mini profile=battery_node`."""
    with initialize_config_dir(str(ROOT / "configs"), version_base="1.3"):
        return compose("config", overrides=overrides or [])


def prices_of(cfg: DictConfig) -> Prices:
    return Prices.model_validate(OmegaConf.to_container(cfg.profile.prices))


def energy_of(cfg: DictConfig) -> EnergyModel:
    return EnergyModel.model_validate(OmegaConf.to_container(cfg.profile.energy))


def as_dict(cfg: DictConfig) -> dict:
    out = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(out, dict)
    return out
