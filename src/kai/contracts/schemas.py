"""Pydantic schemas for the artifacts roles exchange (PRD §5.4, §14.1)."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

Partition = Literal["train", "ptrain", "calib", "test"]
PARTITIONS: tuple[Partition, ...] = get_args(Partition)
Flag = Literal["certified", "deferred", "uncertified", "dropped"]
FLAGS: tuple[Flag, ...] = get_args(Flag)
EnergySource = Literal["measured", "modelled"]
Variant = Literal["v0", "v1", "v1_per_rep", "v2"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class SplitFold(_Frozen):
    """One leave-bearings-out assignment. No bearing may appear in two partitions."""

    repeat: int
    fold: int
    train: list[str]
    ptrain: list[str]
    calib: list[str]
    test: list[str]
    # L2 only: test windows come from this operating condition, all other partitions exclude it.
    heldout_condition: str | None = None

    @model_validator(mode="after")
    def _disjoint_and_nonempty(self) -> Self:
        seen: dict[str, str] = {}
        for part in PARTITIONS:
            ids = self.bearings(part)
            if not ids:
                raise ValueError(f"partition {part} is empty")
            for b in ids:
                if b in seen:
                    raise ValueError(f"bearing {b} is in both {seen[b]} and {part}")
                seen[b] = part
        return self

    def bearings(self, partition: Partition) -> list[str]:
        ids: list[str] = getattr(self, partition)
        return ids


class SplitManifest(_Frozen):
    dataset_id: str
    level: Literal["L1", "L2"]
    seed: int
    folds: list[SplitFold]

    @property
    def split_id(self) -> str:
        return _digest(self.model_dump(mode="json"))[:12]


class Prices(_Frozen):
    """Physical price preset (PRD §5.6). Costs are in arbitrary but fixed currency units."""

    name: str
    story: str
    p_byte: float = Field(ge=0)
    p_j: float = Field(ge=0)
    p_cpu_ms: float = Field(ge=0)
    p_late_ms: float = Field(ge=0)


class EnergyModel(_Frozen):
    """P = p_idle_w + p_cpu_w·util on the edge CPU, plus e_bit_j per transmitted bit."""

    p_cpu_w: float = Field(ge=0)
    e_bit_j: float = Field(ge=0)
    source: EnergySource = "modelled"


class RepProfile(_Frozen):
    """Measured size and timing of one representation (PRD §14.1 `profiles.parquet`)."""

    rep_id: str
    size_mean_bytes: float
    size_q95_bytes: float
    t_enc_ms: float  # median at edge load ρ = 0
    t_enc_mad_ms: float
    t_enc_q95_ms: float
    t_dec_ms: float
    t_head_ms: float
    # t_enc(ρ) = t_enc_ms / (1 - ρ)**load_exponent; 1.0 is the processor-sharing model.
    load_exponent: float = 1.0


class LambdaEntry(_Frozen):
    variant: Variant
    alpha: float
    delta: float | None = None
    lam: float | None = None  # single threshold (v0, v1, v2)
    lam_per_rep: dict[str, float] | None = None  # v1_per_rep
    n_calib: int

    @model_validator(mode="after")
    def _one_threshold_kind(self) -> Self:
        if (self.lam is None) == (self.lam_per_rep is None):
            raise ValueError("exactly one of lam / lam_per_rep must be set")
        return self


class LambdaTable(_Frozen):
    entries: list[LambdaEntry]

    def get(self, variant: Variant, alpha: float) -> LambdaEntry:
        for e in self.entries:
            if e.variant == variant and abs(e.alpha - alpha) < 1e-12:
                return e
        raise KeyError(f"no calibration for variant={variant} alpha={alpha}")


class BundleManifest(_Frozen):
    """Immutable calibration bundle index. `files` maps relative path → sha256."""

    bundle_id: str
    created_utc: str
    git_sha: str
    git_dirty: bool
    lock_hash: str
    dataset_id: str
    split_id: str
    repeat: int
    fold: int
    seed: int
    window_n: int
    sample_rate_hz: float
    onnx_opset: int
    rep_ids: list[str]  # menu order; index = column in predictor output
    contract_hashes: dict[str, str]
    files: dict[str, str]


class RunRecord(_Frozen):
    """One row of the run registry."""

    run_id: str
    experiment: str
    created_utc: str
    git_sha: str
    git_dirty: bool
    lock_hash: str
    config_hash: str
    seed: int
    params: dict[str, str | int | float | bool | None]
    metrics: dict[str, float]


def config_hash(config: object) -> str:
    return _digest(config)[:12]
