"""Representation contract (PRD §5.4): what every payload representation must implement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

WindowArray = npt.NDArray[np.int16]
FloatArray = npt.NDArray[np.float32]

# One byte on the wire (frame field `rep_id`). Append only: codes are part of the protocol.
REP_CODES: dict[str, int] = {
    "r0_raw": 0,
    "r1_lossless": 1,
    "r2a_wavelet": 2,
    "r2b_wavelet": 3,
    "r3_spectral": 4,
    "r4_summary": 5,
    "r5_ae": 6,
    "r6_bottleneck": 7,
    "r7_vq": 8,
    "r8_jepa": 9,
    "r9_minirocket": 10,
}
RAW_REP_ID = "r0_raw"

# "waveform": decode_for_head returns a float32 window the raw reference model consumes.
# "features": decode_for_head returns a feature vector for a per-representation head.
HeadInput = Literal["waveform", "features"]


@dataclass(frozen=True)
class SizeModel:
    """Payload size in bytes, excluding the 28-byte frame overhead."""

    mean_bytes: float
    q95_bytes: float  # == mean for fixed-size codecs


@runtime_checkable
class Representation(Protocol):
    @property
    def rep_id(self) -> str: ...

    @property
    def version(self) -> str: ...  # bump on ANY change affecting payload semantics

    @property
    def head_input(self) -> HeadInput: ...

    def contract_hash(self) -> str: ...  # 8 hex chars = 4 bytes on the wire

    def size_model(self) -> SizeModel: ...

    def encode(self, x: WindowArray) -> bytes: ...  # edge-safe, deterministic

    def decode_for_head(self, payload: bytes) -> FloatArray: ...  # cloud side


def contract_hash_of(
    rep_id: str, version: str, params: dict[str, Any], blobs: tuple[bytes, ...] = ()
) -> str:
    """hash(code version, params, quantization, dtype, endianness, model bytes) → 8 hex chars."""
    h = hashlib.sha256()
    h.update(json.dumps([rep_id, version, params], sort_keys=True, default=str).encode())
    for blob in blobs:
        h.update(hashlib.sha256(blob).digest())
    return h.hexdigest()[:8]
