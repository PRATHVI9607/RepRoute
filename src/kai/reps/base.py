"""Shared plumbing for representations: contract hash, measured size model, (de)serialisation."""

from __future__ import annotations

from typing import Any

import numpy as np

from kai.contracts.representation import HeadInput, SizeModel, WindowArray, contract_hash_of


class BaseRep:
    version: str = "1"
    head_input: HeadInput = "features"
    onnx_bytes: bytes | None = None  # learned representations carry a model ...
    table: np.ndarray | None = None  # ... and a quantisation scale or codebook

    def __init__(self, rep_id: str, window_n: int, fixed_bytes: int | None = None) -> None:
        self.rep_id = rep_id
        self.window_n = window_n
        self._size = None if fixed_bytes is None else SizeModel(fixed_bytes, fixed_bytes)

    # --- contract -------------------------------------------------------------------------
    def codec_params(self) -> dict[str, Any]:
        """Everything that changes payload semantics. Feeds the contract hash."""
        return {"window_n": self.window_n}

    def blobs(self) -> tuple[bytes, ...]:
        return ()

    def contract_hash(self) -> str:
        return contract_hash_of(self.rep_id, self.version, self.codec_params(), self.blobs())

    def size_model(self) -> SizeModel:
        if self._size is None:
            raise RuntimeError(f"{self.rep_id}: variable-size codec, call fit() first")
        return self._size

    def encode(self, x: WindowArray) -> bytes:
        raise NotImplementedError

    def decode_for_head(self, payload: bytes) -> np.ndarray:
        raise NotImplementedError

    # --- lab ------------------------------------------------------------------------------
    def set_size(self, sizes_bytes: np.ndarray) -> None:
        """Declare the size model from measured payload sizes (training windows only)."""
        self._size = SizeModel(float(sizes_bytes.mean()), float(np.quantile(sizes_bytes, 0.95)))

    def fit(self, train: WindowArray) -> None:
        """Measure the size model on training windows (variable-size codecs only)."""
        self.set_size(np.array([len(self.encode(w)) for w in train]))

    # --- bundle ---------------------------------------------------------------------------
    def to_params(self) -> dict[str, Any]:
        s = self.size_model()
        return {
            **self.codec_params(),
            "size_mean_bytes": s.mean_bytes,
            "size_q95_bytes": s.q95_bytes,
        }

    def _restore_size(self, params: dict[str, Any]) -> None:
        self._size = SizeModel(params["size_mean_bytes"], params["size_q95_bytes"])
