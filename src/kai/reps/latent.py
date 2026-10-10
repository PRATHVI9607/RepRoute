"""Learned representations R5–R7 at run time: ONNX Runtime + numpy only (no torch on the edge)."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnxruntime as ort

from kai.contracts.representation import FloatArray, WindowArray
from kai.reps.base import BaseRep


def _session(onnx_bytes: bytes) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # one thread: reproducible timings, honest edge cost
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(onnx_bytes, opts, providers=["CPUExecutionProvider"])


class _OnnxRep(BaseRep):
    onnx_bytes: bytes
    table: np.ndarray

    def __init__(
        self, rep_id: str, window_n: int, onnx_bytes: bytes, table: np.ndarray, nbytes: int
    ):
        super().__init__(rep_id, window_n, fixed_bytes=nbytes)
        self.onnx_bytes = onnx_bytes
        self.table = np.ascontiguousarray(table, dtype="<f4")
        self._sess = _session(onnx_bytes)

    def blobs(self) -> tuple[bytes, ...]:
        return (self.onnx_bytes, self.table.tobytes())

    def run(self, x: WindowArray) -> np.ndarray:
        """Encoder output for one window [N] or a batch [n, N]."""
        xf = np.atleast_2d(x).astype(np.float32)[:, None, :] / np.float32(32768.0)
        out: np.ndarray = self._sess.run(None, {"x": xf})[0]
        return out


class OnnxLatent(_OnnxRep):
    """R5 / R6: dense latent quantised to int8 with a fixed per-dimension scale (1 byte per dim)."""

    def __init__(self, rep_id: str, window_n: int, onnx_bytes: bytes, scale: np.ndarray) -> None:
        super().__init__(rep_id, window_n, onnx_bytes, scale, nbytes=len(scale))

    def codec_params(self) -> dict[str, Any]:
        return {"window_n": self.window_n, "dim": len(self.table), "dtype": "int8"}

    def quantise(self, z: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(z / self.table), -127, 127).astype(np.int8)

    def encode(self, x: WindowArray) -> bytes:
        return self.quantise(self.run(x)[0]).tobytes()

    def decode_for_head(self, payload: bytes) -> FloatArray:
        return np.frombuffer(payload, np.int8).astype(np.float32) * self.table


class OnnxVQ(_OnnxRep):
    """R7: one codebook index (uint8) per latent position; the cloud looks the vectors up."""

    def __init__(self, rep_id: str, window_n: int, onnx_bytes: bytes, codebook: np.ndarray) -> None:
        if len(codebook) > 256:
            raise ValueError("codebook must fit one byte per index")
        n_pos = (
            _session(onnx_bytes).run(None, {"x": np.zeros((1, 1, window_n), np.float32)})[0].size
        )
        super().__init__(rep_id, window_n, onnx_bytes, codebook, nbytes=n_pos)

    def codec_params(self) -> dict[str, Any]:
        return {"window_n": self.window_n, "codebook": list(self.table.shape), "dtype": "uint8"}

    def encode(self, x: WindowArray) -> bytes:
        return self.run(x)[0].astype(np.uint8).tobytes()

    def decode_for_head(self, payload: bytes) -> FloatArray:
        return self.table[np.frombuffer(payload, np.uint8)].ravel()
