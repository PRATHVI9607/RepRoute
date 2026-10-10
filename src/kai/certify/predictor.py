"""Sufficiency predictor ĝ_r(φ) ≈ P(d_r = 1 | φ) at run time: numpy only — edge-safe.

Each non-raw representation has its own small dense network (a logistic model is the depth-1
case). Fitting lives in `kai.certify.fit` (lab plane, scikit-learn).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
Layers = tuple[tuple[F64, F64], ...]  # ((W [in, out], b [out]), ...)


@dataclass(frozen=True)
class Predictor:
    rep_ids: tuple[str, ...]  # full menu; column 0 is raw and always scores 0
    mean: F64  # [P]
    scale: F64  # [P]
    nets: tuple[Layers, ...]  # one per non-raw representation

    def predict(self, phi: npt.NDArray[np.floating]) -> F64:
        """ĝ for one window `[P]` → `[K]`, or a batch `[n, P]` → `[n, K]`."""
        h = (np.atleast_2d(phi).astype(np.float64) - self.mean) / self.scale
        out = np.zeros((len(h), len(self.rep_ids)))
        if self._stacked is not None:  # every net has the same shape: one pass for all of them
            h = h[None]  # [1, n, P] against weights [K-1, in, out]
            for i, (w, b) in enumerate(self._stacked):
                h = h @ w + b[:, None, :]
                if i < len(self._stacked) - 1:
                    h = np.maximum(h, 0.0)
            logits = h[..., 0].T  # [n, K-1]
        else:
            logits = np.empty((len(h), len(self.nets)))
            for k, layers in enumerate(self.nets):
                a = h
                for i, (w, b) in enumerate(layers):
                    a = a @ w + b
                    if i < len(layers) - 1:
                        a = np.maximum(a, 0.0)
                logits[:, k] = a[:, 0]
        out[:, 1:] = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        return out if np.ndim(phi) == 2 else out[0]

    @cached_property
    def _stacked(self) -> tuple[tuple[F64, F64], ...] | None:
        shapes = {tuple(w.shape for w, _ in layers) for layers in self.nets}
        if len(shapes) != 1:
            return None
        depth = len(self.nets[0])
        return tuple(
            (np.stack([n[i][0] for n in self.nets]), np.stack([n[i][1] for n in self.nets]))
            for i in range(depth)
        )

    def save(self, path: Path) -> None:
        arrays: dict[str, npt.NDArray[np.generic]] = {
            "rep_ids": np.asarray(self.rep_ids),
            "mean": self.mean,
            "scale": self.scale,
        }
        for k, layers in enumerate(self.nets):
            for i, (w, b) in enumerate(layers):
                arrays[f"w_{k}_{i}"], arrays[f"b_{k}_{i}"] = w, b
        np.savez(path, **arrays)  # type: ignore[arg-type]

    @staticmethod
    def load(path: Path) -> Predictor:
        with np.load(path) as z:
            rep_ids = tuple(z["rep_ids"].tolist())
            nets = []
            for k in range(len(rep_ids) - 1):
                depth = sum(1 for name in z.files if name.startswith(f"w_{k}_"))
                nets.append(tuple((z[f"w_{k}_{i}"], z[f"b_{k}_{i}"]) for i in range(depth)))
            return Predictor(rep_ids, z["mean"], z["scale"], tuple(nets))
