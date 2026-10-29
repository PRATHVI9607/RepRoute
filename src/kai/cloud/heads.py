"""Cloud head zoo: one decision function per representation (no torch — ONNX Runtime + LightGBM)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as ort

from kai.reps.base import BaseRep
from kai.reps.registry import load_menu

N_CLASSES = 3


@dataclass(frozen=True)
class LinearHead:
    """Standardise → affine → argmax. A constant decision is the zero-weight case."""

    mean: np.ndarray  # [F]
    scale: np.ndarray  # [F]
    weight: np.ndarray  # [C, F]
    bias: np.ndarray  # [C]

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(x).astype(np.float64) - self.mean) / self.scale
        out: np.ndarray = (z @ self.weight.T + self.bias).argmax(axis=1)
        return out


@dataclass(frozen=True)
class TreeHead:
    booster: lgb.Booster
    classes: np.ndarray  # class index of each booster output column

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = np.asarray(self.booster.predict(np.atleast_2d(x).astype(np.float64), num_threads=1))
        column = (proba > 0.5).astype(int) if proba.ndim == 1 else proba.argmax(axis=1)
        out: np.ndarray = self.classes[column]
        return out


Head = LinearHead | TreeHead


def save_head(head: Head, stem: Path) -> None:
    """`<stem>.npz` for a linear head; `<stem>.txt` + `<stem>.classes.npy` for a tree head."""
    if isinstance(head, LinearHead):
        np.savez(
            stem.with_suffix(".npz"),
            mean=head.mean,
            scale=head.scale,
            weight=head.weight,
            bias=head.bias,
        )
    else:
        head.booster.save_model(stem.with_suffix(".txt"))
        np.save(stem.with_suffix(".classes.npy"), head.classes)


def load_head(stem: Path) -> Head:
    if stem.with_suffix(".npz").exists():
        with np.load(stem.with_suffix(".npz")) as z:
            return LinearHead(z["mean"], z["scale"], z["weight"], z["bias"])
    booster = lgb.Booster(model_file=str(stem.with_suffix(".txt")))
    return TreeHead(booster, np.load(stem.with_suffix(".classes.npy")))


class HeadZoo:
    """`f_raw` for waveform representations, a fitted head for every feature representation."""

    def __init__(self, directory: Path) -> None:
        self.reps: dict[str, BaseRep] = {r.rep_id: r for r in load_menu(directory / "menu")}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._f_raw = ort.InferenceSession(
            (directory / "f_raw.onnx").read_bytes(), opts, providers=["CPUExecutionProvider"]
        )
        self._heads = {
            rid: load_head(directory / "heads" / rid)
            for rid, rep in self.reps.items()
            if rep.head_input == "features"
        }
        self.by_hash = {bytes.fromhex(r.contract_hash()): r for r in self.reps.values()}

    def head(self, rep_id: str, decoded: np.ndarray) -> int:
        """Decision for one decoded payload (the output of `decode_for_head`)."""
        if self.reps[rep_id].head_input == "waveform":
            logits = self._f_raw.run(None, {"x": decoded[None, None, :].astype(np.float32)})[0]
            return int(logits[0].argmax())
        return int(self._heads[rep_id].predict(decoded)[0])

    def decide_timed(self, rep_id: str, payload: bytes) -> tuple[int, float, float]:
        """Decode and decide for one payload, returning (decision, t_dec_ms, t_head_ms)."""
        t0 = time.perf_counter_ns()
        decoded = self.reps[rep_id].decode_for_head(payload)
        t1 = time.perf_counter_ns()
        decision = self.head(rep_id, decoded)
        t2 = time.perf_counter_ns()
        return decision, (t1 - t0) / 1e6, (t2 - t1) / 1e6

    def decide(self, rep_id: str, payload: bytes) -> int:
        decision, _, _ = self.decide_timed(rep_id, payload)
        return decision
