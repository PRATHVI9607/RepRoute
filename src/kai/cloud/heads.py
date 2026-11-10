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


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


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

    def proba(self, x: np.ndarray) -> np.ndarray:
        """[n, C] class probabilities (softmax of the logits); argmax equals `predict`."""
        z = (np.atleast_2d(x).astype(np.float64) - self.mean) / self.scale
        return _softmax(z @ self.weight.T + self.bias)


@dataclass(frozen=True)
class TreeHead:
    booster: lgb.Booster
    classes: np.ndarray  # class index of each booster output column

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = np.asarray(self.booster.predict(np.atleast_2d(x).astype(np.float64), num_threads=1))
        column = (proba > 0.5).astype(int) if proba.ndim == 1 else proba.argmax(axis=1)
        out: np.ndarray = self.classes[column]
        return out

    def proba(self, x: np.ndarray, n_classes: int | None = None) -> np.ndarray:
        """[n, C] with column c = class c; classes the booster never saw get probability 0."""
        p = np.asarray(self.booster.predict(np.atleast_2d(x).astype(np.float64), num_threads=1))
        if p.ndim == 1:  # binary: the booster outputs P(classes[1])
            p = np.stack([1.0 - p, p], axis=1)
        out = np.zeros((len(p), max(int(self.classes.max()) + 1, n_classes or 0)))
        out[:, self.classes] = p
        return out


class OnnxHead:
    """A CNN head exported to ONNX (R10 log-mel, ADR 0018): input [B, 1, F], argmax of logits."""

    def __init__(self, onnx_bytes: bytes) -> None:
        self.onnx_bytes = onnx_bytes
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(onnx_bytes, opts, providers=["CPUExecutionProvider"])

    def predict(self, x: np.ndarray, batch: int = 2048) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, np.float32))
        logits = [
            self._session.run(None, {"x": x[i : i + batch, None, :]})[0]
            for i in range(0, len(x), batch)
        ]
        out: np.ndarray = np.concatenate(logits).argmax(axis=1)
        return out

    def proba(self, x: np.ndarray, batch: int = 2048) -> np.ndarray:
        """[n, C] softmax of the logits; argmax equals `predict`."""
        x = np.atleast_2d(np.asarray(x, np.float32))
        logits = [
            self._session.run(None, {"x": x[i : i + batch, None, :]})[0]
            for i in range(0, len(x), batch)
        ]
        return _softmax(np.concatenate(logits).astype(np.float64))


Head = LinearHead | TreeHead | OnnxHead


def save_head(head: Head, stem: Path) -> None:
    """`<stem>.npz` linear, `<stem>.onnx` CNN, `<stem>.txt` + `<stem>.classes.npy` tree."""
    # a rerun may change the head family: a stale file of another family would be loaded first
    for suffix in (".npz", ".onnx", ".txt", ".classes.npy"):
        stem.with_suffix(suffix).unlink(missing_ok=True)
    if isinstance(head, OnnxHead):
        stem.with_suffix(".onnx").write_bytes(head.onnx_bytes)
    elif isinstance(head, LinearHead):
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
    if stem.with_suffix(".onnx").exists():
        return OnnxHead(stem.with_suffix(".onnx").read_bytes())
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
        if len(self.by_hash) != len(self.reps):  # the wire carries a short hash: never guess
            raise ValueError("two representations share a contract hash")

    def head(self, rep_id: str, decoded: np.ndarray) -> int:
        """Decision for one decoded payload (the output of `decode_for_head`)."""
        if self.reps[rep_id].head_input == "waveform":
            logits = self._f_raw.run(
                None, {"x": decoded[None, None, :].astype(np.float32, copy=False)}
            )[0]
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
