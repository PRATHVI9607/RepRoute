"""Lab pipeline: one leave-bearings-out fold → models, menu, heads and the Disagreement Table.

Everything here is fit on the `train` partition only. Predictions are then produced for every
window so that `ptrain` (predictor), `calib` (λ̂) and `test` see models that never saw their
bearings.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from kai.certify.probe import PROBE_VERSION, probe
from kai.cloud.heads import save_head
from kai.contracts.schemas import PARTITIONS, Partition, SplitFold
from kai.contracts.windows import WindowSet
from kai.data.splits import assert_leakage_safe, partition_mask
from kai.reps.base import BaseRep
from kai.reps.classical import Lossless
from kai.reps.latent import OnnxLatent, OnnxVQ
from kai.reps.registry import MENU_ORDER, classical_menu, save_menu
from kai.tasks import train as tr
from kai.tasks.heads import tune_and_fit

LEARNED = ("r5_ae", "r6_bottleneck", "r7_vq")


@dataclass(frozen=True)
class FoldConfig:
    f_raw: str = "spec"  # "spec" (SpecNet, ADR 0004) or "wave" (RawNet, kept for comparison)
    epochs_raw: int = 30  # only used by f_raw="wave"
    epochs_latent: int = 15
    ae_dim: int = 32
    bottleneck_dim: int = 16
    vq_code_dim: int = 8
    head_trials: int = 10
    seed: int = 0
    device: str = "auto"
    learned: tuple[str, ...] = LEARNED


@dataclass(frozen=True)
class Table:
    """Disagreement Table as arrays. Column k is representation `rep_ids[k]`; column 0 is raw."""

    rep_ids: tuple[str, ...]
    window_id: np.ndarray  # [n]
    bearing_id: np.ndarray  # [n] physical bearing: the exchangeability / bootstrap block
    partition: np.ndarray  # [n] index into PARTITIONS
    label: np.ndarray  # [n]
    yhat: np.ndarray  # [n, K] decision of each representation's head
    payload_bytes: np.ndarray  # [n, K]
    phi: np.ndarray  # [n, P] probe features

    @property
    def d(self) -> np.ndarray:
        """Decision-agreement loss d_r(x) = 1[ŷ_r(x) ≠ f_raw(x)]. Not label accuracy."""
        return self.yhat != self.yhat[:, :1]

    def of(self, partition: Partition) -> np.ndarray:
        return self.partition == PARTITIONS.index(partition)

    def save(self, path: Path) -> None:
        arrays = {k: np.asarray(v) for k, v in asdict(self).items()}
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]  # numpy stub: **kwargs

    @staticmethod
    def load(path: Path) -> Table:
        with np.load(path) as z:
            arrays = {f.name: z[f.name] for f in fields(Table)}
            return Table(**{**arrays, "rep_ids": tuple(arrays["rep_ids"].tolist())})

    def to_frame(self, ws: WindowSet) -> pd.DataFrame:
        """Long format of PRD §14.1: one row per (window, representation)."""
        n, k = self.yhat.shape
        return pd.DataFrame(
            {
                "window_id": np.repeat(self.window_id, k),
                "bearing_id": np.repeat(self.bearing_id, k),
                "condition": np.repeat(ws.condition, k),
                "partition": np.repeat(np.asarray(PARTITIONS)[self.partition], k),
                "rep_id": np.tile(np.asarray(self.rep_ids), n),
                "label": np.repeat(self.label, k),
                "yhat": self.yhat.ravel(),
                "yhat_raw": np.repeat(self.yhat[:, 0], k),
                "d": self.d.ravel(),
                "correct": (self.yhat == self.label[:, None]).ravel(),
                "payload_bytes": self.payload_bytes.ravel(),
            }
        )


# --- fold-independent outputs of the classical codecs (computed once per dataset) ---------------


def _classical_chunk(args: tuple[np.ndarray, int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    x, window_n = args
    out = {}
    for rep in classical_menu(window_n):
        payloads = [rep.encode(w) for w in x]
        sizes = np.array([len(p) for p in payloads], dtype=np.int32)
        if rep.rep_id == "r0_raw":
            decoded = np.empty((len(x), 0), np.float32)  # identical to x / 32768; never stored
        elif isinstance(rep, Lossless):
            pairs = zip(payloads, x, strict=True)
            if any(not np.array_equal(rep.decode_samples(p), w) for p, w in pairs):
                raise AssertionError("lossless codec failed to round-trip")
            decoded = np.empty((len(x), 0), np.float32)  # bit-identical to raw by the check above
        else:
            decoded = np.stack([rep.decode_for_head(p) for p in payloads])
        out[rep.rep_id] = (decoded, sizes)
    out["phi"] = (np.stack([probe(w) for w in x]).astype(np.float32), np.empty(0, np.int32))
    return out


def classical_outputs(ws: WindowSet, cache_dir: Path, workers: int = 8) -> dict[str, np.ndarray]:
    """Decoded head inputs, payload sizes and probe features for every window; cached on disk."""
    # keyed on the data and on the code that produced the features: no stale cache hits
    menu = classical_menu(ws.window_n)
    code = [PROBE_VERSION, *(f"{r.rep_id}:{r.version}:{r.codec_params()}" for r in menu)]
    crc = zlib.crc32(ws.x[::97].tobytes() + repr(code).encode())
    tag = f"{ws.dataset_id}_{len(ws)}x{ws.window_n}_{crc:08x}"
    names = [*(r.rep_id for r in classical_menu(ws.window_n)), "phi"]
    paths = {
        f"{n}.{kind}": cache_dir / f"{tag}_{n}.{kind}.npy"
        for n in names
        for kind in ("out", "bytes")
    }
    if not all(p.exists() for p in paths.values()):
        cache_dir.mkdir(parents=True, exist_ok=True)
        chunks = [(c, ws.window_n) for c in np.array_split(ws.x, max(1, len(ws) // 256))]
        # spawn, not fork: the parent may already hold torch / BLAS threads
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            parts = list(pool.map(_classical_chunk, chunks))
        for n in names:
            np.save(paths[f"{n}.out"], np.concatenate([p[n][0] for p in parts]))
            np.save(paths[f"{n}.bytes"], np.concatenate([p[n][1] for p in parts]))
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


# --- one fold ------------------------------------------------------------------------------------


def _learned_rep(
    rep_id: str, x: np.ndarray, y: np.ndarray, cfg: FoldConfig, device: torch.device
) -> OnnxLatent | OnnxVQ:
    """Train one learned encoder on training windows and wrap its ONNX export as an edge rep."""
    n = x.shape[1]
    if rep_id == "r5_ae":
        ae = tr.train_autoencoder(x, cfg.ae_dim, cfg.epochs_latent, cfg.seed, device)
        z = tr.predict(ae.encoder, x[:4096].astype(np.float32) / 32768.0)
        scale = np.maximum(np.quantile(np.abs(z), 0.999, axis=0), 1e-6) / 127.0
        return OnnxLatent(rep_id, n, tr.export_onnx(ae.encoder, n), scale)
    if rep_id == "r6_bottleneck":
        model = tr.train_bottleneck(x, y, cfg.bottleneck_dim, cfg.epochs_latent, cfg.seed, device)
        encoder = nn.Sequential(model.encoder, nn.Tanh())  # tanh output ∈ (-1, 1): scale 1/127
        return OnnxLatent(
            rep_id, n, tr.export_onnx(encoder, n), np.full(cfg.bottleneck_dim, 1 / 127.0)
        )
    if rep_id == "r7_vq":
        vq = tr.train_vqvae(x, cfg.vq_code_dim, cfg.epochs_latent, cfg.seed, device).encoder
        return OnnxVQ(rep_id, n, tr.export_onnx(vq, n), vq.codebook.detach().numpy())
    raise ValueError(f"unknown learned representation {rep_id}")


def _latent_features(rep: OnnxLatent | OnnxVQ, x: np.ndarray, batch: int = 512) -> np.ndarray:
    """Head input for every window, through the same quantise → dequantise path the wire uses."""
    out = []
    for i in range(0, len(x), batch):
        z = rep.run(x[i : i + batch])
        if isinstance(rep, OnnxLatent):
            out.append(rep.quantise(z).astype(np.float32) * rep.table)
        else:
            out.append(rep.table[z].reshape(len(z), -1))
    return np.concatenate(out)


def run_fold(  # noqa: PLR0915 - one linear recipe; splitting it would hide the order of steps
    ws: WindowSet, fold: SplitFold, cfg: FoldConfig, out_dir: Path, cache_dir: Path
) -> tuple[Table, dict[str, float]]:
    """Train everything for one fold and write its artifacts to `out_dir`."""
    assert_leakage_safe(ws, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = tr.resolve_device(cfg.device)
    cache = classical_outputs(ws, cache_dir)
    part = np.full(len(ws), -1, np.int8)  # -1: in no partition (L2 drops the held-out condition)
    for i, p in enumerate(PARTITIONS):
        part[partition_mask(ws, fold, p)] = i
    is_train = part == PARTITIONS.index("train")
    x_tr, y_tr, g_tr = ws.x[is_train], ws.label[is_train], ws.bearing_id[is_train]

    t0 = time.perf_counter()

    def stage(name: str) -> None:
        print(f"  [{out_dir.name}] {name}: {time.perf_counter() - t0:.0f}s", flush=True)

    f_raw: nn.Module
    if cfg.f_raw == "spec":
        f_raw = tr.train_specnet(x_tr, y_tr, g_tr, cfg.seed)
    else:
        f_raw = tr.train_rawnet(x_tr, y_tr, cfg.epochs_raw, cfg.seed, device)
    stage("f_raw trained")
    f_raw_onnx = tr.export_onnx(f_raw, ws.window_n)
    (out_dir / "f_raw.onnx").write_bytes(f_raw_onnx)
    # decisions come from the exported model: what is certified is exactly what the cloud runs
    session = ort.InferenceSession(f_raw_onnx, providers=["CPUExecutionProvider"])

    def raw_decision(x_float: np.ndarray) -> np.ndarray:
        logits = session.run(None, {"x": np.asarray(x_float, np.float32)[:, None, :]})[0]
        return logits.argmax(1)

    reps: list[BaseRep] = classical_menu(ws.window_n)
    yhat: dict[str, np.ndarray] = {}
    nbytes: dict[str, np.ndarray] = {}
    (out_dir / "heads").mkdir(exist_ok=True)

    def fit_head(rep_id: str, feats: np.ndarray) -> np.ndarray:
        # distillation: the head learns the reference model's decision, the thing being certified
        target = yhat["r0_raw"][is_train]
        head = tune_and_fit(feats[is_train], target, g_tr, cfg.head_trials, cfg.seed)
        save_head(head, out_dir / "heads" / rep_id)
        return head.predict(feats)

    for rep in reps:
        rid = rep.rep_id
        nbytes[rid] = np.asarray(cache[f"{rid}.bytes"])
        rep.set_size(nbytes[rid][is_train])
        if rid == "r0_raw":
            yhat[rid] = np.concatenate(
                [
                    raw_decision(ws.x[i : i + 2048] / np.float32(32768))
                    for i in range(0, len(ws), 2048)
                ]
            )
        elif rid == "r1_lossless":
            yhat[rid] = yhat["r0_raw"]  # decoded samples are bit-identical (checked in the cache)
        elif rep.head_input == "waveform":
            dec = cache[f"{rid}.out"]
            yhat[rid] = np.concatenate(
                [raw_decision(np.asarray(dec[i : i + 2048])) for i in range(0, len(ws), 2048)]
            )
        else:
            yhat[rid] = fit_head(rid, np.asarray(cache[f"{rid}.out"]))
        stage(rid)

    for rid in cfg.learned:
        # the task-aware bottleneck is distilled from the reference model (supervised compression)
        learned = _learned_rep(rid, x_tr, yhat["r0_raw"][is_train], cfg, device)
        reps.append(learned)
        nbytes[rid] = np.full(len(ws), learned.size_model().mean_bytes, np.int32)
        yhat[rid] = fit_head(rid, _latent_features(learned, ws.x))
        stage(rid)

    order = [r for r in MENU_ORDER if r in yhat]
    reps.sort(key=lambda r: order.index(r.rep_id))
    save_menu(reps, out_dir / "menu")
    table = Table(
        rep_ids=tuple(order),
        window_id=ws.window_id,
        bearing_id=ws.bearing_id,
        partition=part,
        label=ws.label,
        yhat=np.stack([yhat[r] for r in order], axis=1).astype(np.int8),
        payload_bytes=np.stack([nbytes[r] for r in order], axis=1).astype(np.int32),
        phi=np.asarray(cache["phi.out"]),
    )
    table.save(out_dir / "table.npz")
    table.to_frame(ws).to_parquet(out_dir / "disagreement.parquet", index=False)

    metrics: dict[str, float] = {}
    test = table.of("test")
    for k, rid in enumerate(order):
        metrics[f"macro_f1/{rid}"] = float(
            f1_score(ws.label[test], table.yhat[test, k], average="macro")
        )
        for p in ("ptrain", "calib", "test"):
            metrics[f"disagreement/{p}/{rid}"] = float(table.d[table.of(p), k].mean())
        metrics[f"bytes_mean/{rid}"] = float(table.payload_bytes[test, k].mean())
    metrics["majority_baseline_macro_f1"] = float(
        f1_score(ws.label[test], np.full(test.sum(), np.bincount(y_tr).argmax()), average="macro")
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return table, metrics
