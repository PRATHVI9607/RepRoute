"""Lab pipeline: one leave-bearings-out fold → models, menu, heads and the Disagreement Table.

Everything here is fit on the `train` partition only. Predictions are then produced for every
window so that `ptrain` (predictor), `calib` (λ̂) and `test` see models that never saw their
bearings.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import zlib
from collections.abc import Callable
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
from kai.cloud.heads import Head, OnnxHead, save_head
from kai.contracts.schemas import PARTITIONS, Partition, SplitFold
from kai.contracts.windows import WindowSet
from kai.data.splits import assert_leakage_safe, partition_mask
from kai.reps.base import BaseRep
from kai.reps.classical import Lossless
from kai.reps.latent import OnnxLatent, OnnxVQ
from kai.reps.logmel import LogMel
from kai.reps.registry import MENU_ORDER, classical_menu, save_menu
from kai.tasks import train as tr
from kai.tasks.heads import tune_and_fit
from kai.tasks.models import SpecNet, W2VClassifier

R5_SCALE_SAMPLE = 16384  # train windows behind the r5 int8 scale (at least 4096)
LEARNED = ("r5_ae", "r6_bottleneck", "r7_vq")
F_RAW_RULES = ("ptrain_bootstrap", "ptrain_best")
F_RAW_BOOTSTRAP = 2000


@dataclass(frozen=True)
class FoldConfig:
    # "env" (EnvNet, ADR 0006), "kin" (KinNet, long windows, ADR 0012), "spec" (SpecNet, ADR 0004)
    # or "wave" (RawNet, kept for comparison)
    f_raw: str = "spec"
    epochs_raw: int = 30  # only used by f_raw="wave"
    epochs_latent: int = 15
    ae_dim: int = 32
    bottleneck_dim: int = 16
    vq_code_dim: int = 8
    head_trials: int = 10
    # R6 encoder: "generic" (a CNN on the waveform) or "teacher_front" (the reference model's
    # fixed front-end + a learned projection; needs f_raw "env" or "spec"). ADR 0006, addendum.
    bottleneck: str = "generic"
    seed: int = 0
    device: str = "auto"
    learned: tuple[str, ...] = LEARNED
    extra_wavelets: bool = False  # r2c/r2d mid-size codecs (ADR 0016, exploratory)
    logmel: bool = (
        False  # r10a/r10b log-mel grids with a distilled CNN head (ADR 0018, exploratory)
    )
    # r5 int8 scale: "random_train" = quantile over a seeded random sample of the fold's train
    # windows (ADR 0028 fix 15); folds built before it used the first 4096 in loader order.
    r5_scale: str = "random_train"
    # per-fold f_raw choice (ADR 0028 §7): candidates trained on the fold's train units, scored on
    # its ptrain units. `f_raw` is the incumbent (first candidate). Empty = the single f_raw above.
    f_raw_candidates: tuple[str, ...] = ()
    f_raw_rule: str = ""  # "ptrain_bootstrap" (challenger must beat the incumbent) | "ptrain_best"

    def __post_init__(self) -> None:
        if bool(self.f_raw_candidates) != bool(self.f_raw_rule):
            raise ValueError("f_raw_candidates and f_raw_rule go together")
        if self.f_raw_rule and self.f_raw_rule not in F_RAW_RULES:
            raise ValueError(f"f_raw_rule must be one of {F_RAW_RULES}")
        if self.f_raw_candidates and self.f_raw_candidates[0] != self.f_raw:
            raise ValueError("the first f_raw candidate is the incumbent and must equal f_raw")

    def recipe(self) -> dict[str, object]:
        """What determines the fold's models (the device does not): stored next to the fold."""
        out = asdict(self)
        del out["device"]
        out["learned"] = list(self.learned)
        if not self.extra_wavelets:  # folds trained before ADR 0016 keep matching their recipe
            del out["extra_wavelets"]
        if not self.logmel:  # ... and before ADR 0018
            del out["logmel"]
        if self.f_raw_candidates:
            out["f_raw_candidates"] = list(self.f_raw_candidates)
        else:  # folds trained before ADR 0028 §7 keep matching their recipe
            del out["f_raw_candidates"], out["f_raw_rule"]
        return out


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


def _classical_chunk(
    args: tuple[np.ndarray, int, bool, float | None, tuple[str, ...]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    x, window_n, extra_wavelets, logmel_hz, todo = args
    out = {}
    for rep in classical_menu(window_n, extra_wavelets, logmel_hz):
        if rep.rep_id not in todo:
            continue
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
    if "phi" in todo:
        out["phi"] = (np.stack([probe(w) for w in x]).astype(np.float32), np.empty(0, np.int32))
    return out


def _tmp(path: Path) -> Path:
    """A per-process temporary next to `path`: concurrent runs never write the same file."""
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def classical_outputs(
    ws: WindowSet,
    cache_dir: Path,
    workers: int = 8,
    extra_wavelets: bool = False,
    logmel: bool = False,
) -> dict[str, np.ndarray]:
    """Decoded head inputs, payload sizes and probe features for every window; cached on disk."""
    # keyed on the data and on the code that produced the features: no stale cache hits. Each
    # codec has its own key, so an opt-in codec (--logmel, --extra-wavelets) encodes only itself.
    logmel_hz = ws.sample_rate_hz if logmel else None
    menu = classical_menu(ws.window_n, extra_wavelets, logmel_hz)
    code = {r.rep_id: f"{r.rep_id}:{r.version}:{r.codec_params()}" for r in menu}
    code["phi"] = PROBE_VERSION
    data_crc = zlib.crc32(np.ascontiguousarray(ws.x).data)  # every window (ADR 0028 fix 16)
    paths = {}
    for n, c in code.items():
        tag = f"{ws.dataset_id}_{len(ws)}x{ws.window_n}_{zlib.crc32(c.encode(), data_crc):08x}"
        for kind in ("out", "bytes"):
            paths[f"{n}.{kind}"] = cache_dir / f"{tag}_{n}.{kind}.npy"
    todo = tuple(
        n for n in code if not (paths[f"{n}.out"].exists() and paths[f"{n}.bytes"].exists())
    )
    if todo:
        cache_dir.mkdir(parents=True, exist_ok=True)
        chunks = [
            (c, ws.window_n, extra_wavelets, logmel_hz, todo)
            for c in np.array_split(ws.x, max(1, len(ws) // 256))
        ]
        # filled chunk by chunk: no list of parts plus a concatenated copy in memory
        mmaps: dict[str, np.memmap] = {}
        start = 0
        # spawn, not fork: the parent may already hold torch / BLAS threads
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            for chunk, part in zip(chunks, pool.map(_classical_chunk, chunks), strict=True):
                for n, (out, sizes) in part.items():
                    for key, arr in ((f"{n}.out", out), (f"{n}.bytes", sizes)):
                        if key == "phi.bytes":
                            continue  # written below
                        if key not in mmaps:
                            mmaps[key] = np.lib.format.open_memmap(
                                _tmp(paths[key]), "w+", arr.dtype, (len(ws), *arr.shape[1:])
                            )
                        mmaps[key][start : start + len(chunk[0])] = arr
                start += len(chunk[0])
        if "phi" in todo:  # the probe has no payload: its sizes are stored as an empty array
            mmaps["phi.bytes"] = np.lib.format.open_memmap(
                _tmp(paths["phi.bytes"]), "w+", np.int32, (0,)
            )
        for key, mm in mmaps.items():
            mm.flush()
            os.replace(_tmp(paths[key]), paths[key])  # atomic: a crash never leaves a short file
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


# --- per-fold f_raw choice (ADR 0028 §7) ---------------------------------------------------


@dataclass
class Candidate:
    name: str
    model: nn.Module
    onnx: bytes


def train_candidate(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    cfg: FoldConfig,
    device: torch.device,
) -> Candidate:
    """Train one f_raw architecture on the given (train) windows and export it."""
    fixed_front = {"env": tr.train_envnet, "kin": tr.train_kinnet, "spec": tr.train_specnet}
    if name == "w2v":
        rng = np.random.default_rng([cfg.seed, 28, 1])
        # wav2vec2 always trains on the GPU when there is one; the small nets train on the fold's
        # device (the CPU is ~4x faster than the RTX 2050 for them, ADR 0028)
        w2v = tr.train_w2v(x, y, g, rng, tr.resolve_device("auto"))
        return Candidate(name, w2v, tr.export_w2v_onnx(w2v, x.shape[1]))
    model: nn.Module
    if name in fixed_front:
        model = fixed_front[name](x, y, g, cfg.seed)
    elif name == "wave":
        model = tr.train_rawnet(x, y, cfg.epochs_raw, cfg.seed, device)
    else:
        raise ValueError(f"unknown f_raw {name!r}")
    return Candidate(name, model, tr.export_onnx(model, x.shape[1]))


def _group_counts(
    y: np.ndarray, pred: np.ndarray, g: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group, per-class true positives, predicted and true counts: [G, K] each."""
    _, gi = np.unique(g, return_inverse=True)
    n = int(gi.max()) + 1
    tp, pc, tc = (np.zeros((n, k)) for _ in range(3))
    np.add.at(tc, (gi, y), 1)
    np.add.at(pc, (gi, pred), 1)
    np.add.at(tp, (gi[y == pred], y[y == pred]), 1)
    return tp, pc, tc


def _macro_f1(tp: np.ndarray, pc: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """Macro-F1 over the classes present (true or predicted); rows are independent resamples."""
    denom = pc + tc
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.sum(-1) / np.maximum((denom > 0).sum(-1), 1)


def select_f_raw(
    rule: str,
    names: tuple[str, ...],
    y: np.ndarray,
    g: np.ndarray,
    pred: dict[str, np.ndarray],
    size_bytes: dict[str, int],
    seed: int,
) -> tuple[str, dict[str, float]]:
    """Choose the fold's f_raw from candidates' predictions on its ptrain units.

    `ptrain_bootstrap`: a challenger replaces the incumbent (names[0]) only if the 95 % lower
    bound of the unit-bootstrap of F1_challenger − F1_incumbent is > 0 (the best passing one wins).
    `ptrain_best`: highest macro-F1; ties go to the smaller model file, then list order.
    Returns the choice and a flat record: `ptrain_f1/<name>`, `boot_lb/<name>`.
    """
    k = int(max(y.max(), *(p.max() for p in pred.values()))) + 1
    counts = {n: _group_counts(y, pred[n], g, k) for n in names}
    f1 = {n: float(_macro_f1(*(c.sum(0) for c in counts[n]))) for n in names}
    record = {f"ptrain_f1/{n}": f1[n] for n in names}
    if rule == "ptrain_best":
        order = sorted(
            range(len(names)), key=lambda i: (-round(f1[names[i]], 12), size_bytes[names[i]], i)
        )
        return names[order[0]], record
    if rule != "ptrain_bootstrap":
        raise ValueError(f"unknown rule {rule!r}")
    inc = names[0]
    n_groups = len(counts[inc][0])
    rng = np.random.default_rng([seed, 28, 2])
    draw = rng.integers(0, n_groups, size=(F_RAW_BOOTSTRAP, n_groups))
    w = np.zeros((F_RAW_BOOTSTRAP, n_groups))
    np.add.at(w, (np.arange(F_RAW_BOOTSTRAP)[:, None], draw), 1.0)
    boot = {n: _macro_f1(*(w @ c for c in counts[n])) for n in names}
    best, best_f1 = inc, f1[inc]
    for n in names[1:]:
        lb = float(np.quantile(boot[n] - boot[inc], 0.025))
        record[f"boot_lb/{n}"] = lb
        if lb > 0 and f1[n] > best_f1:
            best, best_f1 = n, f1[n]
    return best, record


def choose_f_raw(
    cfg: FoldConfig,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    ptrain: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    build: Callable[..., Candidate] = train_candidate,
) -> tuple[Candidate, dict[str, float]]:
    """Train the candidates on `train` and pick one by `ptrain`. Never sees calib or test."""
    names = cfg.f_raw_candidates or (cfg.f_raw,)
    if len(names) == 1:
        return build(names[0], *train, cfg, device), {}
    x_pt, y_pt, g_pt = ptrain
    built: dict[str, Candidate] = {}
    pred: dict[str, np.ndarray] = {}
    for n in names:
        built[n] = build(n, *train, cfg, device)
        pred[n] = tr.predict_on(built[n].model, x_pt.astype(np.float32) / 32768.0, device).argmax(1)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    sizes = {n: len(c.onnx) for n, c in built.items()}
    pick, record = select_f_raw(cfg.f_raw_rule, names, y_pt, g_pt, pred, sizes, cfg.seed)
    return built[pick], record


# --- one fold ------------------------------------------------------------------------------------


def _learned_rep(
    rep_id: str,
    x: np.ndarray,
    y: np.ndarray,
    cfg: FoldConfig,
    device: torch.device,
    f_raw: nn.Module,
) -> OnnxLatent | OnnxVQ:
    """Train one learned encoder on training windows and wrap its ONNX export as an edge rep."""
    n = x.shape[1]
    if rep_id == "r5_ae":
        ae = tr.train_autoencoder(x, cfg.ae_dim, cfg.epochs_latent, cfg.seed, device)
        rng = np.random.default_rng([cfg.seed, 5])  # the fold's own Generator, not loader order
        sample = np.sort(rng.choice(len(x), size=min(len(x), R5_SCALE_SAMPLE), replace=False))
        z = tr.predict(ae.encoder, x[sample].astype(np.float32) / 32768.0)
        scale = np.maximum(np.quantile(np.abs(z), 0.999, axis=0), 1e-6) / 127.0
        return OnnxLatent(rep_id, n, tr.export_onnx(ae.encoder, n), scale)
    if rep_id == "r6_bottleneck":
        if cfg.bottleneck == "teacher_front":
            if not isinstance(f_raw, SpecNet):
                raise ValueError(
                    "bottleneck=teacher_front needs a reference model with a front-end"
                )
            encoder = tr.train_front_bottleneck(
                f_raw, x, cfg.bottleneck_dim, cfg.epochs_latent, cfg.seed, device
            )
        else:
            model = tr.train_bottleneck(
                x, y, cfg.bottleneck_dim, cfg.epochs_latent, cfg.seed, device
            )
            encoder = nn.Sequential(model.encoder, nn.Tanh())
        # tanh output ∈ (-1, 1): scale 1/127
        return OnnxLatent(
            rep_id, n, tr.export_onnx(encoder, n), np.full(cfg.bottleneck_dim, 1 / 127.0)
        )
    if rep_id == "r7_vq":
        vq = tr.train_vqvae(x, cfg.vq_code_dim, cfg.epochs_latent, cfg.seed, device).encoder
        return OnnxVQ(rep_id, n, tr.export_onnx(vq, n), vq.codebook.detach().numpy())
    raise ValueError(f"unknown learned representation {rep_id}")


def latent_features(rep: OnnxLatent | OnnxVQ, x: np.ndarray, batch: int = 512) -> np.ndarray:
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
    (out_dir / "metrics.json").unlink(missing_ok=True)  # written last: marks a finished fold
    (out_dir / "fold_config.json").write_text(json.dumps(cfg.recipe(), indent=1))
    device = tr.resolve_device(cfg.device)
    cache = classical_outputs(ws, cache_dir, extra_wavelets=cfg.extra_wavelets, logmel=cfg.logmel)
    part = np.full(len(ws), -1, np.int8)  # -1: in no partition (L2 drops the held-out condition)
    for i, p in enumerate(PARTITIONS):
        part[partition_mask(ws, fold, p)] = i
    is_train = part == PARTITIONS.index("train")
    x_tr, y_tr, g_tr = ws.x[is_train], ws.label[is_train], ws.bearing_id[is_train]

    t0 = time.perf_counter()

    def stage(name: str) -> None:
        print(f"  [{out_dir.name}] {name}: {time.perf_counter() - t0:.0f}s", flush=True)

    is_pt = part == PARTITIONS.index("ptrain")
    chosen, f_raw_rec = choose_f_raw(
        cfg,
        (x_tr, y_tr, g_tr),
        (ws.x[is_pt], ws.label[is_pt], ws.bearing_id[is_pt]),
        device,
    )
    f_raw, f_raw_onnx = chosen.model, chosen.onnx
    stage(f"f_raw trained ({chosen.name})")
    (out_dir / "f_raw.onnx").write_bytes(f_raw_onnx)
    # decisions come from the exported model: what is certified is exactly what the cloud runs
    session = ort.InferenceSession(f_raw_onnx, providers=["CPUExecutionProvider"])
    on_gpu = isinstance(f_raw, W2VClassifier)  # 700 ms/clip on one CPU thread: the cloud has a GPU
    # wav2vec2 inference runs on the GPU like its training, whatever the fold's --device
    f_dev = tr.resolve_device("auto") if on_gpu else device
    f_raw_meta: dict[str, object] = {}
    if on_gpu:
        probe_x = ws.x[:32].astype(np.float32) / 32768.0
        flips = int(
            (
                tr.predict_on(f_raw, probe_x, f_dev).argmax(1)
                != session.run(None, {"x": probe_x[:, None, :]})[0].argmax(1)
            ).sum()
        )
        f_raw_rec["onnx_argmax_flips_of_32"] = float(flips)

    def raw_decision(x_float: np.ndarray) -> np.ndarray:
        if on_gpu:  # parity-checked against the ONNX export (flips recorded in the metrics)
            return tr.predict_on(f_raw, np.asarray(x_float, np.float32), f_dev).argmax(1)
        logits = session.run(None, {"x": np.asarray(x_float, np.float32)[:, None, :]})[0]
        return logits.argmax(1)

    logmel_hz = ws.sample_rate_hz if cfg.logmel else None
    reps: list[BaseRep] = classical_menu(ws.window_n, cfg.extra_wavelets, logmel_hz)
    yhat: dict[str, np.ndarray] = {}
    nbytes: dict[str, np.ndarray] = {}
    (out_dir / "heads").mkdir(exist_ok=True)

    def fit_head(rep_id: str, feats: np.ndarray, rep: BaseRep | None = None) -> np.ndarray:
        # distillation: the head learns the reference model's decision, the thing being certified
        target = yhat["r0_raw"][is_train]
        head: Head
        if isinstance(rep, LogMel):  # a CNN head on the grid (ADR 0018)
            net = tr.train_logmel_head(
                feats[is_train], target, rep.n_mels, rep.n_frames, cfg.seed, device
            )
            head = OnnxHead(tr.export_onnx(net, feats.shape[1]))
        else:
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
            yhat[rid] = fit_head(rid, np.asarray(cache[f"{rid}.out"]), rep)
        stage(rid)

    for rid in cfg.learned:
        # the task-aware bottleneck is distilled from the reference model (supervised compression)
        learned = _learned_rep(rid, x_tr, yhat["r0_raw"][is_train], cfg, device, f_raw)
        reps.append(learned)
        nbytes[rid] = np.full(len(ws), learned.size_model().mean_bytes, np.int32)
        yhat[rid] = fit_head(rid, latent_features(learned, ws.x))
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
    metrics["device_is_cuda"] = float(device.type == "cuda")  # not in the recipe: same models
    metrics["majority_baseline_macro_f1"] = float(
        f1_score(ws.label[test], np.full(test.sum(), np.bincount(y_tr).argmax()), average="macro")
    )
    metrics.update({f"f_raw_{k}": v for k, v in f_raw_rec.items()})
    if on_gpu and f_dev.type == "cuda":
        ms, mad = tr.time_gpu_ms(f_raw, ws.window_n, f_dev)
        metrics |= {"f_raw_cloud_gpu_ms": ms, "f_raw_cloud_gpu_ms_mad": mad}
    f_raw_meta["f_raw_choice"] = chosen.name
    f_raw_meta["f_raw_candidates"] = list(cfg.f_raw_candidates or (cfg.f_raw,))
    (out_dir / "metrics.json").write_text(json.dumps(metrics | f_raw_meta, indent=1))
    return table, metrics
