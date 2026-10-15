"""Reference-model search: every candidate is scored by bearing-grouped inner CV on `train` only.

Protocol (ADR 0006): for each outer L1 fold, the training bearings are split into inner folds that
hold out one bearing per class; a candidate is fitted on the rest and scored on the held-out
bearings. `ptrain`, `calib` and `test` are never read here. Every candidate tried is appended to
`experiments/model_search/results.jsonl`, losers included.

    uv run --no-sync python scripts/model_search.py features           # cache the feature sets
    uv run --no-sync python scripts/model_search.py linear             # stage A: features x heads
    uv run --no-sync python scripts/model_search.py report             # docs/research/model-search.md
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch
import typer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kai.bench.config import work
from kai.contracts.windows import WindowSet
from kai.data.splits import load_manifest, partition_mask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/model_search"
RESULTS = OUT / "results.jsonl"
SHAFT_HZ = {"N15_M07_F10": 25.0, "N09_M07_F10": 15.0, "N15_M01_F10": 25.0, "N15_M07_F04": 25.0}
app = typer.Typer(add_completion=False)


@app.callback()
def _main() -> None:
    """Reference-model search (see the module docstring)."""


# ---- data -----------------------------------------------------------------------------------


def load(window_mult: int = 1) -> WindowSet:
    """The processed windows; `window_mult` > 1 joins consecutive windows of one recording run."""
    ws = WindowSet.load(work() / "data/processed/pu.npz")
    if window_mult == 1:
        return ws
    key = np.char.add(ws.bearing_id, ws.condition)
    starts = np.arange(0, len(ws) - window_mult + 1, window_mult)
    starts = starts[key[starts] == key[starts + window_mult - 1]]
    x = np.concatenate([ws.x[starts + j] for j in range(window_mult)], axis=1)
    return WindowSet(
        x, ws.window_id[starts], ws.bearing_id[starts], ws.condition[starts], ws.label[starts],
        ws.sample_rate_hz, ws.dataset_id,
    )  # fmt: skip


def outer_folds(n: int = 5) -> list:
    (path,) = (work() / "data/splits").glob("pu-v1_L1_*.json")  # exactly one, or stop
    return load_manifest(path).folds[:n]


def inner_folds(
    bearings: np.ndarray, labels: np.ndarray, seed: int, shuffles: int = 2
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train mask, validation mask): each inner fold holds out ≥ 1 bearing of every class."""
    by_class = {c: sorted(set(bearings[labels == c])) for c in np.unique(labels)}
    k = min(len(b) for b in by_class.values())
    for s in range(shuffles):
        rng = np.random.default_rng([seed, s])
        fold_of = {}
        for ids in by_class.values():
            for j, b in enumerate(rng.permutation(ids)):
                fold_of[b] = j % k
        which = np.array([fold_of[b] for b in bearings])
        for j in range(k):
            yield which != j, which == j


# ---- fixed front-ends (torch; every operation has a conv / matmul form that exports to ONNX) --


def _stft_power(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=x.device)
    s = torch.stft(x, n_fft, hop, window=win, center=False, return_complex=True)
    return s.real**2 + s.imag**2  # [B, bins, frames]


def _log_edges(n_bins: int, n_bands: int, first: int = 1) -> np.ndarray:
    return np.unique(np.round(np.geomspace(first, n_bins, n_bands + 1)).astype(int))


def _pool(power: torch.Tensor, edges: np.ndarray) -> torch.Tensor:
    return torch.stack(
        [power[:, a:b].sum(1) for a, b in zip(edges[:-1], edges[1:], strict=True)], 1
    )


def band_shape(x: torch.Tensor, n_fft: int = 1024, n_bands: int = 24) -> torch.Tensor:
    """Welch log band energies minus their mean (the current SpecNet front-end)."""
    psd = _stft_power(x, n_fft, n_fft // 2).mean(-1)
    lb = torch.log(_pool(psd, _log_edges(psd.shape[1], n_bands)) + 1e-12)
    return lb - lb.mean(1, keepdim=True)


def _envelopes(x: torch.Tensor, n_fft: int, hop: int, carriers: int) -> torch.Tensor:
    """Squared envelope in `carriers` log-spaced carrier bands: [B, carriers, frames]."""
    p = _stft_power(x, n_fft, hop)
    edges = _log_edges(p.shape[1], carriers, first=max(1, p.shape[1] // 64))
    return torch.stack([p[:, a:b].sum(1) for a, b in zip(edges[:-1], edges[1:], strict=True)], 1)


def _mod_at(env: torch.Tensor, freqs_hz: torch.Tensor, rate_hz: float) -> torch.Tensor:
    """|DFT| of the mean-removed envelope at arbitrary frequencies, relative to its mean level.

    `freqs_hz` is [F] or [B, F] (per-window grids, used for order tracking).
    """
    t = torch.arange(env.shape[-1], device=env.device) / rate_hz
    mean = env.mean(-1, keepdim=True)
    e = (env - mean) * torch.hann_window(env.shape[-1], device=env.device)
    ang = 2 * torch.pi * freqs_hz[..., None] * t  # [F, T] or [B, F, T]
    eq = "bkt,ft->bkf" if ang.dim() == 2 else "bkt,bft->bkf"
    re, im = torch.einsum(eq, e, torch.cos(ang)), torch.einsum(eq, e, torch.sin(ang))
    return torch.sqrt(re**2 + im**2 + 1e-20) / (mean * env.shape[-1] / 2 + 1e-12)


def mod_spectrum(
    x: torch.Tensor, n_fft: int = 128, hop: int = 16, carriers: int = 6, f_max: float = 640.0,
    fs: float = 64000.0,
) -> torch.Tensor:  # fmt: skip
    """Log modulation (envelope) spectrum per carrier band, in Hz."""
    env = _envelopes(x, n_fft, hop, carriers)
    rate = fs / hop
    df = rate / env.shape[-1]
    freqs = torch.arange(df, f_max, df, device=x.device)
    return torch.log(_mod_at(env, freqs, rate) + 1e-6).flatten(1)


def order_spectrum(
    x: torch.Tensor, shaft_hz: torch.Tensor, n_fft: int = 128, hop: int = 16, carriers: int = 6,
    max_order: float = 12.0, step: float = 0.25, fs: float = 64000.0,
) -> torch.Tensor:  # fmt: skip
    """Log envelope spectrum sampled at multiples of the shaft frequency (needs the speed)."""
    env = _envelopes(x, n_fft, hop, carriers)
    orders = torch.arange(step, max_order + 1e-9, step, device=x.device)
    return torch.log(_mod_at(env, shaft_hz[:, None] * orders, fs / hop) + 1e-6).flatten(1)


# PU test rig, bearing 6203: 8 balls, ball 6.75 mm, pitch 28.55 mm, contact angle 0
BPFO, BPFI, BSF2 = 3.053, 4.947, 3.994
KIN_ORDERS = (
    1.0, 2.0, BPFO, 2 * BPFO, 3 * BPFO, 4 * BPFO, BPFI, 2 * BPFI, 3 * BPFI, 4 * BPFI,
    BPFI - 1, BPFI + 1, BSF2,
)  # fmt: skip


KIN_ORDERS_H = (
    1.0, 2.0, 3.0, *(k * BPFO for k in range(1, 7)), *(k * BPFI for k in range(1, 7)),
    BPFI - 1, BPFI + 1, 2 * BPFI - 1, 2 * BPFI + 1, BPFO - 1, BPFO + 1, BSF2, 2 * BSF2,
)  # fmt: skip


def kinematic(
    x: torch.Tensor, shaft_hz: torch.Tensor, n_fft: int = 128, hop: int = 16, carriers: int = 6,
    fs: float = 64000.0, orders: tuple[float, ...] = KIN_ORDERS, pool_max: bool = False,
) -> torch.Tensor:  # fmt: skip
    """Envelope-spectrum level at the bearing's characteristic orders over the spectrum's median."""
    env = _envelopes(x, n_fft, hop, carriers)
    grid = torch.arange(0.5, 12.01, 0.25, device=x.device)
    floor = _mod_at(env, shaft_hz[:, None] * grid, fs / hop).median(-1, keepdim=True).values
    at = _mod_at(env, shaft_hz[:, None] * torch.tensor(orders, device=x.device), fs / hop)
    out = torch.log(at / (floor + 1e-9) + 1e-6)
    return out.amax(1) if pool_max else out.flatten(1)


def mod_floor(
    x: torch.Tensor, n_fft: int = 256, hop: int = 32, carriers: int = 6, n_freqs: int = 24,
    f_lo: float = 20.0, f_hi: float = 640.0, fs: float = 64000.0,
    shaft_hz: torch.Tensor | None = None,
) -> torch.Tensor:  # fmt: skip
    """Envelope spectrum on a log-spaced Hz grid, relative to its own median (no bearing physics)."""
    env = _envelopes(x, n_fft, hop, carriers)
    freqs = torch.tensor(np.geomspace(f_lo, f_hi, n_freqs), dtype=torch.float32, device=x.device)
    if shaft_hz is not None:  # order domain: the same grid, scaled to each window's shaft speed
        freqs = freqs * shaft_hz[:, None] / 25.0
    m = _mod_at(env, freqs, fs / hop)
    return torch.log(m / (m.median(-1, keepdim=True).values + 1e-9) + 1e-6).flatten(1)


def kin_blind(x: torch.Tensor, n_fft: int = 256, hop: int = 32, carriers: int = 6) -> torch.Tensor:
    """`kinematic` at both rig speeds (25 Hz and 15 Hz shaft): needs no speed input."""
    return torch.cat(
        [
            kinematic(x, torch.full((len(x),), hz, device=x.device), n_fft, hop, carriers)
            for hz in (25.0, 15.0)
        ],
        1,
    )


def low_spectrum(x: torch.Tensor, n_bins: int = 160) -> torch.Tensor:
    """Log power of the first `n_bins` FFT bins (0–2.5 kHz at N = 4096), mean removed."""
    spec = torch.fft.rfft(x * torch.hann_window(x.shape[1], device=x.device))
    lp = torch.log(spec.real[:, 1 : n_bins + 1] ** 2 + spec.imag[:, 1 : n_bins + 1] ** 2 + 1e-12)
    return lp - lp.mean(1, keepdim=True)


def kurt_bands(
    x: torch.Tensor, n_fft: int = 128, hop: int = 16, carriers: int = 12
) -> torch.Tensor:
    """Spectral kurtosis per carrier band (impulsiveness, level-free) and its crest."""
    env = _envelopes(x, n_fft, hop, carriers)
    m = env.mean(-1, keepdim=True)
    k = ((env / (m + 1e-12)) ** 2).mean(-1)
    crest = env.amax(-1) / (m.squeeze(-1) + 1e-12)
    return torch.cat([torch.log(k), torch.log(crest)], 1)


FEATURES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "band24": lambda x, _: band_shape(x),
    "band48": lambda x, _: band_shape(x, 2048, 48),
    "mod6": lambda x, _: mod_spectrum(x),
    "mod12": lambda x, _: mod_spectrum(x, carriers=12),
    "mod6_fine": lambda x, _: mod_spectrum(x, n_fft=64, hop=8, f_max=1000.0),
    "order6": lambda x, f: order_spectrum(x, f),
    "order12": lambda x, f: order_spectrum(x, f, carriers=12),
    "kurt12": lambda x, _: kurt_bands(x),
    "kin6": lambda x, f: kinematic(x, f),
    "kin12": lambda x, f: kinematic(x, f, carriers=12),
    "lowspec": lambda x, _: low_spectrum(x),
    "kin6_fft256_at25": lambda x, f: kinematic(x, torch.full_like(f, 25.0), n_fft=256, hop=32),
    "kin6_fft256_at15": lambda x, f: kinematic(x, torch.full_like(f, 15.0), n_fft=256, hop=32),
    "modfloor24": lambda x, _: mod_floor(x),
    "modfloor48": lambda x, _: mod_floor(x, n_freqs=48),
    "modfloor16": lambda x, _: mod_floor(x, n_freqs=16),
    "modfloor16_order": lambda x, f: mod_floor(x, n_freqs=16, shaft_hz=f),
    "modfloor24_order": lambda x, f: mod_floor(x, n_freqs=24, shaft_hz=f),
    "modfloor24_c4": lambda x, _: mod_floor(x, carriers=4),
    "modfloor24_c8": lambda x, _: mod_floor(x, carriers=8),
    "modfloor24_fft128": lambda x, _: mod_floor(x, 128, 16),
    "modfloor24_fft512": lambda x, _: mod_floor(x, 512, 64),
    "modfloor24_hi320": lambda x, _: mod_floor(x, f_hi=320.0),
    "kinb_fft256": lambda x, _: kin_blind(x),
    "kinb_fft64": lambda x, _: kin_blind(x, 64, 8),
    "kinb_fft128": lambda x, _: kin_blind(x, 128, 16),
    "kinb_fft256_c3": lambda x, _: kin_blind(x, carriers=3),
    "kinb_fft256_c10": lambda x, _: kin_blind(x, carriers=10),
    "kin3": lambda x, f: kinematic(x, f, carriers=3),
    "kinmax": lambda x, f: kinematic(x, f, pool_max=True),
    "kin6h": lambda x, f: kinematic(x, f, orders=KIN_ORDERS_H),
    "kin6_fft256": lambda x, f: kinematic(x, f, n_fft=256, hop=32),
    "kin6_fft64": lambda x, f: kinematic(x, f, n_fft=64, hop=8),
}


def feature_path(name: str, mult: int) -> Path:
    return work() / f"artifacts/model_search/{name}_x{mult}.npy"


def features(name: str, mult: int = 1) -> np.ndarray:
    """One named feature block, or several joined with '+'. Cached on disk."""
    if "+" in name:
        return np.concatenate([features(n, mult) for n in name.split("+")], axis=1)
    path = feature_path(name, mult)
    if not path.exists():
        ws = load(mult)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shaft = torch.tensor([SHAFT_HZ[c] for c in ws.condition], dtype=torch.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(ws), 256):
                xb = torch.from_numpy(ws.x[i : i + 256].astype(np.float32) / 32768.0).to(dev)
                out.append(FEATURES[name](xb, shaft[i : i + 256].to(dev)).cpu().numpy())
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.concatenate(out).astype(np.float32))
    return np.load(path)


# ---- heads ----------------------------------------------------------------------------------
# Full-batch torch on the GPU: sklearn's solvers took minutes per candidate on 14k x 300 inputs.

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchHead:
    """Standardise → (optional hidden layer) → softmax, class-balanced, L2 `weight_decay`.

    `group_dro` > 0 replaces the mean loss by a soft maximum over training bearings (temperature
    1/group_dro), which penalises solutions that fit some bearings and ignore others.
    """

    def __init__(
        self, l2: float, hidden: int = 0, group_dro: float = 0.0, dropout: float = 0.0,
        steps: int = 300, seed: int = 0,
    ) -> None:  # fmt: skip
        self.l2, self.hidden, self.group_dro, self.dropout = l2, hidden, group_dro, dropout
        self.steps, self.seed = steps, seed

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> TorchHead:
        torch.manual_seed(self.seed)
        xt = torch.from_numpy(x).float().to(DEV)
        self.mean, self.scale = xt.mean(0), xt.std(0).clamp_min(1e-6)
        xt = (xt - self.mean) / self.scale
        yt = torch.from_numpy(y).long().to(DEV)
        counts = torch.bincount(yt, minlength=3).float()
        w = (counts.sum() / (3 * counts.clamp_min(1)))[yt]
        gid = torch.from_numpy(np.unique(groups, return_inverse=True)[1]).to(DEV)
        n_groups = int(gid.max()) + 1
        d = xt.shape[1]
        layers: list[torch.nn.Module] = []
        if self.hidden:
            layers += [
                torch.nn.Linear(d, self.hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
            ]
            d = self.hidden
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 3)).to(DEV)
        opt = torch.optim.Adam(self.net.parameters(), lr=0.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        self.net.train()
        for _ in range(self.steps):
            losses = torch.nn.functional.cross_entropy(self.net(xt), yt, reduction="none") * w
            if self.group_dro > 0:
                per_group = torch.zeros(n_groups, device=DEV).index_add_(0, gid, losses)
                per_group = per_group / torch.bincount(gid, minlength=n_groups).clamp_min(1)
                loss = (torch.softmax(self.group_dro * per_group.detach(), 0) * per_group).sum()
            else:
                loss = losses.mean()
            loss = loss + self.l2 * sum((p**2).sum() for p in self.net.parameters() if p.dim() > 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        xt = (torch.from_numpy(x).float().to(DEV) - self.mean) / self.scale
        return self.net(xt).argmax(1).cpu().numpy()


class TwoStageHead:
    """Healthy versus damaged first, then inner versus outer on the damaged windows."""

    def __init__(self, l2: float) -> None:
        self.l2 = l2

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> TwoStageHead:
        self.damaged = TorchHead(self.l2).fit(x, (y > 0).astype(np.int64), groups)
        d = y > 0
        self.which = TorchHead(self.l2).fit(x[d], y[d] - 1, groups[d])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.damaged.predict(x) == 1, self.which.predict(x) + 1, 0)


class BaggedHead:
    """Average of linear heads, each fitted on a bootstrap sample of the training *bearings*."""

    def __init__(self, l2: float, n_bags: int = 15, seed: int = 0) -> None:
        self.l2, self.n_bags, self.seed = l2, n_bags, seed

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> BaggedHead:
        rng = np.random.default_rng(self.seed)
        by_class = {c: np.unique(groups[y == c]) for c in np.unique(y)}
        self.members = []
        for _ in range(self.n_bags):
            picked = np.concatenate([rng.choice(b, len(b)) for b in by_class.values()])
            idx = np.concatenate([np.flatnonzero(groups == b) for b in picked])
            self.members.append(TorchHead(self.l2).fit(x[idx], y[idx], groups[idx]))
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(x).float().to(DEV)
        probs = sum(torch.softmax(m.net((xt - m.mean) / m.scale), 1) for m in self.members)
        return probs.argmax(1).cpu().numpy()  # type: ignore[union-attr]


class SkHead:
    def __init__(self, model: object) -> None:
        self.model = model

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> SkHead:
        if callable(self.model):  # a factory that needs the sample count (C = 1 / (2 n penalty))
            self.model = self.model(len(x))
        self.model.fit(x, y)  # type: ignore[attr-defined]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(x))  # type: ignore[attr-defined]


HEADS: dict[str, Callable[[], TorchHead | SkHead | TwoStageHead | BaggedHead]] = {
    **{
        f"lin_l2_{l2:g}": (lambda l2=l2: TorchHead(l2)) for l2 in (1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0)
    },
    **{f"lin_dro_l2_{l2:g}": (lambda l2=l2: TorchHead(l2, group_dro=5.0)) for l2 in (1e-3, 1e-2)},
    **{
        f"mlp{h}_l2_{l2:g}": (lambda h=h, l2=l2: TorchHead(l2, hidden=h, dropout=0.3))
        for h in (16, 64)
        for l2 in (1e-3, 1e-2)
    },
    "two_stage_l2_0.3": lambda: TwoStageHead(0.3),
    "bagged15_l2_0.3": lambda: BaggedHead(0.3),
    "mlp64_dro_l2_0.01": lambda: TorchHead(1e-2, hidden=64, dropout=0.3, group_dro=5.0),
    **{
        # the production trainer's solver and parametrisation (kai.tasks.train.train_envnet)
        f"sk_logreg_pen_{pen:g}": (
            lambda pen=pen: SkHead(
                lambda n: make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1 / (2 * n * pen), max_iter=3000, class_weight="balanced"),
                )
            )
        )
        for pen in (0.1, 0.3, 1.0)
    },
    "lda_shrink": lambda: SkHead(
        make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    ),
}


# ---- evaluation -----------------------------------------------------------------------------


def record(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def inner_cv(
    feats: np.ndarray,
    ws: WindowSet,
    make: Callable[[], TorchHead | SkHead | TwoStageHead | BaggedHead],
    n_outer: int = 5,
) -> dict[str, float | list[float]]:
    """Macro-F1 on held-out training bearings, per outer fold (mean over its inner folds)."""
    per_fold, per_inner = [], []
    for fold in outer_folds(n_outer):
        m = partition_mask(ws, fold, "train")
        x, y, g = feats[m], ws.label[m], ws.bearing_id[m]
        scores = []
        for tr, va in inner_folds(g, y, seed=fold.fold):
            model = make().fit(x[tr], y[tr], g[tr])
            scores.append(float(f1_score(y[va], model.predict(x[va]), average="macro")))
        per_fold.append(float(np.mean(scores)))
        per_inner += scores
    return {
        "inner_f1_mean": float(np.mean(per_fold)),
        "inner_f1_worst_fold": float(np.min(per_fold)),
        "inner_f1_sd_inner": float(np.std(per_inner)),
        "inner_f1_per_fold": per_fold,
    }


@app.command()
def linear(
    feature_sets: str = "band24,band48,mod6,mod12,mod6_fine,order6,order12,kurt12,"
    "band24+mod6,band24+kurt12,band24+mod6+kurt12,band24+order6,band24+order6+kurt12",
    heads: str = ",".join(HEADS),
    mult: int = 1,
    stage: str = "A",
) -> None:
    """Stage A: fixed front-ends × linear heads."""
    ws = load(mult)
    for fs in feature_sets.split(","):
        feats = features(fs, mult)
        for head in heads.split(","):
            t0 = time.perf_counter()
            res = inner_cv(feats, ws, HEADS[head])
            row = {
                "stage": stage, "features": fs, "head": head, "window_n": ws.window_n,
                "n_features": int(feats.shape[1]), "needs_speed": ("order" in fs or "kin" in fs) and "kinb" not in fs and "_at" not in fs, **res,
                "seconds": round(time.perf_counter() - t0, 1),
            }  # fmt: skip
            record(row)
            typer.echo(
                f"{fs:28s} {head:14s} N={ws.window_n:<6d} mean {res['inner_f1_mean']:.3f} "
                f"worst {res['inner_f1_worst_fold']:.3f}  ({row['seconds']}s)"
            )


# ---- stage B: trained networks on physics-informed inputs --------------------------------------


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):  # type: ignore[no-untyped-def]
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[no-untyped-def]
        return -ctx.weight * grad, None


def _block(c_in: int, c_out: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, c_out, 3, padding=1), torch.nn.BatchNorm2d(c_out), torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
    )  # fmt: skip


class ModCNN(torch.nn.Module):
    """Log-STFT with the per-frequency time mean removed → small 2-D CNN → class logits.

    Removing each bin's mean over time deletes the stationary spectral shape (which identifies the
    bearing) and keeps only how the spectrum is modulated in time. `adversary` adds a bearing-ID
    head behind a gradient reversal (DANN).
    """

    def __init__(self, n_bearings: int, width: int = 16, n_fft: int = 256, hop: int = 32) -> None:
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.body = torch.nn.Sequential(
            _block(1, width), _block(width, 2 * width), _block(2 * width, 4 * width),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Dropout(0.3),
        )  # fmt: skip
        self.cls = torch.nn.Linear(4 * width, 3)
        self.adv = torch.nn.Linear(4 * width, n_bearings)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        lp = torch.log(_stft_power(x, self.n_fft, self.hop) + 1e-10)
        return self.body((lp - lp.mean(-1, keepdim=True)).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.features(x))


class EnvCNN(torch.nn.Module):
    """1-D CNN along modulation frequency over the floor-normalised envelope spectrum."""

    def __init__(self, n_bearings: int, carriers: int = 6, width: int = 32) -> None:
        super().__init__()
        self.carriers = carriers
        self.body = torch.nn.Sequential(
            torch.nn.Conv1d(carriers, width, 5, padding=2), torch.nn.BatchNorm1d(width),
            torch.nn.ReLU(), torch.nn.Conv1d(width, width, 5, padding=2),
            torch.nn.BatchNorm1d(width), torch.nn.ReLU(), torch.nn.AdaptiveMaxPool1d(1),
            torch.nn.Flatten(), torch.nn.Dropout(0.3),
        )  # fmt: skip
        self.cls = torch.nn.Linear(width, 3)
        self.adv = torch.nn.Linear(width, n_bearings)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        m = mod_floor(x, n_freqs=48, carriers=self.carriers)
        return self.body(m.view(len(x), self.carriers, 48))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.features(x))


NETS: dict[str, Callable[[int], torch.nn.Module]] = {
    "modcnn_w16": lambda nb: ModCNN(nb),
    "modcnn_w8": lambda nb: ModCNN(nb, width=8),
    "envcnn_w32": lambda nb: EnvCNN(nb),
}


def train_net(
    net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, gid: torch.Tensor, epochs: int,
    adversary: float, seed: int,
) -> None:  # fmt: skip
    """AdamW + one-cycle, random circular shifts, class-balanced loss, optional DANN term."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net.to(DEV).train()
    counts = torch.bincount(y, minlength=3).float()
    w = counts.sum() / (3 * counts.clamp_min(1))
    n_batches = len(x) // 128
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * n_batches)
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(len(x))).to(DEV)
        for b in range(n_batches):
            idx = order[b * 128 : (b + 1) * 128]
            xb = torch.roll(x[idx].float() / 32768.0, int(rng.integers(0, x.shape[1])), dims=1)
            feat = net.features(xb)  # type: ignore[operator]
            loss = torch.nn.functional.cross_entropy(net.cls(feat), y[idx], weight=w)  # type: ignore[operator]
            if adversary > 0:
                rev = _GradReverse.apply(feat, adversary)
                loss = loss + torch.nn.functional.cross_entropy(net.adv(rev), gid[idx])  # type: ignore[operator]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()


@app.command()
def deep(nets: str = ",".join(NETS), epochs: int = 12, adversary: str = "0,0.3") -> None:
    """Stage B: small CNNs, with and without a bearing-adversarial term. One inner shuffle."""
    ws = load(1)
    x_all = torch.from_numpy(ws.x).to(DEV)
    for name in nets.split(","):
        for adv in (float(a) for a in adversary.split(",")):
            t0 = time.perf_counter()
            per_fold = []
            for fold in outer_folds():
                m = np.flatnonzero(partition_mask(ws, fold, "train"))
                y, g = ws.label[m], ws.bearing_id[m]
                scores = []
                for tr, va in inner_folds(g, y, seed=fold.fold, shuffles=1):
                    gid = torch.from_numpy(np.unique(g[tr], return_inverse=True)[1]).to(DEV)
                    net = NETS[name](int(gid.max()) + 1)
                    yt = torch.from_numpy(y[tr]).to(DEV)
                    train_net(net, x_all[m[tr]], yt, gid, epochs, adv, seed=fold.fold)
                    with torch.no_grad():
                        xv = x_all[m[va]]
                        pred = torch.cat(
                            [net(xv[i : i + 512].float() / 32768.0).argmax(1) for i in range(0, len(xv), 512)]
                        ).cpu().numpy()  # fmt: skip
                    scores.append(float(f1_score(y[va], pred, average="macro")))
                per_fold.append(float(np.mean(scores)))
            row = {
                "stage": "B", "features": name, "head": f"adversary={adv:g},epochs={epochs}",
                "window_n": ws.window_n, "n_features": sum(p.numel() for p in NETS[name](8).parameters()),
                "needs_speed": False, "inner_f1_mean": float(np.mean(per_fold)),
                "inner_f1_worst_fold": float(np.min(per_fold)), "inner_f1_per_fold": per_fold,
                "inner_shuffles": 1, "seconds": round(time.perf_counter() - t0, 1),
            }  # fmt: skip
            record(row)
            typer.echo(
                f"{name:14s} adv={adv:g} mean {row['inner_f1_mean']:.3f} "
                f"worst {row['inner_f1_worst_fold']:.3f} ({row['seconds']}s)"
            )


LEGEND = {
    "band*": "Welch log band energies minus their mean — the stationary spectral shape (ADR 0004 model is `band24`)",
    "mod*": "log envelope spectrum per carrier band on a linear Hz grid, relative to the mean envelope",
    "order*": "the same, sampled at multiples of the shaft frequency (0.25 … 12 orders)",
    "kin*": "envelope-spectrum level at the 6203's fault orders (BPFO, BPFI, harmonics, sidebands) over the spectrum's median; `kinb_*` evaluates both rig speeds and needs no speed input; `_at25` / `_at15` assume one speed for every window",
    "modfloor*": "envelope spectrum at N log-spaced modulation frequencies (20–640 Hz) over its own median; no bearing geometry, no speed. **`modfloor16` is EnvNet's front-end**",
    "kurt12": "spectral kurtosis and crest factor of 12 carrier-band envelopes",
    "lowspec": "log power of the first 160 FFT bins, mean removed (used for the speed-detection check)",
    "modcnn_*": "2-D CNN on the log-STFT with each frequency bin's time mean removed; `adversary` > 0 adds a bearing-ID head behind a gradient reversal",
    "envcnn_*": "1-D CNN along modulation frequency over the `modfloor48` map",
    "r6_generic_cnn_16B": "the pipeline's original R6: a CNN on the waveform with a 16-value tanh bottleneck, trained to repeat EnvNet (hard decisions or soft probabilities; `augment` = shift, mix, random EQ, resample with the teacher re-labelling each batch)",
    "r6_teacher_front_16B": "EnvNet's fixed front-end followed by a learned 96 → 16 projection, trained on soft targets",
    "suffixes": "`_fftK` STFT length (carrier resolution), `_cK` number of carrier bands, `_hi320` upper modulation frequency, `a+b` concatenation",
    "heads": "`two_stage` healthy-vs-damaged then inner-vs-outer; `bagged15` average of 15 linear heads fitted on bootstrap samples of the training bearings; `lin_l2_p` multinomial logistic regression with mean-loss L2 penalty p; `lin_dro` soft group-DRO over training bearings; `mlpH` one hidden layer of H units with dropout 0.3; `lda_shrink` shrinkage LDA; `logreg_c*` / `sk_logreg_pen_*` scikit-learn solver",
}

STAGES = {
    "A": "Stage A — fixed front-ends × heads, shaft speed given where a front-end needs it",
    "A-blind": "Stage A (speed-blind) — front-ends that need no side information",
    "A-window": "Window-length study — longer windows built by joining consecutive ones",
    "B": "Stage B — trained CNNs (one inner shuffle; `n_features` is the parameter count)",
    "C-production": "Stage C — the chosen front-end with the production solver",
    "D": "Stage D — can a 16-byte latent repeat EnvNet? **Disagreement** with EnvNet on held-out training bearings, lower is better (first inner fold of each outer fold)",
    "E": "Stage E — multi-label training: two detectors (inner? outer?), with and without the three combined-damage bearings on the fitting side",
    "T-test": "Test partitions — read once, after the choice was frozen in ADR 0006",
}


@app.command()
def report(out: Path = ROOT / "docs/research/model-search.md") -> None:
    """Write every recorded candidate, losers included, as Markdown tables."""
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    latest = {(r["stage"], r["features"], r["head"], r["window_n"]): r for r in rows}
    lines = [
        "# Reference-model search — every candidate tried",
        "",
        "Generated by `scripts/model_search.py report` from `experiments/model_search/results.jsonl`;",
        "do not edit by hand. Protocol and the decision are in",
        "[ADR 0006](../adr/0006-envelope-reference-model.md).",
        "",
        "Scores are **macro-F1 on held-out training bearings** (bearing-grouped inner CV inside the",
        "`train` partition of each of the five L1 folds). They are label scores of the reference",
        "model, not decision agreement, and they are not test results.",
        f"{len(rows)} runs recorded, {len(latest)} distinct candidates.",
    ]
    lines += ["", "## Names", "", "| Name | Meaning |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in LEGEND.items()]
    for stage, title in STAGES.items():
        key = {"T-test": "test_f1", "D": "disagreement"}.get(stage, "inner_f1")
        sign = 1 if key == "disagreement" else -1  # disagreement: lower is better
        chosen = sorted(
            (r for r in latest.values() if r["stage"] == stage),
            key=lambda r, key=key, sign=sign: sign * r[f"{key}_mean"],
        )
        if not chosen:
            continue
        lines += [
            "", f"## {title}", "",
            "| Front-end | Head | Window N | Features | Needs speed | Mean | Worst fold | Per fold |",
            "|---|---|---|---|---|---|---|---|",
        ]  # fmt: skip
        for r in chosen:
            per_fold = " ".join(f"{v:.2f}" for v in r[f"{key}_per_fold"])
            lines.append(
                f"| `{r['features']}` | `{r['head']}` | {r['window_n']} | {r['n_features']} "
                f"| {'yes' if r['needs_speed'] else 'no'} | **{r[f'{key}_mean']:.3f}** "
                f"| {r[f'{key}_worst_fold']:.3f} | {per_fold} |"
            )
    out.write_text("\n".join(lines) + "\n")
    typer.echo(f"wrote {out} ({len(latest)} candidates)")


# ---- stage D: can a 16-byte latent repeat EnvNet's decision on unseen bearings? -----------------


def _augment(x: torch.Tensor) -> torch.Tensor:
    """New 'bearings' from training windows: shift, mix two windows, smooth random EQ, resample."""
    n = x.shape[1]
    x = torch.roll(x, int(torch.randint(0, n, (1,))), dims=1)
    w = torch.rand(len(x), 1, device=x.device) * 0.7
    x = (1 - w) * x + w * x[torch.randperm(len(x), device=x.device)]
    spec = torch.fft.rfft(x)
    knots = torch.randn(len(x), 1, 12, device=x.device) * 0.7
    gain = torch.nn.functional.interpolate(knots, spec.shape[1], mode="linear", align_corners=True)
    x = torch.fft.irfft(spec * torch.exp(gain[:, 0]), n=n)
    rate = float(torch.empty(1).uniform_(0.75, 1.35))
    y = torch.nn.functional.interpolate(x[:, None], scale_factor=rate, mode="linear")[:, 0]
    return y[:, :n] if y.shape[1] >= n else torch.cat([y, y[:, : n - y.shape[1]]], 1)


class FrontBottleneck(torch.nn.Module):
    """Supervised compression in the split-computing sense: the teacher's own (fixed) front-end,
    then a learned `dim`-value bottleneck. The edge would have to run the teacher's front-end."""

    def __init__(self, teacher: torch.nn.Module, dim: int = 16) -> None:
        super().__init__()
        self.front, self.mean, self.scale = teacher.front, teacher.mean, teacher.scale
        self.down = torch.nn.Linear(teacher.front.n_features, dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)
        )

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.down((self.front(x) - self.mean) / self.scale))


@app.command()
def distill(epochs: str = "15,60", dim: int = 16) -> None:
    """Stage D: disagreement of a `dim`-byte int8 latent with EnvNet on held-out training bearings."""
    from kai.tasks import train as tr
    from kai.tasks.models import Bottleneck

    ws = load(1)
    recipes = [("generic_cnn", "hard", False), ("generic_cnn", "soft", False),
               ("generic_cnn", "soft", True), ("teacher_front", "soft", False)]  # fmt: skip
    for n_epochs in (int(e) for e in epochs.split(",")):
        for arch, target, augment in recipes:
            if n_epochs > 15 and not augment:
                continue  # the long schedule is only tried where there is fresh data every epoch
            t0, per_fold = time.perf_counter(), []
            for fold in outer_folds():
                m = np.flatnonzero(partition_mask(ws, fold, "train"))
                y, g = ws.label[m], ws.bearing_id[m]
                tr_mask, va_mask = next(inner_folds(g, y, seed=fold.fold, shuffles=1))
                teacher = tr.train_envnet(ws.x[m[tr_mask]], y[tr_mask], g[tr_mask], seed=0).to(DEV)
                for p in teacher.parameters():
                    p.requires_grad_(False)
                torch.manual_seed(fold.fold)
                student = (
                    FrontBottleneck(teacher, dim)
                    if arch == "teacher_front"
                    else Bottleneck(4096, dim)
                ).to(DEV)
                x_tr = torch.from_numpy(ws.x[m[tr_mask]]).to(DEV)
                n_batches = len(x_tr) // 128
                params = [p for p in student.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.OneCycleLR(
                    opt, 2e-3, total_steps=n_epochs * n_batches
                )
                student.train()
                for _ in range(n_epochs):
                    order = torch.randperm(len(x_tr), device=DEV)
                    for b in range(n_batches):
                        xb = x_tr[order[b * 128 : (b + 1) * 128]].float() / 32768.0
                        if augment:
                            xb = _augment(xb)
                        with torch.no_grad():
                            logits_t = teacher(xb[:, None])
                        if arch == "teacher_front":
                            z = student.latent(xb[:, None])
                            out = student.head(z + (torch.rand_like(z) - 0.5) / 127.0)
                        else:
                            out = student(xb[:, None])
                        if target == "soft":
                            loss = torch.nn.functional.kl_div(
                                torch.log_softmax(out, 1),
                                torch.softmax(logits_t, 1),
                                reduction="batchmean",
                            )
                        else:
                            loss = torch.nn.functional.cross_entropy(out, logits_t.argmax(1))
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        sched.step()
                student.eval()
                with torch.no_grad():
                    disagree = []
                    xv = torch.from_numpy(ws.x[m[va_mask]]).to(DEV)
                    for i in range(0, len(xv), 512):
                        xb = xv[i : i + 512].float()[:, None] / 32768.0
                        z = (
                            student.latent(xb)
                            if arch == "teacher_front"
                            else torch.tanh(student.encoder(xb))
                        )
                        pred = student.head(torch.round(z * 127) / 127).argmax(
                            1
                        )  # int8 on the wire
                        disagree.append((pred != teacher(xb).argmax(1)).float())
                per_fold.append(float(torch.cat(disagree).mean()))
            row = {
                "stage": "D", "features": f"r6_{arch}_{dim}B", "window_n": 4096,
                "head": f"target={target},augment={augment},epochs={n_epochs}", "n_features": dim,
                "needs_speed": False, "disagreement_mean": float(np.mean(per_fold)),
                "disagreement_worst_fold": float(np.max(per_fold)), "disagreement_per_fold": per_fold,
                "seconds": round(time.perf_counter() - t0, 1),
            }  # fmt: skip
            record(row)
            typer.echo(
                f"{row['features']:26s} {row['head']:40s} disagreement {row['disagreement_mean']:.3f} "
                f"worst {row['disagreement_worst_fold']:.3f} ({row['seconds']}s)"
            )


# ---- stage E: multi-label training with the combined-damage bearings --------------------------

COMBINED = ("KB23", "KB24", "KB27")  # inner and outer race damaged: no 3-class label


def combined_features(name: str) -> np.ndarray:
    """Front-end features of the combined-damage bearings (five recordings per condition)."""
    from kai.data import pu

    path = work() / f"artifacts/model_search/{name}_combined.npy"
    if not path.exists():
        raw, interim = work() / "data/raw/pu", work() / "data/interim/pu"
        out = []
        for bearing in COMBINED:
            pu.extract(raw / f"{bearing}.rar", interim, range(1, 6))
            for cond in pu.CONDITIONS:
                for i in range(1, 6):
                    signal = pu.load_vibration(interim / bearing / f"{cond}_{bearing}_{i}.mat")
                    x = pu.segment(pu.quantise(signal, 16.0)[0], 4096).astype(np.float32) / 32768.0
                    with torch.no_grad():
                        out.append(
                            FEATURES[name](torch.from_numpy(x).to(DEV), torch.zeros(1)).cpu()
                        )
        np.save(path, torch.cat(out).numpy().astype(np.float32))
    return np.load(path)


@app.command()
def multilabel(feature_set: str = "modfloor16", l2: float = 0.3) -> None:
    """Stage E: two detectors (inner damaged? outer damaged?), with and without KB23/24/27.

    Validation is on held-out training bearings with a single damage, as everywhere else; the
    combined-damage bearings only ever join the fitting side.
    """
    ws = load(1)
    feats, extra = features(feature_set), combined_features(feature_set)
    for with_combined in (False, True):
        per_fold = []
        for fold in outer_folds():
            m = partition_mask(ws, fold, "train")
            x, y, g = feats[m], ws.label[m], ws.bearing_id[m]
            scores = []
            for tr, va in inner_folds(g, y, seed=fold.fold):
                xa, ya, ga = x[tr], y[tr], g[tr]
                inner, outer = (ya == 1).astype(np.int64), (ya == 2).astype(np.int64)
                if with_combined:
                    ones = np.ones(len(extra), np.int64)
                    xa = np.concatenate([xa, extra])
                    inner, outer = np.concatenate([inner, ones]), np.concatenate([outer, ones])
                    ga = np.concatenate([ga, np.full(len(extra), "KB")])
                p_in = TorchHead(l2).fit(xa, inner, ga)
                p_out = TorchHead(l2).fit(xa, outer, ga)
                with torch.no_grad():
                    xv = torch.from_numpy(x[va]).float().to(DEV)
                    probs = [
                        torch.softmax(h.net((xv - h.mean) / h.scale), 1)[:, 1]
                        for h in (p_in, p_out)
                    ]
                both = torch.stack(probs, 1)
                pred = torch.where(both.amax(1) < 0.5, 0, both.argmax(1) + 1).cpu().numpy()
                scores.append(float(f1_score(y[va], pred, average="macro")))
            per_fold.append(float(np.mean(scores)))
        row = {
            "stage": "E", "features": feature_set, "window_n": 4096, "n_features": int(feats.shape[1]),
            "head": f"two_detectors_l2_{l2:g},combined_bearings={with_combined}", "needs_speed": False,
            "inner_f1_mean": float(np.mean(per_fold)), "inner_f1_worst_fold": float(np.min(per_fold)),
            "inner_f1_per_fold": per_fold,
        }  # fmt: skip
        record(row)
        typer.echo(
            f"{row['head']:55s} mean {row['inner_f1_mean']:.3f} worst {row['inner_f1_worst_fold']:.3f}"
        )


@app.command()
def diagnose(feature_sets: str = "modfloor16,band24+mod6,kinb_fft256", mult: int = 1) -> None:
    """Why the score is what it is: per-bearing votes, binary task, speed detection, leaky split.

    The "same bearings" number splits one bearing's windows between fit and validation. It is a
    diagnostic of what random-window evaluation would report, never a result.
    """
    ws = load(mult)
    out = {}
    for name in feature_sets.split(","):
        feats = features(name, mult)
        votes, binary, speed, same, per_cond = [], [], [], [], {}
        for fold in outer_folds():
            m = partition_mask(ws, fold, "train")
            x, y, g, c = feats[m], ws.label[m], ws.bearing_id[m], ws.condition[m]
            slow = (c == "N09_M07_F10").astype(np.int64)
            damaged = (y > 0).astype(np.int64)
            for tr, va in inner_folds(g, y, seed=fold.fold, shuffles=1):
                pred = TorchHead(0.3).fit(x[tr], y[tr], g[tr]).predict(x[va])
                for cond in np.unique(c):
                    k = c[va] == cond
                    per_cond.setdefault(str(cond), []).append(
                        float(f1_score(y[va][k], pred[k], average="macro"))
                    )
                votes += [
                    bool(
                        np.bincount(pred[g[va] == b], minlength=3).argmax() == y[va][g[va] == b][0]
                    )
                    for b in np.unique(g[va])
                ]
                p2 = TorchHead(0.3).fit(x[tr], damaged[tr], g[tr]).predict(x[va])
                binary.append(float(f1_score(damaged[va], p2, average="macro")))
                p3 = TorchHead(0.01).fit(x[tr], slow[tr], g[tr]).predict(x[va])
                speed.append(float((p3 == slow[va]).mean()))
            idx = np.random.default_rng(fold.fold).permutation(len(x))
            a, b = idx[: len(x) // 2], idx[len(x) // 2 :]
            pred = TorchHead(1e-3, hidden=64).fit(x[a], y[a], g[a]).predict(x[b])
            same.append(float(f1_score(y[b], pred, average="macro")))
        out[name] = {
            "window_n": ws.window_n,
            "bearing_majority_vote_accuracy": float(np.mean(votes)),
            "n_bearing_votes": len(votes),
            "binary_healthy_vs_damaged_macro_f1": float(np.mean(binary)),
            "speed_900_vs_1500_accuracy": float(np.mean(speed)),
            "per_condition_macro_f1": {k: float(np.mean(v)) for k, v in per_cond.items()},
            "same_bearings_macro_f1_diagnostic_only": float(np.mean(same)),
        }
        typer.echo(f"{name}: {json.dumps(out[name])}")
    path = OUT / f"diagnostics_n{ws.window_n}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(old | out, indent=1) + "\n")


@app.command()
def record_test(folds_dir: Path, features: str, head: str) -> None:
    """Append the test macro-F1 of `f_raw` from trained L1 folds (`metrics.json`) as a T-test row."""
    f1 = [
        json.loads((d / "metrics.json").read_text())["macro_f1/r0_raw"]
        for d in sorted(folds_dir.glob("s0_r0f*"))
    ]
    record(
        {
            "stage": "T-test", "features": features, "head": head, "window_n": 4096,
            "n_features": 0, "needs_speed": False, "test_f1_mean": float(np.mean(f1)),
            "test_f1_worst_fold": float(np.min(f1)), "test_f1_per_fold": f1,
        }
    )  # fmt: skip
    typer.echo(
        f"{features}: test macro-F1 mean {np.mean(f1):.3f} worst {np.min(f1):.3f} {np.round(f1, 3)}"
    )


if __name__ == "__main__":
    app()
