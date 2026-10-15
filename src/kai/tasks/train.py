"""Deterministic training loops and ONNX export (lab plane only)."""

from __future__ import annotations

import copy
import io
import warnings
from collections.abc import Callable

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn

from kai.tasks.models import VQVAE, AutoEncoder, Bottleneck, EnvNet, KinNet, RawNet, SpecNet

ONNX_OPSET = 17
Step = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def resolve_device(device: str = "auto") -> torch.device:
    """`auto` picks CUDA when a GPU build of torch is installed, else CPU (the PRD default)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda requested but this torch build has no CUDA (see README: GPU)"
        )
    return torch.device(device)


def to_tensor(x_int16: np.ndarray) -> torch.Tensor:
    """int16 windows [n, N] → float32 [n, 1, N] in [-1, 1): the input convention of every model."""
    return torch.from_numpy(x_int16.astype(np.float32) / 32768.0).unsqueeze(1)


def _fit(
    model: nn.Module,
    step: Step,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
    batch: int = 128,
    lr: float = 2e-3,
) -> list[float]:
    """AdamW + one-cycle schedule; batch order comes from a seeded numpy Generator."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng([seed, 1])
    model.to(device).train()
    xt, yt = to_tensor(x), torch.from_numpy(y.astype(np.int64))
    batch = min(batch, max(8, len(xt) // 8))  # tiny datasets still get ≥ 8 steps per epoch
    n_batches = max(1, len(xt) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * n_batches)
    history = []
    for _ in range(epochs):
        order, total = rng.permutation(len(xt)), 0.0
        for b in range(n_batches):
            idx = torch.from_numpy(order[b * batch : (b + 1) * batch])
            loss = step(xt[idx].to(device), yt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        history.append(total / n_batches)
    model.eval().cpu()
    return history


def _spectral_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vibration is noise-like, so waveform MSE alone teaches little; match the log-spectrum too."""
    fx, fy = torch.fft.rfft(x.squeeze(1)).abs(), torch.fft.rfft(y.squeeze(1)).abs()
    return F.l1_loss(torch.log1p(fy), torch.log1p(fx))


def _class_weights(y: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    return torch.from_numpy(counts.sum() / (n_classes * np.maximum(counts, 1))).to(device)


def train_rawnet(
    x: np.ndarray, y: np.ndarray, epochs: int, seed: int, device: torch.device
) -> RawNet:
    torch.manual_seed(seed)
    model = RawNet()
    w = _class_weights(y, 3, device)
    _fit(model, lambda xb, yb: F.cross_entropy(model(xb), yb, weight=w), x, y, epochs, seed, device)
    return model


def _fit_front_linear(
    model: SpecNet,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    cs: tuple[float, ...],
) -> SpecNet:
    """Fit standardisation + multinomial logistic regression on a model's fixed front-end.

    The only hyper-parameter, the regularisation strength, is chosen from `cs` by bearing-grouped
    inner CV on the training bearings. Deterministic and CPU-only; the fitted weights are copied
    into the torch module so that the model exports to ONNX like every other one.
    """
    feats = predict(model.front, x.astype(np.float32) / 32768.0).astype(np.float64)
    mean, scale = feats.mean(0), np.maximum(feats.std(0), 1e-9)
    z = (feats - mean) / scale

    def make(c: float) -> LogisticRegression:
        return LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=seed)

    folds = list(GroupKFold(min(3, len(np.unique(groups)))).split(z, y, groups))
    scores = {
        c: np.mean(
            [
                f1_score(y[va], make(c).fit(z[tr], y[tr]).predict(z[va]), average="macro")
                for tr, va in folds
            ]
        )
        for c in cs
    }
    best = make(max(scores, key=scores.__getitem__)).fit(z, y)
    with torch.no_grad():
        model.mean.copy_(torch.from_numpy(mean).float())
        model.scale.copy_(torch.from_numpy(scale).float())
        model.linear.weight.zero_()
        model.linear.bias.fill_(-1e9)  # classes absent from training can never be predicted
        for row, cls in enumerate(best.classes_):
            model.linear.weight[cls] = torch.from_numpy(best.coef_[row]).float()
            model.linear.bias[cls] = float(best.intercept_[row])
    return model.eval()


def train_specnet(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> SpecNet:
    """The spectral-shape reference model (ADR 0004)."""
    return _fit_front_linear(SpecNet(x.shape[1]), x, y, groups, seed, (0.01, 0.1, 1.0))


def train_kinnet(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> KinNet:
    """The fault-order reference model for long windows (ADR 0012); same penalty grid as EnvNet."""
    cs = tuple(1.0 / (2 * len(x) * penalty) for penalty in (0.1, 0.3, 1.0))
    model = _fit_front_linear(KinNet(x.shape[1]), x, y, groups, seed, cs)
    assert isinstance(model, KinNet)
    return model


def train_envnet(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> EnvNet:
    """The envelope-spectrum reference model (ADR 0006).

    The search preferred strong shrinkage: a mean-loss L2 penalty of 0.1–1, which in
    scikit-learn's parametrisation is C = 1 / (2 n penalty).
    """
    # shortcut: C is scaled by the full training size, so the inner CV (fits on about 2/3 of it)
    # scores each penalty about 1.5x stronger than the final fit; the trained folds were made this
    # way, change it only together with retraining them (ADR 0010)
    cs = tuple(1.0 / (2 * len(x) * penalty) for penalty in (0.1, 0.3, 1.0))
    model = _fit_front_linear(EnvNet(x.shape[1]), x, y, groups, seed, cs)
    assert isinstance(model, EnvNet)
    return model


def train_bottleneck(
    x: np.ndarray, y: np.ndarray, dim: int, epochs: int, seed: int, device: torch.device
) -> Bottleneck:
    torch.manual_seed(seed)
    model = Bottleneck(x.shape[1], dim)
    w = _class_weights(y, 3, device)
    _fit(model, lambda xb, yb: F.cross_entropy(model(xb), yb, weight=w), x, y, epochs, seed, device)
    return model


def train_front_bottleneck(
    teacher: SpecNet, x: np.ndarray, dim: int, epochs: int, seed: int, device: torch.device
) -> nn.Sequential:
    """R6 as supervised compression in the split-computing sense (ADR 0006, addendum).

    The edge runs the reference model's own fixed front-end and a learned `dim`-value projection;
    the projection is trained so that a small head on the noisy latent reproduces the teacher's
    class probabilities. Returns the edge encoder `front → Linear → Tanh` with the teacher's
    standardisation folded into the linear layer.
    """
    torch.manual_seed(seed)
    teacher = copy.deepcopy(teacher).to(device).eval()
    down = nn.Linear(teacher.front.n_features, dim)
    head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 3))
    student = nn.ModuleDict({"down": down, "head": head})

    def step(xb: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = (teacher.front(xb) - teacher.mean) / teacher.scale
            target = torch.softmax(teacher.linear(feats), 1)
        z = torch.tanh(down(feats))
        z = z + (torch.rand_like(z) - 0.5) / 127.0  # stand-in for int8 quantisation noise
        return F.kl_div(torch.log_softmax(head(z), 1), target, reduction="batchmean")

    _fit(student, step, x, np.zeros(len(x)), epochs, seed, device)
    teacher.cpu()
    proj = nn.Linear(teacher.front.n_features, dim)
    with torch.no_grad():
        proj.weight.copy_(down.weight / teacher.scale)
        proj.bias.copy_(down.bias - (down.weight * (teacher.mean / teacher.scale)).sum(1))
    return nn.Sequential(teacher.front, proj, nn.Tanh()).eval()


def train_autoencoder(
    x: np.ndarray, dim: int, epochs: int, seed: int, device: torch.device
) -> AutoEncoder:
    torch.manual_seed(seed)
    model = AutoEncoder(x.shape[1], dim)
    scale = float(to_tensor(x[:512]).std()) + 1e-8  # put the waveform loss on a unit scale

    def step(xb: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        out = model(xb)
        return F.mse_loss(out / scale, xb / scale) + _spectral_loss(xb / scale, out / scale)

    _fit(model, step, x, np.zeros(len(x)), epochs, seed, device)
    return model


def train_vqvae(
    x: np.ndarray, code_dim: int, epochs: int, seed: int, device: torch.device
) -> VQVAE:
    torch.manual_seed(seed)
    model = VQVAE(code_dim)
    scale = float(to_tensor(x[:512]).std()) + 1e-8

    def step(xb: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        out, vq_loss = model(xb)
        return (
            F.mse_loss(out / scale, xb / scale) + _spectral_loss(xb / scale, out / scale) + vq_loss
        )

    with torch.no_grad():  # start the codebook on real encoder outputs to avoid dead codes
        z = model.encoder.latents(to_tensor(x[:1024])).reshape(-1, code_dim)
        pick = np.random.default_rng([seed, 2]).choice(len(z), len(model.encoder.codebook))
        model.encoder.codebook.copy_(z[torch.from_numpy(pick)])
    _fit(model, step, x, np.zeros(len(x)), epochs, seed, device)
    return model


@torch.no_grad()
def predict(model: nn.Module, x_float: np.ndarray, batch: int = 512) -> np.ndarray:
    """Run a model on float32 windows [n, N] (already scaled to [-1, 1)) on CPU."""
    model.eval()
    xt = torch.from_numpy(np.array(x_float, dtype=np.float32)).unsqueeze(1)  # copy: may be a memmap
    return np.concatenate([model(xt[i : i + batch]).numpy() for i in range(0, len(xt), batch)])


def export_onnx(model: nn.Module, window_n: int) -> bytes:
    """Export with a pinned opset and a dynamic batch axis; returns the serialised model."""
    model.eval()
    buf = io.BytesIO()
    with warnings.catch_warnings():
        # the legacy TorchScript exporter is deliberate: pinned opset, no onnxscript dependency
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            model,
            (torch.zeros(1, 1, window_n),),
            buf,  # type: ignore[arg-type]  # a file-like object is supported at run time
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
            opset_version=ONNX_OPSET,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load_from_string(buf.getvalue()))
    return buf.getvalue()
