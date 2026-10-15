"""Torch models (lab plane only): raw reference model f_raw and the learned encoders R5–R7."""

from __future__ import annotations

import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DOWNSAMPLE = 256  # total stride of ConvEncoder


def _conv(c_in: int, c_out: int, k: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, stride, padding=k // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _conv(c_in, c_out, 9, stride),
            nn.Conv1d(c_out, c_out, 9, padding=4, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x))


class RawNet(nn.Module):
    """Waveform alternative for f_raw: ResNet1D-small on a float window [B, 1, N] in [-1, 1).

    Kept for comparison: on Paderborn it memorises its training bearings (ADR 0004).
    """

    def __init__(self, n_classes: int = 3, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(1),  # fixed affine at inference: keeps amplitude information
            _conv(1, width, 15, 4),
            ResBlock(width, 2 * width, 4),
            ResBlock(2 * width, 4 * width, 4),
            ResBlock(4 * width, 4 * width, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(4 * width, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _stft_kernels(n_fft: int) -> torch.Tensor:
    """Hann-windowed cosine and sine kernels [2 * bins, 1, n_fft]: an STFT as one strided Conv."""
    bins = n_fft // 2 + 1
    t = torch.arange(n_fft, dtype=torch.float64)
    phase = 2 * torch.pi * torch.arange(bins, dtype=torch.float64).unsqueeze(1) * t / n_fft
    win = torch.hann_window(n_fft, dtype=torch.float64)
    return torch.cat([torch.cos(phase) * win, torch.sin(phase) * win]).unsqueeze(1).float()


class SpecFront(nn.Module):
    """Fixed, parameter-free front-end: Welch power spectrum → log band energies → spectral *shape*.

    Subtracting the mean log energy removes the window's overall level, so the features do not
    carry the per-bearing gain that waveform models latch onto (ADR 0004). The STFT is a strided
    convolution with windowed cosine / sine kernels, which exports to ONNX as plain Conv.
    """

    kernels: torch.Tensor
    pool: torch.Tensor

    def __init__(self, n_fft: int, n_bands: int = 24) -> None:
        super().__init__()
        bins = n_fft // 2 + 1
        kernels = _stft_kernels(n_fft)
        edges = torch.unique(
            torch.round(torch.logspace(0, torch.log10(torch.tensor(float(bins))), n_bands + 1))
        )
        pool = torch.zeros(len(edges) - 1, bins)
        for i, (a, b) in enumerate(zip(edges[:-1].int(), edges[1:].int(), strict=True)):
            pool[i, a:b] = 1.0
        self.register_buffer("kernels", kernels)
        self.register_buffer("pool", pool)
        self.n_fft, self.bins, self.n_features = n_fft, bins, len(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.conv1d(x, self.kernels, stride=self.n_fft // 2)
        psd = (z[:, : self.bins] ** 2 + z[:, self.bins :] ** 2).mean(-1)
        log_bands = torch.log(psd @ self.pool.T + 1e-12)
        return log_bands - log_bands.mean(1, keepdim=True)


class SpecNet(nn.Module):
    """f_raw (default): SpecFront → standardise → linear. Low capacity on purpose (ADR 0004)."""

    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, window_n: int, n_classes: int = 3, n_bands: int = 24) -> None:
        super().__init__()
        self.front = SpecFront(window_n // 4, n_bands)
        self.register_buffer("mean", torch.zeros(self.front.n_features))
        self.register_buffer("scale", torch.ones(self.front.n_features))
        self.linear = nn.Linear(self.front.n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear((self.front(x) - self.mean) / self.scale)


class EnvFront(nn.Module):
    """Fixed front-end: band envelopes → envelope spectrum → level over the spectrum's own median.

    A damaged race is struck periodically, which modulates the resonances it excites. The
    front-end therefore discards the stationary spectrum (which mostly identifies the individual
    bearing, ADR 0006) and keeps how each carrier band's energy is modulated in time:

    1. squared envelope in `carriers` log-spaced bands (STFT power summed over a band);
    2. magnitude of its Fourier transform at `n_freqs` log-spaced modulation frequencies, given in
       cycles per window so that nothing depends on the sample rate (1.28–40.96 cycles is
       20–640 Hz for a 64 ms window);
    3. log of that magnitude over its median across modulation frequencies: a peak-to-floor ratio,
       free of the window's level and of the carrier band's gain.

    Everything is Conv / MatMul / TopK, so it exports to ONNX like the other models.
    """

    kernels: torch.Tensor
    pool: torch.Tensor
    basis: torch.Tensor

    def __init__(self, window_n: int, carriers: int = 6, n_freqs: int = 16) -> None:
        super().__init__()
        n_fft = window_n // 16
        self.hop, self.bins = n_fft // 8, n_fft // 2 + 1
        self.frames = (window_n - n_fft) // self.hop + 1
        edges = torch.unique(
            torch.round(
                torch.logspace(
                    math.log10(max(1, self.bins // 64)), math.log10(self.bins), carriers + 1
                )
            )
        ).int()
        pool = torch.zeros(len(edges) - 1, self.bins)
        for i, (a, b) in enumerate(itertools.pairwise(edges.tolist())):
            pool[i, a:b] = 1.0
        cycles = torch.logspace(math.log10(1.28), math.log10(40.96), n_freqs, dtype=torch.float64)
        t = torch.arange(self.frames, dtype=torch.float64) * self.hop / window_n
        angle = 2 * torch.pi * cycles.unsqueeze(1) * t
        win = torch.hann_window(self.frames, dtype=torch.float64)
        self.register_buffer("kernels", _stft_kernels(n_fft))
        self.register_buffer("pool", pool)
        self.register_buffer(
            "basis", torch.cat([torch.cos(angle) * win, torch.sin(angle) * win]).float()
        )
        self.n_freqs, self.n_features = n_freqs, len(pool) * n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.conv1d(x, self.kernels, stride=self.hop)
        env = self.pool @ (z[:, : self.bins] ** 2 + z[:, self.bins :] ** 2)  # [B, carriers, T]
        mean = env.mean(-1, keepdim=True)
        proj = (env - mean) @ self.basis.T
        mag = torch.sqrt(proj[..., : self.n_freqs] ** 2 + proj[..., self.n_freqs :] ** 2 + 1e-20)
        mag = mag / (mean * (self.frames / 2) + 1e-12)
        # lower median as TopK: `median` has no ONNX export at the pinned opset
        floor = torch.topk(mag, (self.n_freqs - 1) // 2 + 1, dim=-1, largest=False).values[..., -1:]
        return torch.log(mag / (floor + 1e-9) + 1e-6).flatten(1)


# PU test bearing 6203 (Lessmeier et al. 2016, Table 1): 8 balls of 6.75 mm on a 28.55 mm pitch
# circle, 0° contact angle → fault frequencies as multiples of the shaft frequency
BPFO, BPFI, BSF2 = 3.053, 4.947, 3.994
FAULT_ORDERS = (
    1.0, 2.0, BPFO, 2 * BPFO, 3 * BPFO, 4 * BPFO, BPFI, 2 * BPFI, 3 * BPFI, 4 * BPFI,
    BPFI - 1, BPFI + 1, BSF2,
)  # fmt: skip
FLOOR_ORDERS = tuple(0.5 + 0.25 * k for k in range(47))  # 0.5 … 12 orders: the spectrum's floor


class KinFront(nn.Module):
    """Fixed front-end for long windows: envelope-spectrum level at the bearing's fault orders.

    For every carrier band (squared STFT envelope, as in `EnvFront`) and for each of the rig's two
    shaft speeds, the magnitude of the envelope spectrum at the 6203's fault orders, over its
    median across 0.5–12 orders. Evaluating both speeds makes the model speed-blind: no shaft
    speed is needed at run time. Specific to the Paderborn rig (ADR 0012).
    """

    kernels: torch.Tensor
    pool: torch.Tensor
    at_basis: torch.Tensor
    floor_basis: torch.Tensor

    def __init__(
        self, window_n: int, fs_hz: float = 64_000.0, shafts_hz: tuple[float, ...] = (25.0, 15.0),
        n_fft: int = 256, carriers: int = 6,
    ) -> None:  # fmt: skip
        super().__init__()
        self.hop, self.bins = n_fft // 8, n_fft // 2 + 1
        self.frames = (window_n - n_fft) // self.hop + 1
        first = max(1, self.bins // 64)
        edges = np.unique(np.round(np.geomspace(first, self.bins, carriers + 1)).astype(int))
        pool = torch.zeros(len(edges) - 1, self.bins)
        for i, (a, b) in enumerate(itertools.pairwise(edges.tolist())):
            pool[i, a:b] = 1.0
        t = torch.arange(self.frames, dtype=torch.float64) * self.hop / fs_hz
        win = torch.hann_window(self.frames, dtype=torch.float64)

        def basis(orders: tuple[float, ...]) -> torch.Tensor:
            hz = torch.tensor([s * o for s in shafts_hz for o in orders], dtype=torch.float64)
            angle = 2 * torch.pi * hz.unsqueeze(1) * t
            return torch.cat([torch.cos(angle) * win, torch.sin(angle) * win]).float()

        self.register_buffer("kernels", _stft_kernels(n_fft))
        self.register_buffer("pool", pool)
        self.register_buffer("at_basis", basis(FAULT_ORDERS))
        self.register_buffer("floor_basis", basis(FLOOR_ORDERS))
        self.n_shafts, self.n_carriers = len(shafts_hz), len(pool)
        self.n_features = self.n_shafts * self.n_carriers * len(FAULT_ORDERS)

    def _magnitude(self, e: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """|DFT| of the windowed envelope at the basis frequencies: [B, carriers, shafts, F]."""
        proj = e @ basis.T
        half = proj.shape[-1] // 2
        mag = torch.sqrt(proj[..., :half] ** 2 + proj[..., half:] ** 2 + 1e-20)
        return mag.reshape(*mag.shape[:2], self.n_shafts, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.conv1d(x, self.kernels, stride=self.hop)
        env = self.pool @ (z[:, : self.bins] ** 2 + z[:, self.bins :] ** 2)  # [B, carriers, T]
        mean = env.mean(-1, keepdim=True)
        e = env - mean
        scale = (mean * (self.frames / 2) + 1e-12).unsqueeze(-1)
        at = self._magnitude(e, self.at_basis) / scale
        grid = self._magnitude(e, self.floor_basis) / scale
        k = (grid.shape[-1] - 1) // 2 + 1  # lower median as TopK (no ONNX median at the opset)
        floor = torch.topk(grid, k, dim=-1, largest=False).values[..., -1:]
        out = torch.log(at / (floor + 1e-9) + 1e-6)  # [B, carriers, shafts, orders]
        return out.permute(0, 2, 1, 3).flatten(1)  # shaft-major, as in the model search


class KinNet(SpecNet):
    """f_raw for long Paderborn windows: KinFront → standardise → linear (ADR 0012)."""

    def __init__(self, window_n: int, n_classes: int = 3) -> None:
        nn.Module.__init__(self)
        self.front = KinFront(window_n)  # type: ignore[assignment]
        self.register_buffer("mean", torch.zeros(self.front.n_features))
        self.register_buffer("scale", torch.ones(self.front.n_features))
        self.linear = nn.Linear(self.front.n_features, n_classes)


class EnvNet(SpecNet):
    """f_raw for the Paderborn data: EnvFront → standardise → linear (ADR 0006)."""

    def __init__(self, window_n: int, n_classes: int = 3) -> None:
        nn.Module.__init__(self)
        self.front = EnvFront(window_n)  # type: ignore[assignment]
        self.register_buffer("mean", torch.zeros(self.front.n_features))
        self.register_buffer("scale", torch.ones(self.front.n_features))
        self.linear = nn.Linear(self.front.n_features, n_classes)


class ConvEncoder(nn.Module):
    """[B, 1, N] → [B, 64, N / 256]."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(1),
            _conv(1, 16, 15, 4),
            _conv(16, 32, 9, 4),
            _conv(32, 64, 9, 4),
            _conv(64, 64, 5, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """[B, 64, N / 256] → [B, 1, N]."""

    def __init__(self) -> None:
        super().__init__()

        def up(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose1d(c_in, c_out, 8, 4, padding=2), nn.ReLU(inplace=True)
            )

        self.net = nn.Sequential(
            up(64, 64), up(64, 32), up(32, 16), nn.ConvTranspose1d(16, 1, 8, 4, padding=2)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentEncoder(nn.Module):
    """ConvEncoder → dense latent [B, dim]. The part that is exported to ONNX for R5 / R6."""

    def __init__(self, window_n: int, dim: int) -> None:
        super().__init__()
        self.conv = ConvEncoder()
        self.fc = nn.Linear(64 * (window_n // DOWNSAMPLE), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class AutoEncoder(nn.Module):
    """R5: task-agnostic. Trained on waveform + log-spectrum reconstruction."""

    def __init__(self, window_n: int, dim: int = 32) -> None:
        super().__init__()
        self.encoder = LatentEncoder(window_n, dim)
        self.fc = nn.Linear(dim, 64 * (window_n // DOWNSAMPLE))
        self.decoder = ConvDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(self.fc(z).view(len(x), 64, -1))


class Bottleneck(nn.Module):
    """R6: task-aware. Encoder trained jointly with a classifier through a noisy bottleneck."""

    def __init__(self, window_n: int, dim: int = 16, n_classes: int = 3) -> None:
        super().__init__()
        self.encoder = LatentEncoder(window_n, dim)
        self.head = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(inplace=True), nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.encoder(x))
        if self.training:  # stand-in for int8 quantisation noise
            z = z + (torch.rand_like(z) - 0.5) / 127.0
        return self.head(z)


class VQEncoder(nn.Module):
    """ConvEncoder → nearest-codebook indices [B, N / 256]. Exported to ONNX for R7."""

    def __init__(self, code_dim: int = 8, n_codes: int = 256) -> None:
        super().__init__()
        self.conv = ConvEncoder()
        self.proj = nn.Conv1d(64, code_dim, 1)
        self.codebook = nn.Parameter(torch.randn(n_codes, code_dim) * 0.1)

    def latents(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x)).transpose(1, 2)  # [B, T, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.latents(x)
        d = (z * z).sum(-1, keepdim=True) - 2 * z @ self.codebook.T + (self.codebook**2).sum(-1)
        return d.argmin(-1)


class VQVAE(nn.Module):
    """R7: discrete latent. Straight-through VQ-VAE with a commitment loss."""

    def __init__(self, code_dim: int = 8, n_codes: int = 256) -> None:
        super().__init__()
        self.encoder = VQEncoder(code_dim, n_codes)
        self.up = nn.Conv1d(code_dim, 64, 1)
        self.decoder = ConvDecoder()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder.latents(x)
        q = self.encoder.codebook[self.encoder(x)]
        vq_loss = F.mse_loss(q, z.detach()) + 0.25 * F.mse_loss(z, q.detach())
        q = z + (q - z).detach()  # straight-through gradient
        return self.decoder(self.up(q.transpose(1, 2))), vq_loss
