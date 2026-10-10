"""Torch models (lab plane only): raw reference model f_raw and the learned encoders R5–R7."""

from __future__ import annotations

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
        t = torch.arange(n_fft, dtype=torch.float64)
        phase = 2 * torch.pi * torch.arange(bins, dtype=torch.float64).unsqueeze(1) * t / n_fft
        win = torch.hann_window(n_fft, dtype=torch.float64)
        kernels = torch.cat([torch.cos(phase) * win, torch.sin(phase) * win]).unsqueeze(1)
        edges = torch.unique(
            torch.round(torch.logspace(0, torch.log10(torch.tensor(float(bins))), n_bands + 1))
        )
        pool = torch.zeros(len(edges) - 1, bins)
        for i, (a, b) in enumerate(zip(edges[:-1].int(), edges[1:].int(), strict=True)):
            pool[i, a:b] = 1.0
        self.register_buffer("kernels", kernels.float())
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
