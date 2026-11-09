"""R10: coarse log-mel spectrogram, uint8 (ADR 0018, opt-in). numpy only — edge-safe.

The window is cut into `n_frames` Hann frames (hop = window_n / n_frames, 50 % overlap) and each
frame's power spectrum is pooled into `n_mels` triangular mel bands. The log energies are sent as
one float16 peak plus uint8 levels over an 80 dB range below that peak. The cloud head is a small
CNN distilled from f_raw (`kai.tasks.train.train_logmel_head`), served through ONNX Runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kai.contracts.representation import FloatArray, WindowArray
from kai.reps import features as ft
from kai.reps.base import BaseRep

RANGE_DB = 80.0
_RANGE = RANGE_DB * np.log(10.0) / 10.0  # 80 dB in natural-log power units


def mel_filterbank(n_fft: int, n_mels: int, sample_rate_hz: float) -> np.ndarray:
    """Triangular filters [n_mels, n_fft // 2 + 1], equally spaced on the mel scale from 20 Hz."""

    def mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    lo_mel, hi_mel = mel(np.array([20.0, sample_rate_hz / 2.0]))
    pts = 700.0 * (10.0 ** (np.linspace(lo_mel, hi_mel, n_mels + 2) / 2595.0) - 1.0)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate_hz)
    lo, mid, hi = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    out: np.ndarray = np.clip(
        np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), 0, None
    )
    return out


class LogMel(BaseRep):
    version = "1"

    def __init__(
        self, rep_id: str, window_n: int, sample_rate_hz: float, n_mels: int, n_frames: int
    ) -> None:
        super().__init__(rep_id, window_n, fixed_bytes=n_mels * n_frames + 2)
        self.sample_rate_hz, self.n_mels, self.n_frames = float(sample_rate_hz), n_mels, n_frames
        self._hop = window_n // n_frames
        n_fft = 2 * self._hop
        self._fb = mel_filterbank(n_fft, n_mels, self.sample_rate_hz)
        self._hann = np.hanning(n_fft)

    def codec_params(self) -> dict[str, Any]:
        return {
            "window_n": self.window_n,
            "sample_rate_hz": self.sample_rate_hz,
            "n_mels": self.n_mels,
            "n_frames": self.n_frames,
        }

    def encode(self, x: WindowArray) -> bytes:
        h = self._hop
        xf = np.pad(ft.to_float(x), (h // 2, h + h // 2))
        idx = np.arange(self.n_frames)[:, None] * h + np.arange(2 * h)
        spec = np.fft.rfft(xf[idx] * self._hann, axis=1)
        power = spec.real * spec.real + spec.imag * spec.imag
        # einsum, not `@`: a BLAS matmul starts threads on the edge (ADR 0017)
        lm = np.log(np.einsum("tb,mb->tm", power, self._fb) + 1e-10)
        peak = np.float16(min(float(lm.max()), 6e4))
        q = np.round(np.clip((lm - (float(peak) - _RANGE)) / _RANGE, 0.0, 1.0) * 255.0)
        return peak.astype("<f2").tobytes() + q.astype(np.uint8).tobytes()

    def decode_for_head(self, payload: bytes) -> FloatArray:
        """[peak / 10, levels in [0, 1] row-major (frame, mel)] as float32."""
        expected = self.n_mels * self.n_frames + 2
        if len(payload) != expected:
            raise ValueError(f"log-mel payload of {len(payload)} bytes, expected {expected}")
        peak = np.frombuffer(payload[:2], "<f2").astype(np.float32) / 10.0
        levels = np.frombuffer(payload[2:], np.uint8).astype(np.float32) / 255.0
        return np.concatenate([peak, levels])
