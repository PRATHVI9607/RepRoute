"""Non-learned representations R0–R4 (PRD §7). numpy / pywt / zstandard only — edge-safe."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pywt
import zstandard

from kai.contracts.representation import FloatArray, WindowArray
from kai.reps import features as ft
from kai.reps.base import BaseRep

_F16_MAX = float(np.finfo(np.float16).max)


def _shuffle(a: np.ndarray) -> bytes:
    """Byte-plane shuffle: high bytes of int16 samples compress far better on their own."""
    return a.astype("<i2").view(np.uint8).reshape(-1, 2).T.tobytes()


def _unshuffle(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.uint8).reshape(2, -1).T.copy().view("<i2").ravel()


def _to_f16(v: np.ndarray) -> bytes:
    return np.clip(v, -_F16_MAX, _F16_MAX).astype("<f2").tobytes()


class Raw(BaseRep):
    """R0: int16 as acquired. The reference representation r0."""

    head_input = "waveform"

    def __init__(self, window_n: int) -> None:
        super().__init__("r0_raw", window_n, fixed_bytes=2 * window_n)

    def encode(self, x: WindowArray) -> bytes:
        return x.astype("<i2").tobytes()

    def decode_for_head(self, payload: bytes) -> FloatArray:
        return np.frombuffer(payload, "<i2").astype(np.float32) / 32768.0


class Lossless(BaseRep):
    """R1: optional delta → byte shuffle → zstd. Bit-exact round trip."""

    head_input = "waveform"

    def __init__(self, window_n: int, level: int = 3, delta: bool = True) -> None:
        super().__init__("r1_lossless", window_n)
        self.level = level
        self.delta = delta
        self._c = zstandard.ZstdCompressor(level=level)
        self._d = zstandard.ZstdDecompressor()

    def codec_params(self) -> dict[str, Any]:
        return {"window_n": self.window_n, "level": self.level, "delta": self.delta}

    def encode(self, x: WindowArray) -> bytes:
        # int16 wrap-around differences are exactly invertible by a wrapping cumulative sum
        d = np.diff(x.astype(np.int32), prepend=0).astype(np.int16) if self.delta else x
        return self._c.compress(_shuffle(d))

    def decode_samples(self, payload: bytes) -> WindowArray:
        d = _unshuffle(self._d.decompress(payload, max_output_size=2 * self.window_n))
        return np.cumsum(d, dtype=np.int64).astype(np.int16) if self.delta else d.astype(np.int16)

    def decode_for_head(self, payload: bytes) -> FloatArray:
        return self.decode_samples(payload).astype(np.float32) / 32768.0


class Wavelet(BaseRep):
    """R2: db4 wavelet, uniform quantisation with relative L2 error ≤ eps, zstd.

    The periodised db4 transform is orthonormal, so the coefficient-domain error equals the
    signal-domain error and the step size is found by bisection without inverse transforms.
    """

    head_input = "waveform"
    _BISECT_ITERS = 24

    def __init__(self, rep_id: str, window_n: int, eps: float, level: int = 5, zlevel: int = 3):
        super().__init__(rep_id, window_n)
        if window_n % (1 << level):
            raise ValueError(f"window_n={window_n} must be divisible by 2**{level}")
        self.eps, self.level, self.zlevel = eps, level, zlevel
        self._c = zstandard.ZstdCompressor(level=zlevel)
        self._d = zstandard.ZstdDecompressor()
        sizes = [window_n >> level] + [window_n >> j for j in range(level, 0, -1)]
        self._splits = np.cumsum(sizes)[:-1]

    def codec_params(self) -> dict[str, Any]:
        return {
            "window_n": self.window_n,
            "eps": self.eps,
            "level": self.level,
            "zlevel": self.zlevel,
            "wavelet": "db4",
            "mode": "periodization",
        }

    def encode(self, x: WindowArray) -> bytes:
        c = np.concatenate(pywt.wavedec(x.astype(np.float64), "db4", "periodization", self.level))
        budget = self.eps * float(np.linalg.norm(c))
        # Rounding moves each coefficient by ≤ step/2, so this step is guaranteed within budget;
        # bisection (in float32, the wire type) then only ever accepts steps it has tested.
        lo = np.float32(2.0 * budget / np.sqrt(len(c)))
        if float(lo) * np.sqrt(len(c)) > 2.0 * budget:
            lo = np.nextafter(lo, np.float32(0))
        hi = np.float32(2.0 * float(np.abs(c).max()) + 1.0)
        for _ in range(self._BISECT_ITERS):
            mid = np.float32(0.5 * (float(lo) + float(hi)))
            if mid <= lo or mid >= hi:
                break
            if np.linalg.norm(c - np.rint(c / float(mid)) * float(mid)) <= budget:
                lo = mid
            else:
                hi = mid
        step = lo if lo > 0 else np.float32(1.0)  # lo == 0 only for an all-zero window
        q = np.rint(c / float(step)).astype(np.int64)
        if np.abs(q).max(initial=0) <= 127:
            kind, body = 0, q.astype(np.int8).tobytes()
        elif np.abs(q).max() <= 32767:
            kind, body = 1, _shuffle(q)
        else:  # eps so small that coefficients overflow int16: fall back to int32
            kind, body = 2, q.astype("<i4").tobytes()
        return struct.pack("<fB", float(step), kind) + self._c.compress(body)

    def decode_for_head(self, payload: bytes) -> FloatArray:
        step, kind = struct.unpack_from("<fB", payload)
        body = self._d.decompress(payload[5:], max_output_size=4 * self.window_n)
        if kind == 0:
            q = np.frombuffer(body, np.int8)
        elif kind == 1:
            q = _unshuffle(body)
        else:
            q = np.frombuffer(body, "<i4")
        coeffs = np.split(q.astype(np.float64) * step, self._splits)
        return (pywt.waverec(coeffs, "db4", "periodization") / 32768.0).astype(np.float32)


class Summary(BaseRep):
    """R4: ten time statistics + spectral centroid and entropy, float16 (24 B)."""

    version = "2"

    N_FEATURES = len(ft.TIME_STAT_NAMES) + 2

    def __init__(self, window_n: int) -> None:
        super().__init__("r4_summary", window_n, fixed_bytes=2 * self.N_FEATURES)

    def encode(self, x: WindowArray) -> bytes:
        xf = ft.to_float(x)
        return _to_f16(
            np.concatenate([ft.time_stats(xf), ft.spectral_shape(ft.power_spectrum(xf))])
        )

    def decode_for_head(self, payload: bytes) -> FloatArray:
        return np.frombuffer(payload, "<f2").astype(np.float32)


class Spectral(BaseRep):
    """R3: envelope-spectrum and raw-spectrum log band energies + time statistics, float16."""

    version = "2"

    def __init__(self, window_n: int, env_bands: int = 32, spec_bands: int = 22) -> None:
        self.env_bands, self.spec_bands = env_bands, spec_bands
        n_feat = env_bands + spec_bands + len(ft.TIME_STAT_NAMES)
        super().__init__("r3_spectral", window_n, fixed_bytes=2 * n_feat)
        half = window_n // 2
        self._env_edges = ft.log_band_edges(half // 2, env_bands)  # envelope: up to fs/4
        self._spec_edges = ft.log_band_edges(half, spec_bands)

    def codec_params(self) -> dict[str, Any]:
        return {
            "window_n": self.window_n,
            "env_bands": self.env_bands,
            "spec_bands": self.spec_bands,
        }

    def encode(self, x: WindowArray) -> bytes:
        xf = ft.to_float(x)
        env_power = ft.power_spectrum(ft.envelope(xf))
        return _to_f16(
            np.concatenate(
                [
                    ft.log_band_energies(env_power, self._env_edges),
                    ft.log_band_energies(ft.power_spectrum(xf), self._spec_edges),
                    ft.time_stats(xf),
                ]
            )
        )

    def decode_for_head(self, payload: bytes) -> FloatArray:
        return np.frombuffer(payload, "<f2").astype(np.float32)
