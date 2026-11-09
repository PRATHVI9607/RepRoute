"""Build, save and load the representation menu. Loading verifies every contract hash."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kai.reps.base import BaseRep
from kai.reps.classical import Lossless, Raw, Spectral, Summary, Wavelet
from kai.reps.logmel import LogMel

# Menu order is fixed: it is the column order of the disagreement table and the predictor output.
MENU_ORDER = (
    "r0_raw",
    "r1_lossless",
    "r2a_wavelet",
    "r2b_wavelet",
    "r2c_wavelet",
    "r2d_wavelet",
    "r3_spectral",
    "r4_summary",
    "r5_ae",
    "r6_bottleneck",
    "r7_vq",
    "r10a_logmel",
    "r10b_logmel",
)
WAVELET_EPS = {"r2a_wavelet": 0.01, "r2b_wavelet": 0.05}
# opt-in mid-size codecs between near-lossless and the learned codes (ADR 0016, exploratory)
EXTRA_WAVELET_EPS = {"r2c_wavelet": 0.15, "r2d_wavelet": 0.30}
# the waveform-preserving family (H5 and the compression-only ablation)
WAVEFORM_REPS = ("r0_raw", "r1_lossless", *WAVELET_EPS, *EXTRA_WAVELET_EPS)
# opt-in coarse log-mel grids (n_mels, n_frames) with a distilled CNN head (ADR 0018, exploratory)
LOGMEL_GRID = {"r10a_logmel": (16, 16), "r10b_logmel": (12, 12)}
# the filterbank starts at 20 Hz: at low rates (HAR 50 Hz, Sleep-EDF 102.4 Hz) bands are empty
LOGMEL_MIN_HZ = 4000.0


class ContractMismatchError(RuntimeError):
    """A stored representation does not hash to what the bundle recorded."""


def classical_menu(
    window_n: int, extra_wavelets: bool = False, logmel_hz: float | None = None
) -> list[BaseRep]:
    """The fold-independent codecs; `logmel_hz` (the sample rate) adds the R10 log-mel grids."""
    if logmel_hz is not None and logmel_hz < LOGMEL_MIN_HZ:
        raise ValueError(f"log-mel is an audio representation: {logmel_hz} Hz < {LOGMEL_MIN_HZ}")
    eps = WAVELET_EPS | (EXTRA_WAVELET_EPS if extra_wavelets else {})
    logmel = (
        []
        if logmel_hz is None
        else [LogMel(rid, window_n, logmel_hz, m, t) for rid, (m, t) in LOGMEL_GRID.items()]
    )
    return [
        Raw(window_n),
        Lossless(window_n),
        *(Wavelet(rid, window_n, e) for rid, e in eps.items()),
        Spectral(window_n),
        Summary(window_n),
        *logmel,
    ]


def _build(rep_id: str, p: dict[str, Any], directory: Path) -> BaseRep:  # noqa: PLR0911 - one branch per codec
    n = p["window_n"]
    if rep_id == "r0_raw":
        return Raw(n)
    if rep_id == "r1_lossless":
        return Lossless(n, p["level"], p["delta"])
    if rep_id in WAVELET_EPS or rep_id in EXTRA_WAVELET_EPS:
        return Wavelet(rep_id, n, p["eps"], p["level"], p["zlevel"])
    if rep_id == "r3_spectral":
        return Spectral(n, p["env_bands"], p["spec_bands"])
    if rep_id == "r4_summary":
        return Summary(n)
    if rep_id in LOGMEL_GRID:
        return LogMel(rep_id, n, p["sample_rate_hz"], p["n_mels"], p["n_frames"])
    from kai.reps.latent import OnnxLatent, OnnxVQ  # onnxruntime only when a learned rep is used

    onnx_bytes = (directory / f"{rep_id}.onnx").read_bytes()
    table = np.load(directory / f"{rep_id}.npy")
    cls = OnnxVQ if rep_id == "r7_vq" else OnnxLatent
    return cls(rep_id, n, onnx_bytes, table)


def save_menu(reps: list[BaseRep], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for rep in reps:
        index.append(
            {"rep_id": rep.rep_id, "contract_hash": rep.contract_hash(), **rep.to_params()}
        )
        if rep.onnx_bytes is not None and rep.table is not None:
            (directory / f"{rep.rep_id}.onnx").write_bytes(rep.onnx_bytes)
            np.save(directory / f"{rep.rep_id}.npy", rep.table)
    (directory / "reps.json").write_text(json.dumps(index, indent=1))


def load_menu(directory: Path) -> list[BaseRep]:
    reps = []
    for entry in json.loads((directory / "reps.json").read_text()):
        rep = _build(entry["rep_id"], entry, directory)
        rep._restore_size(entry)
        if rep.contract_hash() != entry["contract_hash"]:
            raise ContractMismatchError(
                f"{rep.rep_id}: bundle says {entry['contract_hash']}, "
                f"files hash to {rep.contract_hash()}"
            )
        reps.append(rep)
    return reps
