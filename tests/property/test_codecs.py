"""Codec laws (PRD §12.2 #4): lossless identity, lossy error ≤ ε, size ≤ q95, determinism."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from kai.contracts.representation import Representation
from kai.contracts.windows import WindowSet
from kai.reps.classical import Lossless, Raw, Spectral, Summary, Wavelet

N = 1024
windows = hnp.arrays(np.int16, N, elements=st.integers(-32768, 32767))
# realistic windows: band-limited-ish noise at an arbitrary gain, plus extremes
scaled = st.builds(
    lambda seed, gain: np.clip(
        np.random.default_rng(seed).normal(0, gain, N).cumsum() * 0.2, -32768, 32767
    ).astype(np.int16),
    st.integers(0, 10_000),
    st.floats(0.0, 4000.0),
)


def menu() -> list:
    return [
        Raw(N),
        Lossless(N),
        Wavelet("r2a_wavelet", N, eps=0.01),
        Wavelet("r2b_wavelet", N, eps=0.05),
        Spectral(N),
        Summary(N),
    ]


@given(windows | scaled, st.booleans())
def test_lossless_round_trip_is_identity(x: np.ndarray, delta: bool) -> None:
    rep = Lossless(N, delta=delta)
    assert np.array_equal(rep.decode_samples(rep.encode(x)), x)
    assert np.array_equal(Raw(N).decode_for_head(Raw(N).encode(x)), x.astype(np.float32) / 32768)


@given(windows | scaled, st.sampled_from([0.001, 0.01, 0.05, 0.3]))
def test_wavelet_relative_l2_error_is_within_eps(x: np.ndarray, eps: float) -> None:
    rep = Wavelet("r2a_wavelet", N, eps=eps)
    xf = x.astype(np.float64) / 32768
    err = np.linalg.norm(rep.decode_for_head(rep.encode(x)).astype(np.float64) - xf)
    # float32 output adds at most ~1 ulp per sample on top of the certified bound
    assert err <= eps * np.linalg.norm(xf) + 1e-6 * np.sqrt(N)


@given(windows | scaled)
def test_every_encoder_is_deterministic_and_decodes_to_finite_values(x: np.ndarray) -> None:
    for rep in menu():
        a, b = rep.encode(x), rep.encode(x.copy())
        assert a == b
        out = rep.decode_for_head(a)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        if rep.head_input == "waveform":
            assert out.shape == (N,)


def test_payload_within_declared_q95_and_fixed_sizes_exact(ws: WindowSet) -> None:
    for rep in menu():
        if isinstance(rep, Lossless | Wavelet):
            with pytest.raises(RuntimeError, match="fit"):
                rep.size_model()
            rep.fit(ws.x)
        sizes = np.array([len(rep.encode(w)) for w in ws.x])
        model = rep.size_model()
        assert np.mean(sizes <= model.q95_bytes) >= 0.95
        if not isinstance(rep, Lossless | Wavelet):
            assert set(sizes) == {model.mean_bytes}
    assert Summary(N).size_model().mean_bytes == 24
    assert Spectral(N).size_model().mean_bytes == 128


def test_menu_satisfies_protocol_and_hash_tracks_parameters(ws: WindowSet) -> None:
    reps = menu()
    assert all(isinstance(r, Representation) for r in reps)
    hashes = [r.contract_hash() for r in reps]
    assert len(set(hashes)) == len(hashes)
    assert all(len(h) == 8 and bytes.fromhex(h) for h in hashes)
    assert Wavelet("r2a_wavelet", N, eps=0.01).contract_hash() == hashes[2]
    assert Wavelet("r2a_wavelet", N, eps=0.02).contract_hash() != hashes[2]
    assert Lossless(N, level=9).contract_hash() != hashes[1]
    assert Raw(2 * N).contract_hash() != hashes[0]


def test_lossy_is_smaller_than_lossless_is_smaller_than_raw(ws: WindowSet) -> None:
    sizes = {}
    for rep in menu()[:4]:
        if not isinstance(rep, Raw):
            rep.fit(ws.x[::4])
        sizes[rep.rep_id] = rep.size_model().mean_bytes
    assert sizes["r2b_wavelet"] < sizes["r2a_wavelet"] < sizes["r1_lossless"] < sizes["r0_raw"]


def test_wavelet_rejects_indivisible_window_and_handles_silence() -> None:
    with pytest.raises(ValueError, match="divisible"):
        Wavelet("r2a_wavelet", 1000, eps=0.01)
    rep = Wavelet("r2a_wavelet", N, eps=0.01)
    assert not rep.decode_for_head(rep.encode(np.zeros(N, np.int16))).any()
