# ADR 0017 — Edge-side codecs and features avoid multi-threaded BLAS

Date: 2026-10-30 · Status: accepted · Touches: `reps/classical.py` (Wavelet), `reps/features.py`

## Context

The Speech Commands E1 profile reported a wavelet encode of 65.9 ms at load 0 but 4.6 ms at load
0.5: faster under load, which no honest timing does. The codec's step search calls
`np.linalg.norm` on the 16 384 coefficients 24+ times per window, and the probe features call
`np.dot`. OpenBLAS threads these above roughly 10 000 elements, and waking its parked threads on
every call dominated the cost: a median of 1 057 ms per encode with default threads while another
job ran, 14 ms with one thread. Paderborn (4 096-sample windows, below the threshold) was not
affected: 0.95 ms at both loads. An edge device runs this code single-threaded, so the profile, and
every simulated latency built on it, charged a threading artifact.

## Decision

Replace those BLAS calls with element-wise sums (`np.square(c).sum()`, `(a * b).sum()`). No
thread-limiting dependency in the edge.

## Check

On 300 Speech Commands windows all eight classical codecs give byte-identical payloads before and
after; the probe features differ by at most 3e-12 relative, so cached codec outputs stay valid and
`PROBE_VERSION` / codec versions are unchanged. Wavelet encode: 4.1 ms with default threads.

## Consequences

The Speech Commands suite (E1–E10 but E6, E7d) is rerun with the fix; the earlier rows stay in the
registry. Paderborn's results are unaffected and not rerun.
