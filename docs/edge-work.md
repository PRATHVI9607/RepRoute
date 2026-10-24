# Edge-plane work (`edge` branch)

This branch covers the laptop-side systems and emulation work for Role D. The Raspberry Pi profile,
arm64 image builds, and Pi energy measurements are owned by a separate teammate and are out of
scope here.

## What this branch is for

- Closing the simulator-versus-real gap for E7 (loopback sockets) and E7d (Docker + `tc netem`).
- Profiling the edge runtime at the duty cycles the scenarios actually use.
- Replacing the fitted edge-load exponent with a model that matches the measured latency.
- Keeping the Docker closed loop repeatable on this machine.
- Documenting limitations that the paper must not overclaim.

It does not change the ML models, the certificate, the hypothesis thresholds, or the Pi work.

## Current baseline

Measured from the real Paderborn folds on 2026-10-24 under the pinned `performance` profile
(`power-profiles-daemon`; [ADR 0014](adr/0014-performance-timing-regime.md)) and bundle
`bb809ab9f5c8`:

| Check | Result |
|---|---|
| Real folds | 5 folds; mean `f_raw` macro-F1 0.608 |
| E1 profile | probe + predictor + selector 1.0715 ms; `t_io` 0.050 ms (at the floor); R6 encode 0.900 ms. The load-0 calibration was −0.371 ms (the model overpredicted) and the load-0.8 excess was 1.897 ms; `io_load_exponent` is 2.259 |
| E4 phase | **H6 true**: maximum representative-boundary error 4.7%; median latency-model error 7.2%; aggregate latency bias −6.8% at load 0 and −10.1% at load 0.8. The separate family-boundary check still has a 65.6% maximum error |
| E7 loopback | **H7 false** in two runs: p50 gap 6.3 / 16.0 / 20.5% and 23.3 / 38.4 / 42.3% at 256k / 1M / 5M; selection agreement 1.0. The model overpredicts real latency, and the real edge time varies materially between runs |
| E7d Docker | **H7 true**: p50 gap 2.2 / 1.4 / 0.5% at 1M / 256k / 5M; selection agreement 1.0; 300 windows per rate. The 10 ms netem delay dominates the edge-time model bias |
| E8 overhead | 1.0715 ms probe + predictor + selector; 13.7 µs per greedy decision and 43.0 / 75.3 / 294.2 µs per MPC h3/h4/h5 decision; 2.738× the cheapest encoder |
| E6 drift | L1 and L2 now. Drift fired on 97.1% of L1 and 88.5% of L2 test windows. L2 static safe-set risk 12.75% at α=5%; certified policy risk 0.91% static vs 0.87% with ACI; bytes 448 static vs 358 ACI, with 147 audit bytes |
| Selected representation | `r7_vq`: 16 B payload, 44 B frame |
| Rate estimate | stays at the 1 Mbit/s prior because frames are below `MIN_SAMPLE_BYTES` |
| Broader suite | H1/H3 (E3), H2a/H2b (E2), and H4/H5 (E5 and E5pi) also failed; E10 is descriptive |

The loopback pacer originally used asyncio's `call_later`, whose ~1 ms timer resolution dominated a
44 B frame and produced the earlier 28–38% E7 gap. `Edge._send` now uses a precise sleep and
subtracts that wait from `edge_ms`; the change is documented in
[ADR 0013](adr/0013-loopback-pacer-and-io-calibration.md). E7d never used the in-process pacer
(`shape_bps=None`), but it was rerun because the profile and bundle changed.

The earlier `powersave` baseline (`801b7bb6c1a9`) passed E7 at 1.1 / 5.0 / 4.4% and failed E4 at
82.4%. Under the pinned `performance` profile the edge is faster: E4 now matches the model, while
loopback E7 exposes a high-rate overprediction that the `powersave` run did not. Docker E7d still
passes because netem's fixed delay leaves less of the total latency for the edge term to explain.
This is a backend-sensitive result, not a threshold change; both outcomes are kept in the registry.

A second E7 run immediately after the suite produced larger gaps (23.3 / 38.4 / 42.3%) even though
the bundle and DES side were unchanged. The real p50 fell further while the simulated p50 stayed
fixed. That is measurement variance, not evidence for a model change. H7 stays false in both runs,
and E7d remains the only backend that passes.

## Workflow

Keep this order so the bundle matches the profile used by the real runs:

```bash
powerprofilesctl set performance   # record the regime; verify 16/16 governor + EPP
make folds FOLDS=5
make experiments                  # includes the E1 profile; does not run E7d
uv run --no-sync kai replay artifacts/folds/<split>/s0_r0f0 data/processed/replay.npz --n 300
uv run --no-sync kai bundle artifacts/folds/<split>/s0_r0f0
BUNDLE=artifacts/bundles/<id> FOLD=artifacts/folds/<split>/s0_r0f0 make docker-loop
make figures
make report
```

`power-profiles-daemon` owns the governor on this laptop; a bare `cpupower frequency-set` is
temporarily overridden and is not enough to record a stable regime. `make experiments` already
runs E1, so do not place a separate `make profile` before it unless you are deliberately measuring
the profile twice.

`make docker-loop` regenerates `experiments/docker/telemetry_*.msgpack`, which `kai exp e7d`
reads. Those raw telemetry files are local run outputs and are not tracked; the E7d summary and
results remain tracked so `make report` does not need the containers.

## Where outputs live

- `artifacts/folds/`, `artifacts/bundles/`, `artifacts/profiles/`: ignored, regenerate with the
  commands above.
- `data/processed/`: ignored; the validated `pu.npz` is the canonical dataset on this machine.
- `experiments/`: summaries, figures, and the run registry are tracked. Large per-cell grids and
  raw Docker telemetry are ignored.
- `experiments/edge/`: local diagnostics and scratch tables for this branch; ignored.

## Open work

1. **Loopback E7 model bias (P0).** With the faster `performance` profile the DES overpredicts
   loopback latency at 1 and 5 Mbit/s; Docker E7d passes because its fixed delay dominates. Do not
   tune the capacity model to recover H7. First decide whether loopback is a valid backend for the
   claim, or whether a principled transport term needs its own pre-registered ADR.
2. **Timing stability.** The governor is pinned, the profile is recorded, and `t_io` is still at its
   0.05 ms floor. Repeat E1 several times under `performance` to quantify the remaining variance;
   one profile is not evidence that a small gap is stable.
3. **Small-frame rate estimation (resolved).** The 44 B frame cannot produce a reliable goodput
   sample at these rates, so the edge keeps the configured `initial_bps` prior until a larger
   frame is sent. No padding/probe frame is added in this pass; ADR 0015 records the decision and
   the deployment requirement to supply the initial rate.
4. **Duty cycle (resolved).** E1 has 1-, 8-, and 32-sensor timings; E3 keys each cell by
   `n_sensors` and `load_profiles` selects the largest measured profile at or below it. Verified:
   overheads are 1.0715 / 0.7547 / 0.2918 ms and R6 encode is 0.900 / 0.588 / 0.414 ms at
   1 / 8 / 32 sensors. `tests/unit/test_profile_selection.py` guards the selection rule.
5. **E4 family boundaries (P1, exploratory).** H6 passes on the representative boundaries. The
   separate waveform-family check has a 65.6% maximum error, but its 51.7 Mbit/s theoretical
   boundary lies outside the 31.6 Mbit/s measurement grid, so the measured 30.3 Mbit/s crossing is
   not resolvable; the other family pair (11.2 vs 32.6 Mbit/s) still differs. Treat the family
   check as exploratory and do not change the capacity model to chase it.
6. **L2 drift study (resolved).** The five L2 folds were trained and E6 now reports both levels.
   The L2 static safe-set risk is 12.75% at α=5%, but the certified policy risk stays below 1%;
   drift fired on 88.5% of L2 windows. The byte and risk comparison is in
   `experiments/e6/summary.json`.
7. **Zero-bandwidth cells (resolved).** `CostModel.latency_ms` and `evaluate` now return infinity
   for `bandwidth_bps == 0` instead of emitting a divide-by-zero warning; the cell is infeasible.
   `tests/unit/test_analysis.py` covers it.
8. **Bundle provenance (resolved).** E7/E7d summaries now record `bundle_id`, `bundle_git_sha`, and
   `profiles_sha256`, so the cloud/lab/Pi handoff can identify the exact artifact set.

Model changes need an ADR dated before the result is inspected. A failed hypothesis stays failed;
the 10% threshold is not adjusted to make the result pass. The pacer fix, calibration, and timing
regime are recorded in [ADR 0013](adr/0013-loopback-pacer-and-io-calibration.md),
[ADR 0014](adr/0014-performance-timing-regime.md), and
[ADR 0015](adr/0015-small-frame-rate-estimation.md).
