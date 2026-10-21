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

Measured from the real Paderborn folds in the full suite run of 2026-10-21 17:40–18:10 UTC
(~29.5 minutes):

| Check | Result |
|---|---|
| Real folds | 5 folds; mean `f_raw` macro-F1 0.608 |
| E1 profile (latest) | probe + predictor + selector 0.90 ms; `t_io` 0.058 ms; R6 encode 0.784 ms. The preceding profile run measured 0.99 / 0.118 / 0.616 ms, so timing noise is material |
| E7 loopback | **H7 false**: p50 gap 28.5 / 29.9 / 38.1% at 256k / 1M / 5M; selection agreement 1.0 |
| E7d Docker | **H7 false**: p50 gap 4.1 / 10.5 / 6.5% at 1M / 256k / 5M. This summary predates the latest profile and bundle; rerun before quoting it |
| E4 phase | **H6 false**: max boundary error 82.4% at load 0.8; median latency-model error 10.3%; aggregate latency bias +3.6% at load 0 and −1.7% at load 0.8 |
| E6 drift | L1 only; no L2 folds exist. Drift fired on 97.1% of L1 test windows |
| E8 overhead | 0.898 ms probe + predictor + selector; 2.0× the cheapest encoder |
| Selected representation | `r7_vq`: 16 B payload, 44 B frame |
| Rate estimate | stays at the 1 Mbit/s prior because frames are below `MIN_SAMPLE_BYTES` |
| Broader suite | H1/H3 (E3), H2a/H2b (E2), and H4/H5 (E5 and E5pi) also failed; E10 is descriptive |

These numbers are provisional while the machine runs the `powersave` governor; final comparisons
should be repeated with the `performance` governor where possible.

## Workflow

Keep this order so the bundle matches the profile used by the real runs:

```bash
make profile
make folds FOLDS=5
make experiments
uv run --no-sync kai replay artifacts/folds/<split>/s0_r0f0 data/processed/replay.npz --n 300
uv run --no-sync kai bundle artifacts/folds/<split>/s0_r0f0
BUNDLE=artifacts/bundles/<id> FOLD=artifacts/folds/<split>/s0_r0f0 make docker-loop
make figures
make report
```

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

1. **Sim-to-real gap (P0).** The latest E7 run is off by 28.5–38.1%; separate fixed
   event-loop/socket overhead from rate-dependent transport bytes, then calibrate the DES.
   Target: E7 and E7d p50 gap ≤ 10% with selection agreement ≥ 85%.
2. **Rerun Docker E7d (P0 dependency).** The stored E7d summary is from before the latest profile
   and bundle. Rebuild the bundle and rerun the three-rate loop before comparing it with E7.
3. **Small-frame rate estimation (P1).** The edge never updates its bandwidth estimate for 16 B
   payloads. Decide whether to keep the prior, use ACK timing, or probe periodically.
4. **Duty cycle (P1).** E1 now has 1-, 8-, and 32-sensor timings; verify E3 actually selects the
   matching profile at each node size.
5. **Load/phase model (P1).** Aggregate latency bias is now inside ±10%, but the measured phase
   boundaries fail at load 0.8 (82.4% max error). Investigate the boundary procedure before
   changing the capacity model.
6. **L2 drift study.** E6 produced only L1 because no L2 folds exist under `artifacts/folds`.
   Build L2 folds if the L2 drift result is in scope.
7. **Zero-bandwidth warnings.** E3/E5 emit `divide by zero` warnings at grid cells with
   `bandwidth_bps == 0`. Decide whether those cells are intentional and guard the cost model so
   the warnings do not hide real failures.

Model changes need an ADR dated before the result is inspected. A failed hypothesis stays failed;
the 10% threshold is not adjusted to make the result pass.
