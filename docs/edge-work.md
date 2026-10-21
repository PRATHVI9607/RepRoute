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

Measured from the real Paderborn folds on 2026-10-21:

| Check | Result |
|---|---|
| Real folds | 5 folds; mean `f_raw` macro-F1 0.608 |
| E1 profile | probe + predictor + selector 0.99 ms; `t_io` 0.118 ms; R6 encode 0.616 ms |
| E7 loopback | p50 gap 26–36%; selection agreement 1.0 |
| E7d Docker | p50 gap 4.1 / 10.5 / 6.5% at 1M / 256k / 5M; H7 fails at 256k |
| Selected representation | `r7_vq`: 16 B payload, 44 B frame |
| Rate estimate | stays at the 1 Mbit/s prior because frames are below `MIN_SAMPLE_BYTES` |

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

1. **Sim-to-real gap (P0).** Separate fixed event-loop/socket overhead from rate-dependent
   transport bytes, then calibrate the DES. Target: E7 and E7d p50 gap ≤ 10% with selection
   agreement ≥ 85%.
2. **Small-frame rate estimation (P1).** The edge never updates its bandwidth estimate for 16 B
   payloads. Decide whether to keep the prior, use ACK timing, or probe periodically.
3. **Duty cycle (P1).** Measure at 64, 8, and 2 ms periods and make E3 use the matching profile.
4. **Load model (P1).** Replace the `1/(1-rho)^gamma` fit with a capacity model if the measured
   latency supports it. Target: E4 bias within ±10% at loads 0 and 0.8.
5. **Docker repeatability (P1).** Keep the x86 closed loop reproducible and record the governor
   and host-load conditions with each run.

Model changes need an ADR dated before the result is inspected. A failed hypothesis stays failed;
the 10% threshold is not adjusted to make the result pass.
