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

Measured from the real Paderborn folds on 2026-10-24, after the loopback pacer fix (ADR 0013) and
a fresh profile/bundle (`801b7bb6c1a9`):

| Check | Result |
|---|---|
| Real folds | 5 folds; mean `f_raw` macro-F1 0.608 |
| E1 profile (latest) | probe + predictor + selector 1.053 ms; `t_io` 0.050 ms (at the floor); R6 encode 0.970 ms. The load-0 calibration was −0.034 ms (the model overpredicted) and the load-0.8 excess was 4.968 ms, so the calibration is noisy |
| E7 loopback | **H7 true**: p50 gap 1.1 / 5.0 / 4.4% at 256k / 1M / 5M; max 5.0%; selection agreement 1.0. The preceding run measured 1.3 / 2.2 / 5.4% (max 5.4%), so the pass is repeatable on this profile |
| E7d Docker | **H7 true**: p50 gap 1.0 / 6.1 / 1.3% at 1M / 256k / 5M; max 6.1%; selection agreement 1.0; 300 windows per rate |
| E4 phase (pre-refresh) | **H6 false**: max boundary error 82.4% at load 0.8; median latency-model error 10.3%; aggregate latency bias +3.6% at load 0 and −1.7% at load 0.8. Rerun against the new bundle |
| E6 drift (pre-refresh) | L1 only; no L2 folds exist. Drift fired on 97.1% of L1 test windows |
| E8 overhead (pre-refresh) | 0.898 ms probe + predictor + selector; 2.0× the cheapest encoder. The new profile's overhead is 1.053 ms, so E8 needs a rerun |
| Selected representation | `r7_vq`: 16 B payload, 44 B frame |
| Rate estimate | stays at the 1 Mbit/s prior because frames are below `MIN_SAMPLE_BYTES` |
| Broader suite (pre-refresh) | H1/H3 (E3), H2a/H2b (E2), and H4/H5 (E5 and E5pi) also failed; E10 is descriptive |

The loopback pacer originally used asyncio's `call_later`, whose ~1 ms timer resolution dominated a
44 B frame and produced the earlier 28–38% E7 gap. `Edge._send` now uses a precise sleep and
subtracts that wait from `edge_ms`; the change is documented in
[ADR 0013](adr/0013-loopback-pacer-and-io-calibration.md). E7d never used the in-process pacer
(`shape_bps=None`), but it was rerun because the profile and bundle changed.

The profile itself moved materially between runs (overhead 0.90 → 1.053 ms, R6 0.784 → 0.970 ms)
while the machine stayed on the `powersave` governor. E7/E7d pass on this profile, but the remaining
pre-refresh experiments must not be quoted as if they used it.

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

1. **Refresh the remaining experiments.** E4, E6, E8, E10 and the wider E2/E3/E5 suite predate the
   pacer fix and the new profile. Rerun them before quoting their numbers; E4's phase-boundary
   failure and E6's missing L2 folds remain open.
2. **Timing stability (P1).** `t_io` hit its 0.05 ms floor and the load-0 calibration was negative,
   while the profile moved by roughly 17% between runs. Pin the governor or repeat the profile
   several times before treating a small gap as a real improvement.
3. **Small-frame rate estimation (P1).** The edge never updates its bandwidth estimate for 16 B
   payloads. Decide whether to keep the prior, use ACK timing, or probe periodically.
4. **Duty cycle (P1).** E1 now has 1-, 8-, and 32-sensor timings; verify E3 actually selects the
   matching profile at each node size.
5. **Load/phase model (P1).** The pre-refresh aggregate latency bias was inside ±10%, but the
   measured phase boundaries failed at load 0.8 (82.4% max error). Investigate the boundary
   procedure before changing the capacity model.
6. **L2 drift study.** E6 produced only L1 because no L2 folds exist under `artifacts/folds`.
   Build L2 folds if the L2 drift result is in scope.
7. **Zero-bandwidth warnings.** E3/E5 emit `divide by zero` warnings at grid cells with
   `bandwidth_bps == 0`. Decide whether those cells are intentional and guard the cost model so
   the warnings do not hide real failures.

Model changes need an ADR dated before the result is inspected. A failed hypothesis stays failed;
the 10% threshold is not adjusted to make the result pass. The pacer fix and the refreshed
calibration are recorded in [ADR 0013](adr/0013-loopback-pacer-and-io-calibration.md).
