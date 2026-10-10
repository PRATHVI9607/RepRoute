# Network traces

| Name | Kind | Source | Licence |
|---|---|---|---|
| `static` | constant rate | generated | n/a |
| `markov_bursty` | two-state Markov-modulated (Gilbert–Elliott), good rate vs 1/20 of it | `kai.link.traces.markov_bursty` | n/a |
| `3g_synth`, `4g_synth`, `5g_synth` | **synthetic** log-normal AR(1) with rare near-outages | `kai.link.traces.cellular` | n/a |

The cellular traces are **synthetic** and must be labelled so in every figure. Their medians, spread
and correlation are order-of-magnitude choices, not fits to a dataset. Every trace is rescaled so
that its median equals the cell's nominal bandwidth (the "sensor-data regime", PRD §5.5).

## Adding a measured trace

Public 3G/4G/5G throughput logs exist from several groups; none is shipped here because their
availability and licences were not verified in this pass (PRD §5.5 `[VERIFY]`). To use one:

1. Put a one-column CSV (bits per second per sample) under this directory.
2. Record its source URL, licence and sampling period in the table above.
3. Load it with `kai.link.traces.from_csv(path, step_s)`.
