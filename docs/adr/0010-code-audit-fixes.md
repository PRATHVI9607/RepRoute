# ADR 0010 — Code audit: what was wrong and what changed

Date: 2026-10-14 · Status: accepted · Touches: `bench/`, `tasks/pipeline.py`, `edge/runtime.py`,
`data/pu.py`, `link/traces.py`, `certify/runtime.py`, `certify/fit.py`, `configs/scenario/`,
`scripts/`

Two read-only audits (subagents) went through the code; every finding was checked against the
callers before it was fixed, and each fix has a test where the logic is non-trivial.

## The finding that invalidates earlier numbers

**The scenario-grid experiments simulated one healthy bearing per fold.** `FoldCtx.workload` took
the *first* `n_windows` (1200) test windows of the fold. Windows are stored bearing by bearing,
healthy bearings first, and one bearing has about 1240 windows; every fold has a healthy test
bearing. So every cost, SLO and risk number of E3, E5 and the Pi-class E5 — and the windows E7
replays — came from a single healthy machine per fold. The workload is now `spread` evenly over
the whole test partition, in order (`test_workload_samples_every_bearing_…`).

All E3 / E5 / E7 results before this fix, including those in the run history of
[results.md](../research/results.md), are affected and must not be quoted. E1, E2, E4, E6, E8 and
E10 read the whole test partition (or sample it with a stride) and are not affected.

## Other fixes

| Finding | Effect before | Fix |
|---|---|---|
| H1's "main choice" of a cell where the oracle sent nothing was labelled `r0_raw` | could inflate H1's count of dominant representations | such cells have no main choice (`dominant_rep`) |
| A fold whose reference total is 0 (e.g. no SLO violations for B5 on bursty links) gave ±inf in `reduction` | an infinite or NaN H3 component | the fold is left out of that ratio |
| Every bursty cell replayed its trace from t = 0 | at 32 sensors a cell lasts 2.4 s, about three samples of a 1 s trace: "bursty" cells were nearly static | each cell starts at its own seeded point of the trace; the α levels of a cell share it |
| `late_cap_factor` in the config was never read | the config value had no effect (the default matched) | passed through to the metrics |
| `energy_budget_j` and `storage_bytes` in the scenario configs were never read | they described an ablation that does not exist | removed |
| The runtime calibration could write NaN into the profile if no window was delivered | every latency NaN, every Kai window deferred, silently | it raises |
| The predictor cache was keyed on the table only; the bundle fitted its own predictor with the fold's seed | a change to the fitting code would reuse stale predictors; seed ≠ 0 bundles certified with a different ĝ than the grid | keyed on table, seed and `FIT_VERSION`; the bundle reuses the cached predictor |
| Bundles were built in a fixed `_building` directory | two concurrent builds would delete each other's files | a unique temporary directory per build |
| The run registry was read-modify-written without a lock | two experiments finishing together lose a row | `fcntl` lock |
| A crashed `--force` rerun of a fold kept the old `metrics.json` | the half-rebuilt fold looked finished | `metrics.json` is removed when a fold starts |
| An exception in the edge's reply reader was lost when the task was cancelled | later windows looked undelivered | re-raised |
| A Paderborn bearing with no readable recording was only listed in the QA report | a class could silently lose a bearing | it raises |
| The bandwidth estimator divided by zero when a measured trace started in an outage | crash | the first sample is floored at 1 bit/s (found by the run itself) |
| Research scripts picked a manifest with `next(glob)` or hard-coded seed 0 | arbitrary manifest, KeyError for other seeds | exactly one manifest, the configured seed |
| `Certifier.safe_set` copied ĝ on every window | wasted work on the edge hot path (`safe_set` never mutates ĝ) | copies removed |

## Not changed (recorded)

- Baselines that look up representations by name (`r2a_wavelet`, `r6_bottleneck`, …) assume the
  full menu, which is the only menu this project runs.
- `CostModel.evaluate` and `MpcSafe` both write the latency formula; a test pins them together.
- The ACI threshold re-sorts the calibration thresholds per decision (microseconds); left as is.

## What was re-run

E3 (15 folds), the three price presets, E5, the Pi-class E5, E6 and E7, and the transfer run, after
these fixes. The results file says which runs come after this ADR.
