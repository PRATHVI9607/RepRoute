# ADR 0008 — Timing at the node's duty cycle, a certified static safe set, measured traces

Date: 2026-10-14 · Status: accepted · Touches: `profile/profiler.py`, `select/cost.py`,
`select/policies.py`, `certify/fit.py`, `bench/calibrate.py`, `bench/evaluate.py`,
`bench/experiments.py`, `configs/scenario/grid.yaml`, `scripts/download_traces.py`,
`scripts/drift_study.py`

The results of the second pass (2026-10-14, [results.md](../research/results.md) history) showed
Kai with the any-policy certificate costing 6.6 × more than always sending one representation.
Before reading more into that, the parts of the comparison that were artefacts of the simulator had
to be separated from the parts that are real. All changes below were made **after** those results
were seen; none changes a hypothesis, a metric or a threshold, and every experiment was re-run.

## 1. Timings are measured at each node's duty cycle

**Problem.** ADR 0004 §4 established that the same code is 3–6 × slower when it runs once per
64 ms than in a tight loop (cold caches, a CPU that has clocked down), and the profile was
re-measured at one call per window period. That is the right duty cycle for a node with **one**
sensor. A node with 32 sensors handles a window every 2 ms and is never idle, yet the simulator
charged it the cold timings: 2.3 ms of probe, predictor and selector per window against a 2 ms
budget. Every policy that needs the safe set therefore saturated the edge CPU on paper, and the
32-sensor cells dominated the total cost.

**Decision.** For every sensor count `n` of the scenario grid above one, E1 times the stages
**inside the whole per-window pipeline** — probe → predictor → selector → encoder, back to back,
one window every `period / n` (`profile.profiler.profile_pipeline`) — and stores the result under
`busy` in the profile. The grid experiments use the timings measured for the cell's sensor count
(`load_profiles(n_sensors=…)`). `time_ms` itself now starts one call per period and sleeps only
for what is left of it. Load exponents are those fitted at one sensor; decoder and head times are
cloud-side and unchanged. The bundle, and with it the real edge runtime and the sim-to-real
experiments, still use the one-sensor profile: that is what they run.

A first version timed each stage alone at `period / n`. That is not a busy node: with a
millisecond of sleep between calls the laptop's CPU clocks down again, and the probe measured
0.93 ms at a 2 ms period against 1.34 ms at 64 ms — still far from what it costs when the encoder
of the previous window has just run.

**Limits.**

- Static policies run no probe, so on their own they would be a little colder than inside the
  pipeline; they are given the pipeline's encoder timings all the same. That favours the baselines.
- These are timings of a laptop whose frequency governor (`powersave`) cannot be pinned. They move
  by tens of percent between runs, and the 32-sensor cells sit close to the point where a core is
  full. Treat every 32-sensor number as conditional on this profile; the report breaks the main
  comparison down by sensor count for that reason.
- No multi-sensor node was run for real.

## 2. The runtime's per-frame overhead is in the cost model

**Problem.** Micro-benchmarks of the encoders under co-tenant load fit load exponents near 0, yet
real closed-loop latency at edge load 0.8 was about 40 % above the model (E4): framing, the event
loop, the socket and, under load, waiting for the CPU at every wake-up are not in any encoder's
timing.

**Decision.** `Overheads` gains `t_io_ms` and `io_load_exponent`; `CostModel.edge_time_ms` adds
`t_io / (1 − ρ)^γ_io` to every representation, for every policy (a static policy sends frames too).
E1 calibrates the two numbers with two short closed-loop runs over loopback — the smallest
representation, a 20 Mbit/s link, load 0 and 0.8 — as the median latency beyond the model's
prediction. E4 and E7 then test the model at other rates, representations and loads. With
`t_io_ms = 0` the model is the PRD formula, and the unit test of that formula is unchanged.

This is a calibration against the real runtime, so E4 and E7 are no longer fully independent of
the model they test: the model has seen two of the conditions. They are reported as before.

## 3. A certified safe set that needs no probe and no predictor

**Observation.** With the menu of ADR 0006 the sufficiency predictor has nothing to predict: the
representations either disagree with `f_raw` about as often as chance or (R6, the wavelets) in
under 3 % of windows, at windows the 20 probe features cannot single out. The per-window scores
then only add noise and 2.3 ms of work; in the ablation, replacing the predictor by a lookup table
was 4.7 × cheaper.

**Decision.** Proposition 1 asks only that the scores are independent of the calibration and test
windows. A constant per representation qualifies. `certify.fit.constant_scores` gives each
representation its Laplace-smoothed disagreement rate on `ptrain`; `bench.calibrate.static_safe_set`
runs the ordinary V0 calibration with those scores and returns one safe set for every window, with
the same any-policy guarantee. The selector `GreedyStatic` chooses inside it from the link state
and is charged for the selector only. It appears in E2 as variant `v0_static` and in E3 as
`kai_s1_v0_static`.

- It differs from baseline B11 and from the `no_predictor` ablation, which admit each
  representation on its own marginal bound: here the bound is on the *joint* loss of the set, which
  is what "any policy inside the set" needs.
- It is **exploratory**: added after the results, with no hypothesis attached. H3 stays defined on
  Kai with S1 and V0 as registered.
- Test: `test_constant_scores_give_one_valid_safe_set_for_every_window`.

## 4. Measured cellular traces

The grid gains `ucc4g_static`, `ucc4g_pedestrian` and `ucc4g_car`: measured 4G throughput logs
(Raca et al., MMSys 2018, CC BY 4.0), downloaded and checksum-verified by
`scripts/download_traces.py`. The column used is the **downlink** bitrate during continuous
downloads; the uplink column of the same logs is acknowledgement traffic at about 2 % of it and is
not capacity. The traces therefore contribute the *shape over time* of a real cellular link,
rescaled to each cell's uplink rate like every other trace; no measured uplink capacity is claimed
([traces/README.md](../../configs/scenario/traces/README.md)). The synthetic traces stay in the
grid. The grid grows from 972 to 1296 cells; H1 and H3 are evaluated on the grid as it now stands,
and the change is logged in `hypotheses.md`.

## 5. Drift monitor: a negative result

The shipped monitor (per-feature KS test against the pooled calibration bearings) is up on 97 % of
test windows without any condition shift and on 89 % with one: it detects that a bearing is new.
Two alternatives were tried offline on the trained folds (`scripts/drift_study.py`,
`experiments/drift_study/summary.json`):

| Design | Alarm without a condition shift | Alarm under a held-out condition |
|---|---|---|
| KS against pooled calibration bearings (shipped) | 0.97 | 0.89 |
| Share of windows outside the calibration bearings' feature range | 0.29 | 0.42 |
| Per machine: recent windows outside that machine's own earlier range, best setting tried | 0.25 | 0.73 |
| Per machine, other settings | 0.31–0.50 | 0.76–0.83 |

Even with a reference from the same machine in the same conditions, a quarter to a half of the
time the alarm is up: the 20 probe features move between recordings of one bearing in one condition
almost as much as between conditions. No monitor on these features separates "new operating
condition" from ordinary variation at a usable false-alarm rate. The shipped monitor is left as it
is and stays out of the certified path by default; the audit loop (ADR 0007) is the mechanism that
actually reacts to a shift. Target not met: false alarms below 5 %.

## Not changed

- The sim-to-real comparison has no modelling fix beyond §2. On loopback the gap is now within
  ± 0.6 ms on 5–7 ms; a 10 % threshold on latencies that small is at the noise floor of a laptop
  whose frequency governor cannot be pinned.
- No "edge inference" policy was added to the grid. Running the whole reference model on the edge
  costs one linear layer more than R6's encoder and 15 bytes less per frame, so the best-single
  row (R6) stands in for it; the report says so next to the profile table.
