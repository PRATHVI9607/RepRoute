# Pre-registered hypotheses

Frozen on **10 Oct 2026**, before any evaluation experiment (E2–E8) was run on the Paderborn data.
Source: PRD §4. This file fixes the *operational* definitions that the PRD leaves open, so that no
metric can be chosen after seeing results. The repository had no commits when this was written; the
freeze is anchored by the first commit that contains this file.

Amendments: only via an ADR in `docs/adr/` with a date and a reason, and never after seeing
test-fold results for the hypothesis being amended. A failed hypothesis is reported as failed.

## Research question

Can an edge runtime pick, per window, the cheapest representation from a heterogeneous menu that is
statistically certified to preserve the cloud task decision, while meeting deadlines under changing
bandwidth, queue and compute?

## Common definitions

| Term | Definition used in the code |
|---|---|
| Risk | Decision-agreement loss with the raw model, `d_r(x) = 1[ŷ_r(x) ≠ f_raw(x)]`. Never label accuracy. |
| Safe-set risk | `mean(max_{r ∈ S_λ̂(x)} d_r(x))` on test windows: the risk of the *worst* policy inside the safe set. |
| Fold | One leave-bearings-out assignment (L1). The unit of replication and of every confidence interval. |
| Cell | One point of the scenario grid: link × deadline × edge load × sensors × α (`configs/scenario/grid.yaml`). |
| Cost of a run | Mean per-window realised cost under the price preset: actual bytes, actual CPU time, modelled energy, and lateness `max(0, L − D)` capped at 10·D. A window that is never delivered is charged the cap. |
| Total cost | Sum over all grid cells of the cost of a run, per fold. Reductions are `1 − total_A / total_B`, computed per fold, then averaged. |
| Matched risk | Policies are compared at the same α. B4, B8 and B11 may only use representations whose marginal disagreement passes the same finite-sample bound on the calibration partition. |
| Default preset | `metered_cellular`, mission `fidelity_first`. |
| Edge timings | Measured on the development laptop (E1). A Pi-class edge is a labelled sensitivity analysis until the Raspberry Pi is available. |
| Energy | Modelled, never measured, in every result of this pass. |

## Hypotheses

| ID | Statement | Metric and test | Threshold | If false |
|---|---|---|---|---|
| **H1** Diversity | The oracle uses several representations, and no single one is close to it. | (a) Number of representations that are the oracle's most-used choice in ≥ 10 % of (fold, cell) pairs. (b) `total_B4 / total_oracle − 1`, mean over folds. | (a) ≥ 3 and (b) ≥ 0.25 | Reframe as benchmark / regime-analysis paper (C3–C5) |
| **H2a** Certificate, expected | V0 (CRC) controls the safe-set risk on held-out bearings. | Safe-set risk per fold for α ∈ {0.02, 0.05, 0.10}; mean over folds; 95 % bearing-block bootstrap CI on the pooled test windows. | mean ≤ α and CI upper bound ≤ α + 0.01, for every α | Diagnose exchangeability; fall back to per-bearing calibration |
| **H2b** Certificate, PAC | V1 (single-threshold LTT, δ = 0.1) holds in most folds. | Fraction of folds whose safe-set risk ≤ α, for every α. | ≥ 0.90 | Report as limitation; keep H2a |
| **H3** Adaptation value | Kai (S1 greedy-safe, V0) is cheaper than the best single representation and more reliable than a context-blind selector. | (a) `1 − total_Kai / total_B4` over the full grid. (b) `1 − SLO_Kai / SLO_B5` on the bursty-trace cells, where SLO = fraction of windows not delivered within the deadline, summed over cells. | (a) ≥ 0.20 and (b) ≥ 0.30 | Descriptive trade-off paper |
| **H4** Encode cost matters | A planner that believes encoding is free chooses differently under load and misses more deadlines. | On cells with ρ = 0.8 and α = 0.05: (a) fraction of (fold, cell) pairs whose most-used representation differs between Kai and the "no encode cost" ablation; (b) mean SLO violation of the ablation vs Kai. | (a) ≥ 0.15 and (b) ablation > Kai | Drop heavy learned representations |
| **H5** Heterogeneity pays | The full menu beats the best within-family controller. | `1 − total_Kai / min(total_compression-only, total_learned-only)` at α = 0.05, per fold. | ≥ 0.10 | Narrow to the families that matter |
| **H6** Theory predicts reality | Closed-form break-even bandwidths locate the measured phase boundaries. | For every boundary of the predicted latency envelope that is also observed: `|B*_measured − B*_theory| / B*_theory`, where measured latencies come from the real socket runtime. | every matched boundary ≤ 0.20, and ≥ 1 boundary matched | Report the empirical map only |
| **H7** Sim-to-real | The simulator reproduces the real runtime. | Same bundle, workload and scenario in the DES and over real sockets: relative difference of p50 latency, and per-window agreement of the chosen representation. | p50 within 10 % and agreement ≥ 85 % | Report the gap; recalibrate the cost model |

## Multiple testing

One-sided tests of each threshold over folds, Holm–Bonferroni across H1–H7 at family-wise 0.05.
Effect sizes and confidence intervals are reported for every hypothesis regardless of the test outcome.

## Known limits of this pass (stated before the results)

- **Few folds.** Training is expensive; the number of folds and repeats actually run is reported with
  every result. With fewer than the planned 5 folds × 20 repeats the intervals are wide.
- **Six healthy bearings.** Each fold has only two or three healthy bearings for training and one
  each for predictor training and calibration, so the bearing-level exchangeability of the
  certificate rests on very few machines.
- **H7 and H6 without a Raspberry Pi** compare the simulator with a real-socket run on the same
  laptop, not with separate hardware. They are labelled as such.
- **Synthetic cellular traces.** The 3G / 4G / 5G traces are synthetic and labelled as such.

## Log of additions after the freeze

No metric, threshold or definition above was changed. These are additions, listed so that nothing
looks pre-registered that was not.

| Date | What | Why | Seen before adding |
|---|---|---|---|
| 2026-10-10 | Gate G1 threshold (reference-model macro-F1 ≥ 0.40) fixed in ADR 0004 | The PRD leaves it to an ADR | Fold r0f0 results of the discarded waveform model only |
| 2026-10-10 | Reference model, heads and bottleneck changed (ADR 0004) | The waveform model did not transfer across bearings | Label accuracy and marginal disagreement of fold r0f0; no E2–E8 result |
| 2026-10-10 | E2 exchangeability control: V0 calibrated and tested on halves of the *same* calibration bearings | H2a failed; the PRD's pivot is "diagnose exchangeability" | **The H2a / H2b results.** The control is a diagnostic and does not alter either verdict |
| 2026-10-10 | H7 is evaluated on two real-socket backends: loopback with pacing (E7) and Docker with `tc netem` (E7d) | The PRD's fallback names Docker; loopback also runs in CI | No sim-to-real result on Paderborn data |
| 2026-10-10 | V2 and `liveness_first` rows are reported next to the pre-registered Kai (S1, V0, `fidelity_first`) in E3 | They show the cost of V0's any-policy guarantee | Synthetic-fixture results only. They are exploratory and do not enter H3 |
| 2026-10-10 | "Pi-class" E5 rerun with edge CPU times × 8 | H4 depends on how slow the edge is; no Pi was available | Nothing; labelled a modelled sensitivity run |
