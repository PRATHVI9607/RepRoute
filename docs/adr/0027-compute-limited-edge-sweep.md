# ADR 0027 — Compute-limited edge sweep (exploratory)

Date: 2026-11-10. Status: accepted. **Exploratory**: decided after the ADR 0025 and ADR 0026
results were seen. Adds an experiment; changes no pre-registered policy, metric, threshold or
verdict.

## Context

ADR 0026 added `b14_edge_inference` (f_raw on the edge, one byte sent). On all six datasets it is
cheaper than every Kai variant (`X26/b14_edge_inference/*` in each `e3/summary.json`). Its model
files are smaller than Kai's menu of encoders plus predictor, so the memory-budget sweep never has
a Kai policy as the cheapest that fits. All edge times are the laptop's measurements: Kai's premise —
the edge cannot afford f_raw — is not represented in the grid.

A timing-parity defect also came up: `load_profiles(edge_slowdown=k)` scaled encode, probe,
predictor, selection and I/O times but not the head times that b14 and the ADR 0026 committee run
on the edge. ADR 0026's runs used k = 1 and are unaffected.

## Decision

Fixed before anything is computed.

- **Parity fix.** `load_fold(edge_slowdown=k)` passes k to `edge_model` and `committee_models`: f_raw's
  head time (b14) and the committee heads' times are multiplied by k, like every other edge time.
  Cloud-side times are unchanged.
- **Experiment `e3s`.** The E3 main grid with edge slowdown k ∈ {10, 100} (a Pi- or phone-class
  edge and an MCU-class edge). CPU power is `p_cpu_w / k`, so the energy per window equals the
  laptop's and only time grows (a slower, lower-power chip); bytes and link energy are unchanged.
  Prices are the default preset (metered cellular). The first 5 folds of each dataset.
- **Policies:** `b4_best_single`, `b5_context_blind`, `b14_edge_inference`, `kai_s1_v0`,
  `kai_s1_v0_late`, `kai_s1_v0_lldefer`, `kai_s1_v0_abc`, `kai_s1_v0_ab_zsel`, the three ADR 0026
  committee policies, and the oracle.
- **Reported** per dataset and k: each policy's cost reduction versus B4 and versus b14 (fold-level,
  95 % block-bootstrap CI), grid-total cost per fold and SLO violation rate, in
  `experiments/e3_slow<k>/summary.json`. All are reported, whichever wins. No Holm family, no
  verdict.

## Consequences

- If a Kai variant beats b14 at some k, the SHHS pre-registration (ADR 0025 addendum) names that
  variant *and* the edge setting, with b14 as a named comparator. If none does at k = 100, the paper
  states that on these benchmarks local inference dominates and Kai applies only where f_raw cannot
  run on the edge at all.
- The slowdown is a model of a slower device, not a measurement: it scales every edge time by the
  same factor, whereas real devices slow different operations (convolutions, trees, I/O) by
  different factors. The Raspberry Pi run (`scripts/pi_setup.sh`) remains the measured check.
- Choosing the variant and k from these results is post-selection on the six exploratory datasets;
  only SHHS tests the choice.

## Addendum (2026-11-10, after the first results): send-raw baseline

Added after the first e3s results were seen. At k = 100 the edge saturates: every policy that
encodes, runs f_raw or runs the committee misses most deadlines (SLO violations 0.77–0.97 on HAR)
and its cost is dominated by capped lateness, identical across folds; `kai_s1_v0` stays cheap by
sending raw, which needs no edge compute. At k = 10, `kai_s1_v0` is cheaper than b14 on HAR,
Sleep-EDF and Speech Commands with log-mel. Neither can be read without a send-raw baseline, which
the policy list above lacked. `b1_raw` is therefore added to both runs (`reuse=true
add_policies=[b1_raw]`; no other row is re-simulated) and every policy is also compared with it
(`<policy>/cost_reduction_vs_b1`). B1 is a post-hoc comparator and is labelled so.
