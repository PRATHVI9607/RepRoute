# ADR 0007 — Review of the certificate: what was wrong and what changed

Date: 2026-10-14 · Status: accepted · Touches: `docs/theory/`, `certify/crc.py` (comments),
`link/des.py`, `edge/runtime.py`, `bench/evaluate.py`, `bench/experiments.py`, `analysis/stats.py`,
`scripts/report.py`

An automated, read-only review (the `theory-reviewer` subagent) went through Proposition 1 and the
code under `certify/`, the calibration glue and the statistics, line by line. It is **not** the
independent human check PRD §14.2 asks for; that is still owed. Its findings were checked against
the code before anything was changed. The calibration code (`crc_threshold`, `first_loss_threshold`,
the LTT procedures, the V2 envelope) was found correct. The problems were in the written statement,
in what the experiments report, and in the audit loop.

## Findings and what was done

| # | Finding | Severity | Action |
|---|---|---|---|
| 1 | **Proposition 1 was false as written when scores tie.** It defined λ̂ as a `sup` over a continuum and claimed the loss is right-continuous in the direction the cited theorem needs. With `S_λ = {ĝ ≤ λ}` the loss is `1[λ ≥ m]`; the supremum of the feasible set is `m_(k+1)` itself, which is not feasible. Ties are real: the predictor outputs a constant for a representation that never or always disagrees on `ptrain`. The code already took the largest float *below* `m_(k+1)`. | blocker (text) | Statement and step 3 rewritten for a finite grid with a direct exchangeability argument; `crc.py` comments aligned. No behaviour change. |
| 2 | **The reported `risk_certified` is not the bounded quantity.** The proof bounds `E[d_π · 1{certified}]` per offered window; the experiments reported the rate *among* certified decisions, which divides by the certified rate. | should fix | New column `risk_certified_joint` in every grid result and in the report; the proposition says which one is bounded. The clause requiring the policy's state to be independent of the losses was vacuous and is replaced by what really must not depend on them (λ̂, the scores, the set averaged over). |
| 3 | **V2 is used outside its guarantee in the scenario grid.** Its threshold is calibrated for the greedy choice under one nominal cost vector; the simulator's selector sees live state. | should fix | Stated in the proposition and under the E3 table. E2's V2 check uses the nominal cost vector and stands. |
| 4 | **The audit loop could lock at raw-only.** Only non-raw decisions were audited, so once the adaptive threshold reached raw-only no feedback ever arrived again; and the audited decisions were not a uniform subsample, contrary to the text. | should fix | Raw decisions are now audited with the same probability at no byte cost (raw agrees with itself) in the simulator and in the edge runtime. Test: `test_aci_leaves_raw_only_again_because_raw_decisions_count_as_agreements` fails on the old code. **This changes E6's audit-in-the-loop numbers.** |
| 5 | BCa intervals over five folds under-cover; a degenerate sample returned p = 0 (certainty) in `one_sided_p`. | should fix | `one_sided_p` returns 1 without spread. BCa is kept (it is the registered plan) and its under-coverage is stated in the limitations. |
| 6 | **Holm ran over four hypotheses, and verdicts ignored the test.** H5 was printed as "supported" with a Holm-adjusted p of 0.94. | should fix | The family is all eight hypotheses (untested ones enter with p = 1). The report calls a tested hypothesis *supported* only if its threshold is met **and** Holm rejects; otherwise "threshold met, not significant". H1's p is 1 when its untestable first part fails. |
| 7 | E3 / E5 risks were pooled over α. | should fix | Per-α table of `risk_certified_joint` in the report. |
| 13, 15 | CI endpoints from two bootstrap draws; the exchangeability control used one random halving. | nit | One draw; the control averages 20 halvings and its comment no longer says "same bearings" is why it holds (a random halving is exchangeable by construction). |
| 9, 10 | The `1e-9` in `crc_threshold`; "no loss is affordable" for `k < 0`. | nit | Comment added; wording fixed. |

Not acted on in this pass (left for the certification owner):

- **Predictor cache** is keyed by fold directory only. Folds are written to fresh directories here, so
  no stale predictor was read, but `kai folds --force` into an existing directory would reuse it.
- `no_certificate` and `audit_aci` decisions carry the `certified` flag in the ablation.
- `kai_s1_v1` in E3 uses the per-representation thresholds, while H2b is defined on the
  single-threshold V1.
- Under `liveness_first`, `uncertified` sends also feed the audit loop.
- A bearing-level certificate at α = 0.05 would need at least 19 calibration bearings; a fold has
  about five.

## Effect on earlier results

- The verdict rule in finding 6 is stricter than the one used on 2026-10-10. Under it the earlier
  "H5 supported" would have read "threshold met, not significant". The rule only ever weakens a
  claim; it is logged in `docs/research/hypotheses.md`.
- Numbers of earlier runs stay in the registry unchanged.
