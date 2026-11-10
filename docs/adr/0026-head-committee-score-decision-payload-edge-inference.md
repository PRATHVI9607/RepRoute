# ADR 0026 — Head-committee sufficiency score, decision payloads, edge-inference baseline (exploratory)

Date: 2026-11-09. Status: accepted. **Exploratory**: decided after the pre-registered results and
after the ADR 0025 results on Icentia11k were seen. Adds a score, policies and a baseline; changes
no pre-registered policy, metric, threshold or verdict.

## Context

- ADR 0025 on Icentia11k: sending late instead of deferring (A) turns Kai's cost from 145 % above
  best-single (B4) to 17.9 % below it (CI [7.6, 23.2] %), identical to B5. The oracle still costs
  44 % less than that. What separates them is the score: with the probe features φ the predictor's
  mean held-out AUC on `calib` is 0.63 (r6: 0.63), and r6 (16 B) enters the safe set for 2 % of test
  windows at α = 0.05.
- A feasibility study on the stored Icentia11k fold tables (predictor fit on `ptrain`, AUC on `calib`, V0 risk and payload on `test`) used the cloud heads' decisions of r3–r7 as the predictor's input. The held-out AUC and the certified payload improved markedly; subsets of the heads lost most of the gain; certifying only the chosen representation (V2 with a size order) added little, so the score, not the certificate form, is the bottleneck. The study ran outside the report pipeline, so its numbers are not quoted; E2 regenerates them (`X26/auc/…`). **It used Icentia's calib and test data to design D (and to choose the full committee over subsets): Icentia's D results are design-in-sample, not a held-out check.**
- Running the whole f_raw on the edge takes 0.63–0.76 ms on the laptop, sends one byte and never
  disagrees. On this hardware it dominates every policy; Kai is useful only where the edge cannot
  host f_raw. That premise is not explicit in the scenarios.

## Decision

Fixed before it is computed on any dataset except Icentia11k, where the feasibility study
already looked (see Context).

### D. Head-committee score

The committee is every representation of the fold's menu whose cloud head takes features
(r3–r7 here; r10 where present). For each window the edge encodes it with each committee member,
decodes it as the cloud does (`decode_for_head`) and runs that member's cloud head, giving class
probabilities p_r(·|x). The predictor input is `[p_r for r in committee] ⊕ votes`, where votes
counts, per class, the members whose argmax is that class. ĝ_D is `fit_predictor` (MLP, same seed)
on `ptrain` only; λ̂ is V0 on `calib`. Heads and encoders are trained on the fold's `train` units,
disjoint from `ptrain`, `calib` and `test`, so ĝ_D is a fixed function of the window given data
disjoint from calibration and test, so Proposition 1 applies by its general clause (`docs/theory/proposition1.md`, scores that are a fixed function of the window): expected risk ≤ α per offered window, certified decisions only, under exchangeable units.

Edge cost: the committee's encode and head times (the measured cloud head time is used as the
edge's) join the per-window overhead; a committee member's own encode is then free when it is sent.
The argmax of each p_r must equal the stored ŷ_r on (nearly) every window; the run fails if more
than 0.1 % differ.

### Decision payload

Once the edge has run a member's head, it knows that member's decision. The variant sends it
(1 byte) instead of the member's payload: size 1 byte, no encode, decode or head time at the cloud.
Its loss is computed from the decision the edge actually sends, `d_dec,r = 1[argmax p_r ≠ f_raw]`,
on `calib` (for λ̂_dec, V0 with ĝ_D) and on `test` alike, so no edge-versus-cloud mismatch enters
the bound (theory review: a mismatch rate would otherwise add to α). This makes Kai a certified edge-or-cloud
cascade; the paper must say so where the variant is reported.

### Edge-inference baseline and an edge memory budget

`b14_edge_inference`: run f_raw on the edge and send its decision (1 byte, d = 0); no probe or
predictor. Its edge time and the committee's head times come from the same profile measurement
(the stored per-representation head time on this machine). E3 runs without a memory limit and
records the model bytes each policy needs on the edge (file bytes of the stored model files, one
convention for all): f_raw for b14; the committee's encoders and heads plus the predictor for D;
the menu's encoders plus the predictor for the other Kai policies. The report sweeps the edge
budget over those sizes and shows, at each, the cheapest policy that fits and where b14 and D
cross. No budget is chosen (theory review: a budget set after seeing the sizes decides the
comparison in advance). File bytes are a proxy for runtime memory.

### Policies

| policy | score | mission | deferral | payload |
|---|---|---|---|---|
| `kai_s1_v0_committee` | ĝ_D | profile's | raw | representation |
| `kai_s1_v0_late_committee` | ĝ_D | `certified_late` | lossless | representation |
| `kai_s1_v0_late_committee_dec` | ĝ_D | `certified_late` | lossless | decision for committee members |
| `b14_edge_inference` | — | — | — | f_raw decision |

### Reported, on every dataset

- E2: V0 risk and cheapest safe bytes with ĝ_D per α; held-out AUC of ĝ_D on `calib` per
  representation and mean, next to ĝ's.
- E3: cost reduction versus B4 and versus `kai_s1_v0` (the H3 statistic and CI) for the three Kai
  policies; b14's cost versus B4 and versus the best of them; the budget sweep;
  `rate_certified_on_time` next to `rate_certified`.
- Model sizes per dataset (f_raw, each encoder and head, the predictors).
- All are reported whichever wins; none replaces a pre-registered policy.

### Scope

Simulator only (E2, E3). The edge runtime needs the committee (including a numpy evaluator for the
LightGBM heads, since `kai.edge` may not import lightgbm) before E7 or a confirmatory run; only for
the variant chosen for SHHS.

## Consequences

- H3 stays failed on record. The SHHS pre-registration (ADR 0025 addendum) may choose among the
  ADR 0025 and 0026 variants, using these exploratory results, before the SHHS data is downloaded.
- The committee costs about 3.2 ms per window on the laptop for Icentia11k (five encodes and heads),
  more than f_raw itself; it can pay only where f_raw does not fit the edge budget.
- With decision payloads the bytes on the wire no longer carry a representation the cloud could
  reuse; any use of the payload beyond the decision (logging, retraining) is lost for those windows.
