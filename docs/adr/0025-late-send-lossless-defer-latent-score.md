# ADR 0025 — Send late, defer lossless, latent sufficiency score (exploratory)

Date: 2026-11-09. Status: accepted. **Exploratory**: decided after the pre-registered H3 results
were seen (H3 fails on every dataset, including Icentia11k on the corrected simulator). Adds
policies; changes no pre-registered policy, metric, threshold or verdict. `kai_s1_v0` stays as
registered.

## Context

On Icentia11k (`docs/research/results-icentia.md`, corrected simulator, ADR 0023) Kai (S1, V0)
costs 145 % more than best-single (B4). Three causes are visible in the report and the code:

1. **Deferral.** When no safe representation meets the deadline, `fidelity_first` caches the raw
   window and forwards it later (`select/base.py: ladder`, `link/des.py`). The realised cost then
   carries the raw bytes and a lateness of up to 10× the deadline. At 64 kb/s Kai defers 44 % of
   windows, at 256 kb/s 24 %; from 1 Mb/s up Kai is cheaper than B4. B4 never defers: it sends its
   one representation and pays one window's lateness.
2. **Raw as the deferred payload.** Store-and-forward ships raw (8.2 kB). The lossless
   representation (r1) gives the same decision by construction (bit-identical samples) at about a
   quarter of the bytes.
3. **A weak score.** The sufficiency predictor sees only the probe features φ; its unit-grouped AUC
   on `ptrain` is 0.59–0.65. The oracle picks r6 (16 B) in every cell, but the safe set rarely
   contains it.

## Decision

Three changes, each added as a new policy, measured on its own and together. The rules below are
fixed before any of them is computed.

### A. Send late instead of deferring (`certified_late` mission)

Ladder rung 2 becomes: if no allowed representation is feasible, send the cheapest allowed
representation that is sendable — energy and storage constraints hold and the link rate is
positive, the deadline is ignored — flagged `certified`. The cost already prices lateness
(`p_late · max(0, L − D)`). Only when nothing allowed is sendable (link down, energy or storage)
does the window defer. The choice stays inside `S_λ̂(x)`, so Proposition 1 (any policy inside the
safe set) is unchanged; a missed deadline is an SLO event, as before. Windows that used to be
deferred are now certified, so `rate_certified` rises; the rate of decisions certified *and* on time
(`rate_certified_on_time`) is reported next to it.

### B. Defer the lossless copy

A deferred window is cached and forwarded as r1 (lossless) instead of raw: its own payload size,
its encode time charged to the edge at deferral, its decode time at the cloud. Its decision equals
raw's (d = 0 by construction); the flag stays `deferred`, outside the certificate.

### C. Latent sufficiency score

ĝ_z is the same predictor (`fit_predictor`, MLP, fit on `ptrain` only) on `[φ(x), z(x)]`, where
`z(x)` is the dequantised r6 latent of the fold's own r6 encoder — what the cloud head would see.
The r6 encoder is trained on the fold's `train` units only, disjoint from `ptrain`, `calib` and
`test`; ĝ_z is fit on `ptrain`; λ̂ is V0 on `calib`. The scores are a fixed function of the window
given data disjoint from calibration and test, so Proposition 1 applies by its general clause (`docs/theory/proposition1.md`, scores that are a fixed function of the window): expected risk ≤ α per offered window, certified decisions only, under exchangeable units. The edge runs the
r6 encoder on every window: its encode time is added to the per-window overhead, and r6's own
encode time is then zero (the latent is already computed). Folds without r6 get no C policies.

### Policies (E3 main grid, every dataset)

| policy | A | B | C |
|---|---|---|---|
| `kai_s1_v0_late` | ✓ | | |
| `kai_s1_v0_lldefer` | | ✓ | |
| `kai_s1_v0_latent` | | | ✓ |
| `kai_s1_v0_abc` | ✓ | ✓ | ✓ |

### Reported, for each of the four, on every dataset

- Cost reduction versus B4 and versus `kai_s1_v0`: the H3 statistic (fold-level, 95 % block
  bootstrap CI), labelled exploratory; no Holm family, no verdict.
- SLO violation, `rate_deferred`, `risk_certified_joint` per α (what Proposition 1 bounds).
- For C: the mean held-out AUC of ĝ and of ĝ_z per non-raw representation on `calib`
  (descriptive), and E2's V0 risk with ĝ_z per α.
- All four are reported, whichever wins. None replaces `kai_s1_v0` in any pre-registered table.

### Scope

Simulator only (E2, E3). The edge runtime does not implement A, B or C yet
(`shortcut:` comments mark it); a sim-to-real (E7) or confirmatory run needs them first.

## Addendum (2026-11-09 evening): C′, per-representation score choice

Added after C's first E2 result on Icentia11k (exploratory on top of exploratory; C stays
reported as specified). There, ĝ_z raised r6's held-out AUC on `calib` from 0.626 to 0.746 but
lowered most others (mean 0.632 → 0.616), and its cheapest safe payload grew (0.29 versus 0.17 of
raw at α = 0.02).

**C′.** For each non-raw representation k, choose between ĝ_k (φ only) and ĝ_z,k ([φ, z]) by
unit-grouped 3-fold cross-validation on `ptrain` only (same predictor kind and seed; the held-out
AUC of each against d_k, mean over the splits; ties and representations with one class in a
held-out part go to ĝ). The final score of k is the chosen one, refit on all of `ptrain` (the ĝ
and ĝ_z already fitted). λ̂ is V0 on `calib`. The choice uses `ptrain` only, so the scores remain a
fixed function of the window given data disjoint from `calib` and `test`, so Proposition 1 applies by its general clause (`docs/theory/proposition1.md`, scores that are a fixed function of the window): expected risk ≤ α per offered window, certified decisions only, under exchangeable units.
The r6 encode is charged on every window only if at least one representation uses ĝ_z.

| policy | A | B | C′ |
|---|---|---|---|
| `kai_s1_v0_zsel` | | | ✓ |
| `kai_s1_v0_ab_zsel` | ✓ | ✓ | ✓ |

Reported like the other four (same statistics), plus which representations chose ĝ_z per fold.
The SHHS pre-registration is a new dated entry in `docs/research/hypotheses.md`, written before the SHHS data is downloaded (theory review, 2026-11-09). It names one of the Kai variants of ADR 0025 and ADR 0026 and freezes α, grid, folds, seed, any memory budget, the test and the Holm family (descriptive hypotheses at p = 1). H3 has two parts — (a) cost ≥ 20 % below B4 and (b) SLO violations ≥ 30 % below B5 — and a send-late variant equals B5 on Icentia11k, so it fails (b) by construction; the entry must say whether it tests the full H3 or a named cost-only H3′. The choice of variant is made on the six exploratory datasets: its risk and cost there are post-selection (the cheapest variant is favoured by lucky calibration draws, CIs unadjusted); only SHHS tests it.

## Consequences

- H3 stays failed on every dataset; these results are exploratory and are labelled so in
  `updates.md` and in the paper (dagger).
- A confirmatory test is possible only on data not yet seen: SHHS (NSRR request pending). Before the
  SHHS data is downloaded, a dated addition to `docs/research/hypotheses.md` names one of the Kai
  variants of ADR 0025 and ADR 0026 and its test (see the addendum); the choice may use these exploratory results.
- C costs every window one r6 encode on the edge (0.40 ms on the laptop for Icentia); this is in
  the cost, so a weak gain in score quality can lose to the overhead.
