# ADR 0022 — Exploratory fixes after the 2026-11-01 rerun

Date: 2026-11-01 · Status: accepted · Touches: `bench/calibrate.py`, `bench/evaluate.py`, `bench/experiments.py`,
`analysis/phase.py`, `scripts/report.py`

## Context

After the ADR 0020 / 0021 rerun, H3, H6 and H2a failed on every dataset. Four changes were agreed
with the user. This ADR fixes their rules **before** any of them is computed.

The test data has already been read, so every result under this ADR is **exploratory**. The
pre-registered verdicts in `docs/research/hypotheses.md` stay as they were. The first unseen test
of these rules is SHHS.

## Decisions

### 1. Lossless fallback: checked, not needed

The proposal was to send `r1_lossless` instead of raw when raw is the only certified option.

The code already does this:
- `r1_lossless` is in every menu.
- Its disagreement with raw is exactly 0 on all four rerun datasets (E1).
- The degradation ladder (`select/base.py: ladder`) already picks the cheapest certified,
  deadline-feasible representation. That set includes lossless and the near-lossless wavelet.

Kai holds a window back only when even lossless (0.4–19 KB) misses the deadline. No change is made.

### 2. Best-single's screening, and a gated static fallback

B4 (`select/baselines.py: best_single`) chooses among `marginally_admissible(d_cal, α)`: the
representations whose calibration disagreement passes (n·R̂ + 1)/(n + 1) ≤ α. That check is a
valid expected-risk bound only for one representation named before calibration. B4 applies it to
every representation and keeps the cheapest that passes, so its choice carries no guarantee
(multiple testing). B8 does the same per state. B4's frames are uncertified, and H3 compares Kai
with a cheaper but uncertified choice. The certified fixed-choice comparator is
`kai_s1_v0_static`.

New exploratory policy `kai_s1_auto`:
- **Gate.** Per fold, run a 3-fold cross-validation of the sufficiency predictor on `ptrain`,
  grouped by unit. For each non-raw representation that has both outcomes in a held-out part,
  take the AUC of ĝ_r against d_r. Average these AUCs.
- **Rule.** If the mean AUC is below **0.55**, use `kai_s1_v0_static` (constant scores: one safe
  set, no probe, no predictor). Otherwise use `kai_s1_v0`.
- **Data used.** The gate sees `ptrain` only, before calibration, so CRC validity on `calib` is
  unaffected. It never sees `test`.
- **Computation.** Both branches were already simulated on identical cells and seeds. The
  policy's E3 rows are those of the chosen branch for each fold; nothing is re-simulated.
- **Report.** Its cost against B4, beside the other Kai variants.

### 3. H6: crossings must exceed measurement noise

E4 now records, for every (load, rate, representation) cell, the MAD of the per-window latencies
next to their median.

A measured argmin change between grid points g and g + 1 counts as a boundary only if, at both
points, the latency gap between the two representations exceeds **2·√(MAD_lo² + MAD_hi²)**.
Otherwise the two are tied within noise, and no boundary is placed.

The closed-form side is unchanged. H6's pre-registered verdict is still computed the old way,
and the new count is reported beside it as `H6_noise_aware` (exploratory). This needs a new E4
measurement: the stored E4 rows hold only medians.

### 4. H2a non-inferiority

For each α, a one-sided t-test over folds of H0: mean V0 risk ≥ α + **0.01**. The p-value is the
maximum over α. It is reported as `H2a_noninferiority` (exploratory) and does **not** enter the
pre-registered Holm family.

The margin of 0.01 was chosen by the user before the test was computed. At α = 0.05 the test
asks for a risk of at most 0.06. Folds of different repeats share test units, so the p-value is
optimistic, as for every fold-level test here.

## Consequences

- E4 is re-measured on Paderborn, HAR, Speech Commands and Sleep-EDF. E3 and E5 are not rerun.
- Every report gains an "Exploratory (ADR 0022)" section with the three new numbers.

## Addendum (theory review, 2026-11-02)

The first version of §2 claimed that B4 carries an expected-risk bound of its own and that H3
therefore compares like with like. That was wrong, for the multiple-testing reason now stated in
§2. The docs were briefly changed to say so and have been reverted.

The review also found:
- **The gate.** It picks the cheaper branch on every dataset, except 3 Paderborn folds just above
  the 0.55 cut.
- **The H6 noise rule.** It is strict, about 2.7× a 2-SE rule on two medians of 25 iid windows,
  and so biased towards "tied".
- **The verdict of record.** A re-measured H6 does not replace the pre-registered verdict.
