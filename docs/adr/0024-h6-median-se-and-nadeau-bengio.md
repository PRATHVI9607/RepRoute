# ADR 0024 — H6 median-SE crossing rule and a Nadeau–Bengio sensitivity column (exploratory)

Date: 2026-11-09. Status: accepted. Both items are **exploratory**: they were decided after the
pre-registered results were seen. They add rows and columns to the reports; they change no
pre-registered verdict, metric or threshold.

## Context

- ADR 0022 fix 3 counts a measured argmin change in E4 only when the latency gap exceeds
  `2·√(MAD_lo² + MAD_hi²)` at both grid points. No crossing survived on any dataset. That rule uses
  the per-window MAD, about 2.7× a two-standard-error rule on the difference of two medians, so it
  is biased towards "tied" (updates.md §7). The handoff of 2026-11-09 lists a median-SE rule as the
  next step.
- Fold-level p-values are optimistic because folds share test units (repeats) and training units
  (always). The pre-registered test is a plain one-sided t-test over folds; this is kept.

## Decision

### 1. H6 median-SE rule (fixed before it is computed)

For each measured argmin change between two adjacent bandwidth grid points, and for the two
representations involved, compute at **both** grid points

```
SE_r   = 1.2533 · 1.4826 · MAD_r / √n_r          (standard error of a median, normal model)
gap    = |median_hi − median_lo|
count the change only if gap > 2 · √(SE_lo² + SE_hi²) at both grid points
```

`n_r` is the number of windows measured for that representation at that grid point (from the stored
E4 rows; if they do not store it, the configured windows per point). Matching against the closed
form, the 20 % threshold and "≥ 1 matched boundary" are exactly H6's. Reported next to the
pre-registered count and the ADR 0022 MAD rule, labelled exploratory. A finer bandwidth grid is
**not** part of this ADR.

The rule is applied to (a) the stored E4 measurements, by re-analysis, and (b) one E4
re-measurement on a quiet CPU after the ADR 0023 reruns. The H6 verdict of record stays the
pre-registered run (updates.md §7); neither (a) nor (b) replaces it.

### 2. Nadeau–Bengio corrected resampled t-test (sensitivity only)

For each hypothesis tested over folds (H1, H2a, H3, H5), with per-fold values `d_1 … d_J` and the
same one-sided direction as the pre-registered test:

```
t  = (mean(d) − threshold) / √((1/J + n_test/n_train) · s²),    df = J − 1
```

`n_test / n_train` is the number of test units over the number of non-test units of one fold
(from the split manifest). Holm over the same eight-hypothesis family, within one dataset. Shown
as an extra column "p (N–B, Holm)" in the multiple-testing table. **No verdict uses it.** It
accounts for overlapping training sets; it does not make folds independent.

## Consequences

- New summary keys from E4 (`H6x/median_se/…`) and new report lines; a new column in the
  multiple-testing table.
- If the median-SE rule finds matched crossings, the report says "exploratory, not a verdict".
  If it finds none, H6 stays untested at this grid, as before.
- References: C. Nadeau, Y. Bengio, *Inference for the Generalization Error*, Machine Learning 52
  (2003); R. Bouckaert, E. Frank, PAKDD 2004 (verify before citing).
