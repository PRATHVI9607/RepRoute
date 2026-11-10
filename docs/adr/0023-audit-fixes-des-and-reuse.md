# ADR 0023 — Audit fixes: simulator probe interval, delivery-time accounting, reuse provenance

Date: 2026-11-09. Status: accepted. Found by a whole-repository audit (three read-only reviews,
each finding checked against the code before it was accepted).

## Context

The audit found one implementation bug that changes results, and several that could.

1. **Probe interval (changes results).** `bench.evaluate.run_cell` built `Scenario` from 10
   positional arguments; the dataclass has 11 fields, so `seed` (0 in `configs/config.yaml`) was
   stored in `probe_s` and `Scenario.seed` stayed 0. With `probe_s = 0` an idle link forwarded
   cached raw windows back to back instead of once per second, as the edge runtime does
   (`edge/runtime.py`), and live frames queued behind them. Present since the first commit, in every
   E3, E5 and E6 run on every dataset. E7's scenarios set every field correctly and are not
   affected.
   It penalises policies that defer (Kai V0, cascade, static set, B11); B4 never defers.
   (E7 passes its first scenario fields positionally too, onto the right fields; it now uses
   keywords as well.)
2. **Billing and learning at the wrong time.** The DES billed a frame's bytes when it was queued,
   so cached windows never sent by the end of a run were charged; ACI and the AdaEdge baseline
   learned from an audit at decision time, before the audit copy was delivered; the audit copy's
   I/O time was not charged to the edge. The edge runtime does all three at delivery.
3. **Reuse provenance.** `reuse=true add_policies=[…]` could mix fold sets (a fold missing from one
   policy became a reduction of 0.0) and timing profiles (the profile was not in the provenance);
   results without a provenance sidecar were reused with a warning; trained folds were not tied to
   the windows file they were trained on.
4. Smaller: the report printed E1's capped timing period as the window period (HAR, Speech
   Commands, Sleep-EDF said 2.0 ms per window at 32 sensors); the seed table joined by row
   position; the oracle flagged choices outside the safe set as `certified`; the edge CLI read CPU
   load as an all-core average; guards for empty calibration groups and single-group AUC.

## Decision

- `Scenario` is built with keywords after its first two fields; a test checks `run_cell` keeps
  `probe_s = 1.0` and passes the seed.
- The DES bills bytes, and feeds ACI and learning baselines, when a frame is delivered; an audited
  decision pays the copy's I/O time, as on the edge.
- `provenance()` adds the fold list and the sha256 of the timing profile; `fold_reduction` refuses
  a fold missing from either side; reuse without a sidecar is refused; `kai folds` stores a windows
  fingerprint per fold and refuses a mismatch.
- The other items are fixed as listed, each with a test.

## Consequences

- Every E3, E5 and E6 result so far was simulated with `probe_s = 0`. They are rerun on the five
  final-code datasets (Paderborn, HAR, Speech Commands, Speech Commands + log-mel, Sleep-EDF), with
  E4 and E7 re-measured afterwards on a quiet CPU; Icentia11k runs on the fixed code from its E3 on.
  HUST, UORED and PTB-XL keep their earlier-code results, labelled as such.
- This is an implementation-bug fix, as in ADR 0010, not a change of metric, threshold or
  definition. Verdicts of record are recomputed, by the same rules, only for DES-based results
  (E3, E5, E6: H1, H3, H4, H5). Every one of these is reported under both simulator versions, with
  the reason when it changes; the earlier runs stay in the registry. Both fixes move cost in the
  same direction (in favour of deferring policies), which is stated next to the comparison.
- E4 runs no DES, and E7's simulator side is not affected by the probe bug. Their re-measurements
  are exploratory and replace no verdict of record (H6 and H7 keep theirs, ADR 0024).
- HUST, UORED and PTB-XL results are never pooled with corrected-code results.
- Audit feedback now arrives on delivery, as on the edge. Copies still queued at the end of a run
  give none; their share is reported per cell (`rate_audit_unfed`).
- Stored results from before this ADR are refused by `reuse=true`, which is intended.
