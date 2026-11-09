# ADR 0021 — Fixes from the 2026-11-01 code review

Date: 2026-11-01 · Status: accepted · Touches: `edge/runtime.py`, `cloud/server.py`,
`cloud/heads.py`, `contracts/frame.py`, `link/framing.py`, `link/traces.py`, `link/des.py`,
`certify/crc.py`, `certify/aci.py`, `certify/ltt.py`, `profile/profiler.py`, `reps/registry.py`,
`reps/logmel.py`, `tasks/pipeline.py`, `tasks/train.py`, `bench/*`, `scripts/report.py`

## Context

A whole-repo review (four read-only reviewers, one per plane) found defects that bias reported
numbers, plus robustness and speed issues. The fixes were chosen from the code, not from which
one moves a result; every result reported before this ADR stays recorded as it was.

## Decision

Measurement (changes E1, E4, E7 numbers):
- Edge pacing no longer blocks the event loop (`time.sleep` → awaited deadline with a 2 ms
  yield-spin), so acks are read while a frame is paced. Window arrival is the scheduled time
  `t_start + n·period`, and the deadline shrinks by the lag, as in the DES. `edge_ms` times only
  the decision and the send. RTT uses `perf_counter_ns` with a discarded warm-up ping.
- The cloud writes and drains the reply before running the telemetry callback.
- `profile_menu` times decode and head inside one paced call and takes each part's median.

Certificate plane (changes E5, E6; E2 unchanged except on NaN input):
- `first_loss_threshold` and the threshold functions raise on NaN instead of returning a silently
  anti-conservative or NaN λ.
- ACI receives feedback only for `certified` decisions (DES and edge), and is documented as
  steering the realised risk of the policy's certified choices towards α; with ACI on, the
  any-policy bound does not hold. λ lookup no longer re-sorts.
- LTT's binomial tail is computed incrementally (same values up to float rounding at the δ
  boundary; property test).

Robustness (no result change): stale head files are removed on save; a 4-byte head-hash collision
raises; log-mel refuses sample rates below 4 kHz and wrong payload lengths; trace rates must be
finite and non-negative (NaN made `finish_time` loop forever); the k = 1 estimator keeps one
sample; encode caches are written atomically, keyed per representation, and filled via memmap;
the global torch determinism flag is restored after training; edge and cloud close sockets in
`finally`, log swallowed exceptions and stop draining when the stream closes.

`contracts/frame.py`: `unpack_parts(header, rest)` checks the CRC as
`crc32(payload, crc32(header))` without concatenating and copies the payload once. The wire format
is byte-identical (`test_frame.py` unchanged; new property test for split decoding and bit flips).

Statistics and harness (changes reported verdict inputs, not thresholds):
- H1(b) is the pre-registered mean over folds of `total_B4 / total_oracle − 1`; the code took a
  harmonic-type mean of `1 − o/b`.
- A fold whose reference total is 0 is no longer dropped: both 0 → reduction 0; Kai > 0 → −inf
  (a fail). Dropping it biased H3(b) towards a pass.
- H7 fails when any rate has unmeasured (NaN) rows; E4 counts cells where nothing was delivered.
- Bootstrap blocks in E2 are bare unit ids (not `fold/unit`): a bearing tested in several repeats
  is one block. The t-tests and BCa intervals stay over folds, as pre-registered. Folds of
  different repeats share test bearings (every repeat tests every bearing once), so fold-level
  p-values are optimistic; this is recorded as a limitation, not corrected. Averaging folds within
  a repeat was tried during this review and reverted: repeat means differ only by partition, so
  their spread understates the variance over new bearings.
- E5 `audit_aci` was a no-op (no audits ever reached ACI); only policies whose certifier runs ACI
  now receive audits (`p_audit`), so the other E5 rows are unchanged.
- E6 skips a split level without a manifest; E5 honours `max_folds`; E4 honours `reuse`; reuse is
  refused when the stored config hash or split id differ; `risk_certified` is NaN when nothing was
  certified; E8 times 30 blocks (median + MAD); E2 adds `risk_unit_mean`; report counts classes
  and conditions from the data instead of hard-coded text.
- Speed: grid chunks never mix folds or sensor counts and the policy zoo is built lazily
  (toy grid 2.1× faster; ≈ 1 % on real folds, where the DES and cost model dominate).

E7 replays windows in real time, so its length scales with the window period. Each link rate
now replays at most 30 min: `e7_windows` keeps 300 windows unless that exceeds the budget, with a
floor of 60. Only Sleep-EDF (30 s epochs) is affected: 60 windows per rate instead of 7.5 h in
total. The count is in the E7 `backend` string of every report.

## Consequences

- E1 and everything downstream are rerun on every dataset after this ADR.
- The encode caches are rebuilt once; trained folds (`table.npz`, heads, f_raw) stay valid.
- A pipeline whose predictor emits NaN now stops instead of certifying on a wrong threshold.
