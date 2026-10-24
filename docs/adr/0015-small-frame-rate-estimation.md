# ADR 0015 — Keep the configured prior for sub-1000-byte frames

Date: 2026-10-24 · Status: accepted · Touches: no runtime change; records the rate-estimation
decision for the edge work

## Context

The selected representation is `r7_vq`: a 16 B payload, 44 B on the wire. `Edge._acks` only adds a
goodput sample when a frame is at least `MIN_SAMPLE_BYTES` (1000 B), because the ACK timing noise is
larger than the serialization time of a small frame. At 1 Mbit/s a 44 B frame takes 0.35 ms to
serialize; the loopback and container event loops add roughly the same amount of jitter. A sample
from such a frame would mostly measure the scheduler.

The edge therefore keeps `HarmonicMeanEstimator(initial_bps)` unless a larger frame is sent
(deferred raw, an audit copy, or a recovery probe). E7 and E7d set `initial_bps` to the known link
rate, so their selection is not blind. A real deployment without that knowledge would start from
the 1 Mbit/s default and would need a deliberate startup calibration or a larger probe frame.

## Decision

1. Do not lower `MIN_SAMPLE_BYTES`, and do not infer throughput from a single small-frame ACK in
   this pass. The prior remains the fallback.
2. Do not add a padding or probe frame merely to feed the estimator. It would add traffic and a
   new experiment, and would change the measured E7/E7d workload after the fact.
3. Record `initial_bps` as part of the deployment configuration. The current experiments already
   set it to the rate they are testing; a future field deployment must do the same or accept the
   conservative default.
4. Revisit this only with a separate pre-registered experiment comparing ACK aggregation, a
   periodic probe, and the prior. That experiment is outside the current edge pass.

## Consequences

- No change to E7 or E7d, so their results remain comparable with the current bundle.
- The estimator can be wrong when the prior is wrong. That is a deployment limitation, not a
  hidden calibration: the chosen `initial_bps` is visible in the edge configuration.
- A larger frame (audit or deferred raw) can still update the estimate, so the prior is not
  permanent when the workload sends one.

## Limits

- This is a decision to keep a known conservative behavior, not evidence that the prior is
  accurate for every link.
- It does not resolve the loopback E7 model bias; that is recorded separately in ADR 0014.
- Any future probe policy must account for its own bytes, SLO cost and certificate risk.
