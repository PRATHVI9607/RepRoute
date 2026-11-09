# ADR 0019 — Time the cloud as the server runs it; take cloud time out of the goodput sample

Date: 2026-11-01 · Status: accepted · Touches: `profile/profiler.py` (`profile_menu`),
`edge/runtime.py` (`_acks`)

## Context

After the pacer fix (ADR 0013 addendum) and ADR 0017, Speech Commands E7 still failed H7: p50 gap
17.3 % at 5 Mbit/s (simulator slower than the real loopback), choice agreement 84 % at 256 kbit/s.
Fitting `p50 = F + 8S/bps` to the 1 M and 5 M rows gives the same serialised size in both
(≈ 2 011 B sim, 2 042 B real) but a fixed part of 6.21 ms in the simulator and 4.76 ms real. Two
modelling errors explain it; neither was chosen by looking at which fix closes the gap:

1. `profile_menu` timed decode and head as separate calls, each after an idle period, so each
   started cold. The server runs them back to back (`HeadZoo.decide_timed`). The DES charges
   `t_dec + t_head` = 2.84 ms for r2b; the real ack timing implies ≈ 1.7 ms. This is ADR 0008's
   principle ("time each stage where it runs") applied to the cloud.
2. The edge's goodput sample for a frame sent into an idle link is `now − t_send − rtt`. The ack
   also waits for the cloud's decode and head, so the sample under-reads the link (3.33 M at
   5 M; 905 k at 1 M). The DES samples pure serialisation. The reply already carries `cloud_ms`.

## Decision

- `profile_menu` times `decode → head` in one call (`both`) at the same period and reports
  `t_head_ms = median(both) − median(decode)`, so `t_dec_ms + t_head_ms` is what the server spends.
- `_acks` subtracts `reply.cloud_ms` from the idle-link sample.

## Consequences

- E1 and everything downstream are rerun on every dataset reported after this ADR. Results from
  before it stay recorded (`updates.md` §4: H7 failed at 17.3 % / 84.0 %).
- Not changed: the edge-side stage sum (≈ 0.3 ms over the in-pipeline timing at one sensor) and
  the knife-edge at 256 kbit/s, where a raw frame takes 1 025.7 ms against a 1 024 ms period.
  If H7 still fails, it fails on those, and is reported as failed.
