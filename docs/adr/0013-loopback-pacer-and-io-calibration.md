# ADR 0013 — A precise loopback pacer and a fresh `t_io` calibration

Date: 2026-10-24 · Status: accepted · Touches: `edge/runtime.py`, `bench/experiments.py`,
`tests/unit/test_edge_pacing.py`

## Context

E7 (loopback sockets, in-process token-bucket pacing) was 28.5 / 29.9 / 38.1 % above the DES at
256 kbit/s / 1 Mbit/s / 5 Mbit/s. The real edge CPU time already matched the profile: the residual
was entirely after the frame was handed to the link.

The pacer used `asyncio.get_running_loop().call_later(delay, writer.write, buf)`. On this Linux
host asyncio's selector tick quantises a short timer to about 1 ms: a 0.1 ms `call_later` fires
after 1.03 ms (median over 20 trials). A 44-byte frame takes 0.07–1.38 ms to serialise at the E7
rates, so the timer added a roughly constant 0.8–1.0 ms that the DES does not contain. Writing the
same frame directly, without the timer, removed the residual; the remaining event-loop/socket cost
was about 0.3 ms.

The DES already separates the two transport terms: `CostModel.latency_ms` charges
`(backlog + size + 28) * 8 / bandwidth` for bytes on the wire, and `Overheads.t_io_ms` for the
runtime's fixed per-frame event-loop/socket work. The bug was in the loopback emulator, not in that
split.

## Decision

1. `Edge._send` paces the in-process loopback link with a precise `time.sleep` instead of an
   asyncio timer. It returns the time it waited, and `Edge.run` subtracts that wait from `edge_ms`
   so `edge_ms` remains CPU time, as the profiler and the DES define it.
2. This applies only when `shape_bps` is set. Docker E7d uses `tc netem` (`shape_bps=None`) and is
   unchanged.
3. E1's `_calibrate_io` is rerun after the fix. `t_io_ms` remains the fixed per-frame
   event-loop/socket overhead measured against the real runtime; the byte-dependent term stays
   `tx_ms` in `CostModel.latency_ms`. No hypothesis, metric, or threshold changes.

The blocking sleep is deliberate. The edge has no other work to do while the current frame waits
for an idle link, and the wait is at most one small frame's serialization time. This is an
emulator for the loopback experiment, not a production transport.

## Consequences

- E7 and E7d must be rerun after `make profile` rebuilds the bundle, because the profile carried
  the pre-fix `t_io` and the bundle ID changes.
- A diagnostic run after the fix showed a p50 residual near 0.2 ms (about 7–9 % of the E7
  latencies); the formal E7 rerun is the validation, not this diagnostic.
- If the formal rerun still misses the 10 % threshold, the result stays failed and the component
  table is reported as a negative result. `t_io` is not tuned against E7 after seeing that result.

## Validation

After `make profile` and the bundle rebuild (`801b7bb6c1a9`), E7 passed twice with max p50 gaps of
5.4% and 5.0% and selection agreement 1.0. E7d Docker passed at 6.1% max and agreement 1.0. The
profile moved between runs (overhead 0.90 → 1.053 ms, R6 0.784 → 0.970 ms), so this is one profile,
not a claim that the gap is stable to a percentage point; the edge work log records the caveat.

## Addendum (2026-10-24, pinned `performance` profile)

The validation above is specific to the earlier `powersave` profile. After the timing regime was
pinned to `performance` ([ADR 0014](0014-performance-timing-regime.md)), the edge became faster:
E4 passed H6, but loopback E7 failed H7 (20.5% and 42.3% max p50 gap in two runs) while Docker E7d
still passed at 2.2%. The pacer decision is unchanged; the backend-sensitive H7 result is recorded
in ADR 0014 and `docs/research/results.md`.

## Limits

- The calibration still uses a 20 Mbit/s link and the smallest representation; E7 tests the other
  rates. A rate-dependent residual that the fixed `t_io` cannot capture remains a possible cause of
  failure.
- `time.sleep` blocks the edge event loop for the frame's serialization time. A future multi-frame
  pacer should use a dedicated thread or timerfd; E7 has one frame per 64 ms window.
