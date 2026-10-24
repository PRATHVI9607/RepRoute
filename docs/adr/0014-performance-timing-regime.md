# ADR 0014 — Pin the timing regime to the `performance` profile

Date: 2026-10-24 · Status: accepted · Touches: measurement environment and the edge-work
baseline; no model or threshold change

## Context

The edge profile moved materially between runs (overhead 0.90 → 1.053 ms, R6 0.784 → 0.970 ms)
while the laptop was on the `powersave` governor. ADR 0013 recorded a passing E7 loopback result
(1.1 / 5.0 / 4.4% p50 gap) and a failing E4 phase result (82.4% boundary error) on that profile.
A pass that depends on an unpinned CPU frequency regime is not a stable result.

The laptop uses `amd-pstate-epp` with `power-profiles-daemon` active. A direct
`cpupower frequency-set -g performance` is temporary: the daemon reapplies its current profile.
The durable knob is `powerprofilesctl set performance`, which sets all CPUs to the `performance`
governor and the `performance` energy-performance preference. On this 16-CPU machine the result
was 16/16 performance governors and 16/16 performance EPP settings.

Changing the governor during the experiment suite contaminated one timing-sensitive run (E8).
The suite was rerun in full under the pinned profile; E8 and all profile-dependent results were
replaced. No result was silently mixed across regimes.

## Decision

1. The final local timing regime for the edge work is the `performance` profile managed by
   `power-profiles-daemon`. Every timing result must record the regime.
2. `make experiments` is run after the governor is pinned, starting with the E1 profile. The
   resulting bundle is rebuilt before E7d. The `powersave` results stay in the registry but are not
   the baseline.
3. H7 is evaluated on both backends. A loopback failure is not repaired by changing the capacity
   model or the 10% threshold after seeing the result.

## Consequences

Under `performance` the edge is faster. E4 now passes H6, while loopback E7 fails H7:

| Result | `powersave` (`801b7bb6c1a9`) | `performance` (`bb809ab9f5c8`) |
|---|---|---|
| E1 overhead | 1.053 ms | 1.0715 ms |
| E4 / H6 | false, 82.4% max boundary error | true, 4.7% max representative-boundary error |
| E7 loopback / H7 | true, 5.0% max p50 gap | false, 20.5% and 42.3% max p50 gap in two runs |
| E7d Docker / H7 | true, 6.1% max p50 gap | true, 2.2% max p50 gap |

The loopback result is not a contradiction: E7d includes a fixed 10 ms netem delay, so the edge
term is a smaller share of total latency and its bias is diluted. Loopback has no such delay and
exposes the DES's high-rate overprediction. Selection agreement is 1.0 in both E7 and E7d. A second
E7 run produced 23.3 / 38.4 / 42.3%, so the real edge time is not stable enough for a precise gap;
both runs fail and neither is converted into a pass.

The new profile is:

- probe + predictor + selector 1.0715 ms;
- `t_io` 0.050 ms (still at the floor);
- R6 encode 0.900 ms;
- load-0 excess −0.371 ms, load-0.8 excess 1.897 ms, `io_load_exponent` 2.259;
- E8 13.7 µs per greedy decision, 43.0 / 75.3 / 294.2 µs per MPC h3/h4/h5 decision.

## Validation

`make experiments` completed under 16/16 `performance`; Docker E7d read the new bundle
`bb809ab9f5c8` and completed 300 windows per rate. `make figures` and `make report` were run after
the suite. The E7 failure is recorded in `docs/research/results.md` and the open work in
`docs/edge-work.md`; it is not converted into a pass.

## Limits

- One profile is not a claim that the E7 gap is stable to a percentage point. Two E7 runs under the
  same bundle measured 20.5% and 42.3% maximum gaps. The load-0 calibration remains negative and
  `t_io` remains at its floor.
- Thermal throttling can still change timings under sustained load even with the governor pinned.
  The laptop should be on AC power and ventilated; a future run should record package temperature.
- The loopback and Docker backends answer different questions. A H7 pass in Docker must not be
  reported as a loopback pass, and vice versa.
- The Pi and other lab machines are separate timing regimes. Their profiles must record their own
  governor and hardware; this ADR does not transfer the laptop profile to them.
