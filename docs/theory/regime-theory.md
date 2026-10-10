# Closed-form regime theory (contribution C3)

Code: `kai/analysis/phase.py`. Tests: `tests/unit/test_analysis.py`. Experiment: E4.

## Latency model

With an empty uplink queue, the end-to-end latency of representation `r` is

```
L_r(B, ρ) = a_r(ρ) + 8·(S_r + 28) / B
a_r(ρ)    = t_ovh/(1−ρ) + t_enc,r/(1−ρ)^γ_r + RTT/2 + t_dec,r + t_head,r
```

`S_r` is the payload in bytes, 28 the frame overhead, `B` the throughput in bit/s, `ρ` the edge CPU
load from other tenants, `t_ovh` the probe + predictor + selector time, and `γ_r` the load exponent
fitted by the profiler (`γ = 1` is the processor-sharing model).

Refinement over the PRD formula: the per-window overhead `t_ovh` is also divided by `1 − ρ`, because
the probe, the predictor and the selector share the same CPU as the encoder.

## Break-even bandwidth

Take a compact representation `i` and a larger one `j` (`S_j > S_i`). Their latencies are equal when

```
B*_ij(ρ) = 8·(S_j − S_i) / Δt_ij(ρ),      Δt_ij(ρ) = a_i(ρ) − a_j(ρ)
```

- `Δt_ij > 0` (the compact one is slower to produce or to decide on): for `B < B*` the compact
  representation wins on latency, for `B > B*` the larger one does.
- `Δt_ij ≤ 0`: the compact representation is also faster; it wins at every bandwidth (`B* = ∞`).

Because the encode term of `Δt` grows as `1/(1−ρ)^γ`, **B\* shrinks as edge load rises**: a heavy
encoder pays off only on ever-slower links. When only encode time differs, `B*(ρ) = B*(0)·(1−ρ)^γ`.

## The phase diagram is a lower envelope

In `u = 1/B` every `L_r` is a straight line with intercept `a_r` and slope `8(S_r + 28)`. The
fastest representation as `u` grows is the lower envelope of these lines: starting at the smallest
intercept (infinite bandwidth) and moving to lines of ever smaller slope. Consecutive envelope
members `i, j` meet exactly at `B*_ij`. A representation that is not on the envelope at load `ρ` is
never the fastest at that load, whatever the bandwidth.

`latency_envelope(model, ρ)` returns the ordered boundaries; `argmin_map` evaluates the partition on
a grid with a deadline mask and, optionally, by cost instead of latency.

## Energy analogue

With radio energy `e_bit` per bit and active edge power `P_cpu`, the compact representation `i` uses
less energy than `j` iff

```
e_bit · 8·(S_j − S_i)  >  P_cpu · (t_edge,i(ρ) − t_edge,j(ρ))
```

i.e. iff `e_bit > P_cpu · Δt_edge / (8·ΔS)`. On an energy-priced preset this is the cost boundary.

## What the theory leaves out

- **Queueing.** With backlog `Q` the transmission term is `8(Q + S_r + 28)/B`; `Q` shifts every line
  by the same amount, so the *latency* boundaries do not move, but deadline feasibility and the
  backlog growth of large payloads do. The simulator and the MPC selector model this; the closed
  form does not.
- **The safe set.** The envelope is over the representations admitted for a window. A cheap
  representation that is rarely safe occupies its region of the diagram only for some windows.
- **Sub-timeslice work.** Under contention a sub-millisecond encoder is either not preempted at all
  or delayed by whole scheduler slices: the *mean* follows `1/(1−ρ)`, the median does not. The
  profiler therefore fits `γ` on means and reports both.

H6 tests the envelope boundaries against latencies measured over real sockets (±20 %).
