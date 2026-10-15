# ADR 0011 — The cascade selector: pay for the probe only when it can pay for itself

Date: 2026-10-15 · Status: accepted (exploratory: added after the results of ADR 0010's re-run)
· Touches: `certify/runtime.py`, `bench/calibrate.py`, `select/policies.py`, `link/des.py`,
`edge/runtime.py`, `edge/bundle.py`, `bench/bundle.py`, `bench/evaluate.py`

## Problem

The re-run after the code audit showed two regimes with opposite answers:

- **Bytes priced.** One 16-byte representation is sufficient almost everywhere. The per-window
  certificate (V0) is noisy for it — the predictor cannot tell the few windows where it disagrees
  — so it excludes it from about 30 % of safe sets, and those windows wait. A certified *static*
  safe set, which needs no probe at all, is far cheaper.
- **Edge CPU priced.** The oracle needs two representations, and per-window selection is 73 %
  cheaper than the best single one (5 folds).

Neither the static set nor the per-window set is right in both. Running the probe on every window
is a fixed tax of about 2.3 ms; not running it gives up per-window choice.

## Decision

**One certificate, two speeds.**

1. *Two certified sets, each at α/2.* A static set `F` — the V0 certificate with constant scores
   (each representation's disagreement rate on the `ptrain` bearings), calibrated on `calib` at
   α/2 — and the ordinary per-window set `A(x) = {r : ĝ_r(x) ≤ λ̂_A}`, V0 at α/2
   (`bench.calibrate.cascade_certificate`).
2. *Their union is certified at α.* For every window `max_{r ∈ F ∪ A(x)} d_r ≤ max_{F} d_r +
   max_{A(x)} d_r`, so `E[loss of F ∪ A] ≤ α/2 + α/2` by the two ordinary CRC statements. No
   independence between the two calibrations is needed: the union bound holds window by window.
   Any policy that chooses inside `F ∪ A(x)` is covered (Proposition 1, step 4).
3. *Free set.* `F` is the same for every window and known before it arrives. It is encoded for
   `Certifier` as a prior: members of `F` get the score −1, the others keep ĝ, and the threshold
   is λ̂_A (`Certifier.free_set`).
4. *Value-of-information gate* (`select.policies.CascadeSafe`). For each window, cost the free
   members without probe and predictor, and every other representation *with* them. If the
   cheapest feasible free member is no more expensive than the cheapest feasible non-free one,
   send it — certified, no probe. Otherwise run probe and predictor, build `F ∪ A(x)`, and choose
   by the ordinary ladder. Whether to probe may depend on the system state; both paths choose
   inside the certified set.

The simulator charges probe and predictor only on probed windows (`Result.probed`, metric
`probe_rate`); the real edge runtime skips them the same way (`Edge._decide`). Bundles carry
`cascade.json` (prior and λ̂_A per α); `kai edge --selector s1_cascade` uses it. ACI and the drift
monitor are refused for the cascade: ACI re-reads thresholds calibrated without the prior, and
the drift monitor needs probe features on every window, which the cascade skips.

**A first version, rejected on the certificate experiment.** One threshold on the score
`min(ĝ_r(x), c_r)` is also valid, but the noisy per-window scores of the poor representations used
up the risk budget, pushed λ̂ below the 16-byte latent's rate and so removed it from the free set
at α ≤ 0.05 (cheapest safe payload 8.9 % of raw at α = 0.05 against 0.2 % for the static set).
Splitting the budget keeps the two parts from spoiling each other. Both versions were run; only the
union version is in the results.

## Why this and not something else

- *One calibration at the full α* (the rejected first version) wastes the budget on the poor
  representations' noise; the union bound costs a factor of two in each part's α but each part
  spends its own budget only on what it certifies.
- *A learned gate* (bandit on whether to probe) needs feedback the edge does not have and adds no
  guarantee. The gate here is a cost comparison the cost model already makes.
- *Probing every k-th window* lowers the tax but leaves the safe set stale between probes.

## Evidence

Tests: `test_cascade_scores_keep_the_guarantee_and_the_free_set_is_always_safe` (validity on a
synthetic population, free set ⊆ safe set for every window),
`test_cascade_serves_from_its_free_set_only_when_probing_cannot_pay` (the gate, property-based),
`test_cascade_pays_for_the_probe_only_on_the_windows_where_it_runs` (simulator charging, both
regimes), `test_cascade_on_the_edge_skips_the_probe_and_stays_inside_the_safe_set` (real sockets).

Results: E2 row `v0_cascade`; E3 policy `kai_s1_cascade` under every price preset, with its probe
rate ([results.md](../research/results.md)). Exploratory: no hypothesis was registered for it, and
H3 stays defined on Kai with S1 and V0.
