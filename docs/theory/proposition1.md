# Proposition 1 — certify-then-optimize

Status: proof written by the implementation pass. An automated review on 2026-10-14 found the
earlier statement false when scores tie (`sup` over a continuum, and a continuity claim that does
not hold); the statement and step 3 below are the corrected version, and the code was already
computing the corrected threshold ([ADR 0007](../adr/0007-certificate-review.md)). **A check by a
person other than the author is still owed** (PRD §14.2).
Code: `kai/certify/crc.py`. Empirical check: `tests/property/test_crc_validity.py`.

## Setting

- A window `x`, a menu of `K` representations with `r0` = raw.
- Decision-agreement loss `d_r(x) = 1[ŷ_r(x) ≠ f_raw(x)] ∈ {0, 1}`, with `d_{r0}(x) = 0`.
- Sufficiency scores `ĝ_r(x) ∈ [0, 1]`, `ĝ_{r0} = 0`, computed from the probe `φ(x)` by a predictor that
  was fitted on data **independent of** the calibration and test windows (the `ptrain` partition).
- Safe set `S_λ(x) = {r0} ∪ {r : ĝ_r(x) ≤ λ}` for `λ ∈ Λ`, where `Λ` is a **finite** grid in
  `[−1, 1]` that contains `−1` — in the code, the float64 numbers in that interval.
- Safe-set loss `L(x; λ) = max_{r ∈ S_λ(x)} d_r(x) ∈ {0, 1}`.

## Statement

Let `(x_1, …, x_n, x_{n+1})` be exchangeable, the first `n` being calibration windows. Define

```
R̂_n(λ) = (1/n) Σ_{i ≤ n} L(x_i; λ)
λ̂      = max{ λ ∈ Λ : (n · R̂_n(λ) + 1) / (n + 1) ≤ α }        (λ̂ = −1 if the set is empty)
```

Then for **every** policy `π` — deterministic, randomized, learned, state-dependent — and the
event `C = {π(x_{n+1}, z) ∈ S_λ̂(x_{n+1})}` that its choice lies inside the safe set (the
`certified` flag):

```
E[ d_{π(x_{n+1}, z)}(x_{n+1}) · 1{C} ]  ≤  E[ L(x_{n+1}; λ̂) ]  ≤  α.
```

The bound is per **offered** window. It is not a bound on the disagreement rate *among* certified
decisions, `E[d_π | C]`, which divides by `P(C)`.

## Proof

**Step 1 — the loss is a threshold function.** Put
`m(x) = min{ ĝ_r(x) : d_r(x) = 1 }`, with `m(x) = +∞` if no representation disagrees. Because `r0`
never disagrees, a member of `S_λ(x)` disagrees iff some `r` has `d_r(x) = 1` and `ĝ_r(x) ≤ λ`, i.e.

```
L(x; λ) = 1[ λ ≥ m(x) ].
```

So `λ ↦ L(x; λ)` is non-decreasing, bounded by 1, and `L(x; −1) = 0` because `ĝ ≥ 0 > −1`. (`tests/property/test_safe_set.py::test_loss_equals_threshold_form`.)

**Step 2 — exchangeability of the losses.** `m` is a fixed measurable function of `x` once the
predictor and the heads are fixed, and both were fitted on data independent of
`(x_1, …, x_{n+1})`. Hence `L_i(λ) := L(x_i; λ)`, `i = 1, …, n+1`, are exchangeable random functions.

**Step 3 — conformal risk control.** This is the argument of Theorem 1 of Angelopoulos, Bates,
Fisch, Lei & Schuster (*Conformal Risk Control*, ICLR 2024), written out for a finite grid so that
no continuity of the loss is needed (with `S_λ = {ĝ ≤ λ}` the loss `1[λ ≥ m]` is *not*
right-continuous in the direction their theorem assumes, and a supremum over a continuum would not
be attained when scores tie).

Let `R̂_{n+1}(λ) = (1/(n+1)) Σ_{i ≤ n+1} L_i(λ)` and `λ̂' = max{λ ∈ Λ : R̂_{n+1}(λ) ≤ α}`. The set
contains `−1`, where every loss is 0, and `Λ` is finite, so the maximum is attained and
`R̂_{n+1}(λ̂') ≤ α`.

- Because `L_{n+1} ≤ 1`, `R̂_{n+1}(λ) ≤ (n · R̂_n(λ) + 1)/(n + 1)` for every `λ`. Every `λ` that
  qualifies for `λ̂` therefore qualifies for `λ̂'`, so `λ̂ ≤ λ̂'` (also when `λ̂ = −1`), and by
  monotonicity `L_{n+1}(λ̂) ≤ L_{n+1}(λ̂')`.
- `λ̂'` is a symmetric function of the exchangeable `L_1, …, L_{n+1}`, hence
  `E[L_{n+1}(λ̂')] = E[(1/(n+1)) Σ_i L_i(λ̂')] = E[R̂_{n+1}(λ̂')] ≤ α`.

Together:

```
E[ L(x_{n+1}; λ̂) ] ≤ α.
```

**Step 4 — any policy inside the safe set.** On the event `C`, `π(x_{n+1}, z) ∈ S_λ̂(x_{n+1})`,
therefore `d_π(x_{n+1}) · 1{C} ≤ max_{r ∈ S_λ̂} d_r(x_{n+1}) = L(x_{n+1}; λ̂)` pointwise, whatever
the policy looks at — scores, system state, its own randomness, even the losses themselves. Taking
expectations gives the first inequality. What must **not** depend on the test losses is `λ̂`, the
scores `ĝ` (both fixed before the test window), and the set of windows the average is taken over. ∎

## How the code computes λ̂

With `m_(1) ≤ … ≤ m_(n)` the sorted calibration values and `k = ⌊α(n+1) − 1⌋` the number of
calibration losses that can be afforded:

- `k < 0` (i.e. `n < ⌈1/α⌉ − 1`): even zero observed losses fail the bound, `λ̂ = −1`, the safe set
  is `{raw}`.
- `k ≥ n` or `m_(k+1) = +∞`: `λ̂ = 1`, everything is safe.
- otherwise `λ̂` is the largest float strictly below `m_(k+1)` — the maximum over the float grid
  `Λ` — so that ties at `m_(k+1)` are all excluded and `#{i : m_i ≤ λ̂} ≤ k` holds.

`test_crc_threshold_is_the_largest_valid_one` checks both that the bound holds at `λ̂` and that it
fails at every larger loss point.

## What the proposition does not say

1. **Expectation, not high probability.** A single calibration draw exceeds α in roughly 35–50 % of
   draws in simulation; that is expected and is why V1 (Learn-then-Test) exists.
2. **Exchangeability is an assumption.** Windows of one bearing are autocorrelated and a new bearing
   may be shifted. See `limitations.md`.
3. **Only `certified` decisions, counted per offered window.** Deferred, dropped and `uncertified`
   actions are outside the bound. The experiments report both `risk_certified_joint`
   (`E[d_π · 1{C}]`, the bounded quantity) and `risk_certified` (the rate among certified
   decisions, which can exceed α when few decisions are certified).
4. **The predictor only affects efficiency.** A useless predictor makes `λ̂` small and the safe set
   close to `{raw}`; it never voids the bound.

## Variants

| Variant | Calibrated quantity `m_i` | Guarantee |
|---|---|---|
| **V0** | `min{ĝ_r : d_r = 1}` | `E[safe-set loss] ≤ α`, any policy |
| **V1** | same `m_i`, threshold from exact binomial tests (fixed sequence) | `P(R(λ̂) ≤ α) ≥ 1 − δ`, any policy; assumes i.i.d. windows |
| **V1 per-rep** | `m_i^r = ĝ_r` if `d_r = 1` else `∞`, level `α/(K−1)`, `δ/(K−1)` per rep | same as V1 by the union bound |
| **V2** | first `ĝ` at which the *greedy* choice under nominal costs disagrees | `E[d_greedy] ≤ α` for that policy and cost vector only |

For V2 the true policy loss is not monotone in λ (a cheaper, agreeing representation may enter
later). The code calibrates its monotone envelope — the loss is held at 1 from the first λ at which
the greedy choice disagrees — which upper-bounds the true loss, so the guarantee is conservative.

V2's threshold is calibrated for the greedy choice under **one frozen cost vector** (the scenario's
nominal state: nominal bandwidth, empty queue). In the simulator the selector sees the live
bandwidth estimate, backlog and remaining deadline, so its choice can differ from the calibrated
one; V2's threshold is at least V0's, so the any-policy bound does not cover it either. V2 rows in
the scenario grid (E3) are therefore exploratory even though their decisions carry the `certified`
flag; the E2 check uses the nominal cost vector and is the only place where V2's guarantee applies.

The audit-in-the-loop threshold (ACI) has no finite-sample guarantee at all; see `limitations.md`.

## The cascade (ADR 0011)

Two certified sets combine by the union bound. If `F` and `A(x)` are each calibrated by V0 at α/2
(each a fixed score function fitted on `ptrain`, each threshold from `calib`), then for every
window `max_{F ∪ A(x)} d ≤ max_F d + max_{A(x)} d`, hence `E[max_{F ∪ A(x)} d] ≤ α`, and step 4
covers any policy inside the union — also one that decides from the system state whether to
compute ĝ, as long as it sends a member of `F` when it does not. More generally, any score that is
a fixed function of the window, fitted on data independent of `calib` and `test`, can replace ĝ
without changing the proof.
