# Proposition 1 — certify-then-optimize

Status: proof written by the implementation pass; **an independent check by Role E is still owed**
(PRD §14.2). Code: `kai/certify/crc.py`. Empirical check: `tests/property/test_crc_validity.py`.

## Setting

- A window `x`, a menu of `K` representations with `r0` = raw.
- Decision-agreement loss `d_r(x) = 1[ŷ_r(x) ≠ f_raw(x)] ∈ {0, 1}`, with `d_{r0}(x) = 0`.
- Sufficiency scores `ĝ_r(x) ∈ [0, 1]`, `ĝ_{r0} = 0`, computed from the probe `φ(x)` by a predictor that
  was fitted on data **independent of** the calibration and test windows (the `ptrain` partition).
- Safe set `S_λ(x) = {r0} ∪ {r : ĝ_r(x) ≤ λ}` for `λ ∈ Λ = [−1, 1]`.
- Safe-set loss `L(x; λ) = max_{r ∈ S_λ(x)} d_r(x) ∈ {0, 1}`.

## Statement

Let `(x_1, …, x_n, x_{n+1})` be exchangeable, the first `n` being calibration windows. Define

```
R̂_n(λ) = (1/n) Σ_{i ≤ n} L(x_i; λ)
λ̂      = sup{ λ ∈ Λ : (n · R̂_n(λ) + 1) / (n + 1) ≤ α }        (λ̂ = −1 if the set is empty)
```

Then for **every** policy `π` — deterministic, randomized, learned, state-dependent — such that
`π(x, z) ∈ S_λ̂(x)` and whose state `z` and internal randomness are conditionally independent of
`(d_r(x_{n+1}))_r` given `(x_1, …, x_{n+1})`:

```
E[ d_{π(x_{n+1}, z)}(x_{n+1}) ]  ≤  E[ L(x_{n+1}; λ̂) ]  ≤  α.
```

## Proof

**Step 1 — the loss is a threshold function.** Put
`m(x) = min{ ĝ_r(x) : d_r(x) = 1 }`, with `m(x) = +∞` if no representation disagrees. Because `r0`
never disagrees, a member of `S_λ(x)` disagrees iff some `r` has `d_r(x) = 1` and `ĝ_r(x) ≤ λ`, i.e.

```
L(x; λ) = 1[ λ ≥ m(x) ].
```

So `λ ↦ L(x; λ)` is non-decreasing, right-continuous, bounded by `B = 1`, and `L(x; −1) = 0`
because `ĝ ≥ 0 > −1`. (`tests/property/test_safe_set.py::test_loss_equals_threshold_form`.)

**Step 2 — exchangeability of the losses.** `m` is a fixed measurable function of `x` once the
predictor and the heads are fixed, and both were fitted on data independent of
`(x_1, …, x_{n+1})`. Hence `L_i(λ) := L(x_i; λ)`, `i = 1, …, n+1`, are exchangeable random functions.

**Step 3 — conformal risk control.** Theorem 1 of Angelopoulos, Bates, Fisch, Lei & Schuster
(*Conformal Risk Control*, ICLR 2024) is stated for losses that are non-increasing in the
parameter. Apply it to `λ' = −λ`: `L_i` is non-increasing and right-continuous in `λ'`, bounded by
`B = 1`, and satisfies `L_i(λ'_max) = 0 ≤ α` at `λ'_max = 1`. Their threshold
`inf{λ' : (n/(n+1))·R̂_n + B/(n+1) ≤ α}` is exactly `−λ̂`. The theorem gives

```
E[ L(x_{n+1}; λ̂) ] ≤ α.
```

**Step 4 — any policy inside the safe set.** For every realisation, `π(x_{n+1}, z) ∈ S_λ̂(x_{n+1})`,
therefore `d_{π}(x_{n+1}) ≤ max_{r ∈ S_λ̂} d_r(x_{n+1}) = L(x_{n+1}; λ̂)` pointwise. Taking
expectations gives the first inequality. The independence condition on `z` is what makes this a
statement about the policy and not about an oracle: a state that could see `d_r` could still only
pick inside the safe set, so the bound would continue to hold, but nothing stronger is claimed. ∎

## How the code computes λ̂

With `m_(1) ≤ … ≤ m_(n)` the sorted calibration values and `k = ⌊α(n+1) − 1⌋` the number of
calibration losses that can be afforded:

- `k < 0` (i.e. `n < ⌈1/α⌉ − 1`): no loss is affordable, `λ̂ = −1`, the safe set is `{raw}`.
- `k ≥ n` or `m_(k+1) = +∞`: `λ̂ = 1`, everything is safe.
- otherwise `λ̂` is the largest float strictly below `m_(k+1)`, so that ties at `m_(k+1)` are all
  excluded and exactly `#{i : m_i ≤ λ̂} ≤ k` holds.

`test_crc_threshold_is_the_largest_valid_one` checks both that the bound holds at `λ̂` and that it
fails at every larger loss point.

## What the proposition does not say

1. **Expectation, not high probability.** A single calibration draw exceeds α in roughly 35–50 % of
   draws in simulation; that is expected and is why V1 (Learn-then-Test) exists.
2. **Exchangeability is an assumption.** Windows of one bearing are autocorrelated and a new bearing
   may be shifted. See `limitations.md`.
3. **Only `certified` decisions.** Deferred, dropped and `uncertified` actions are outside the bound.
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
