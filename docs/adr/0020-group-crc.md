# ADR 0020 — V0 calibrated with one sample per unit (group CRC), exploratory

Date: 2026-11-01 · Status: accepted (exploratory) · Touches: `certify/crc.py`
(`group_crc_threshold`), `bench/evaluate.py` (`FoldCtx.group`, policy `kai_s1_v0_group`),
`bench/experiments.py` (E2 variant `v0_group`)

## Context

Window-level V0 exceeds α on 3 of 15 Paderborn folds and 3 of 5 Speech Commands folds (mean
0.0386 and 0.0515 at α = 0.05). CRC bounds the *expected* risk, not each fold's: simulating
`crc_threshold` under exact iid exchangeability (n = 400 and 2 000, α = 0.05 and 0.1) gives a mean
risk at α and P(fold risk > α) ≈ 0.47 (theory review, 2026-11-01). The exceedance counts are
therefore what the guarantee predicts, and `folds_within_alpha` is descriptive only. Separately,
windows of one unit (bearing, speaker, subject) are dependent and test units are new units, so
window-level exchangeability is an assumption the split does not give. Calibrating per unit
matches the split, with one caveat: where calibration units are stratified by class (Paderborn:
1 K, 2 KA, 2 KI bearings in fold r0f0), units are exchangeable only within a stratum.

## Decision

Add `group_crc_threshold(m, groups, α)`: the loss of a calibration unit is its mean window loss
`L_j(λ) = mean_{i ∈ j} 1[λ ≥ m_i]`, which is bounded by 1 and monotone in λ, and λ̂ is the largest
float with `(n_g·mean_j L_j(λ) + 1)/(n_g + 1) ≤ α` (CRC, Angelopoulos et al., ICLR 2024, with
one sample per unit). For exchangeable units this bounds the expected mean loss of a new unit.
With fewer than ⌈1/α⌉ − 1 units it returns `RAW_ONLY`; with one window per unit it equals
`crc_threshold` (property test `test_group_crc_threshold_is_the_largest_valid_one`).

It is evaluated as an extra variant (`v0_group` in E2, `kai_s1_v0_group` in E3). It does not
replace V0, is not in the bundle or on the edge, and enters no pre-registered hypothesis.

## Consequences

- The guarantee is per unit, weighted equally; E2 reports window-weighted risk, which differs
  when units have different numbers of windows.
- Paderborn and HAR have too few calibration units (5 bearings on Paderborn) and fall back to raw
  only at every α. Sleep-EDF falls back at α ≤ 0.05 and has enough at α = 0.1 (rerun 2026-11-01:
  risk 0.032, 5/5 folds). Only Speech Commands (≈ 500 calibration speakers per fold) can use it at
  every α; its per-unit-mean risk at α = 0.05 is 0.050 (0.051 with log-mel), on α.
- Dry check on the existing Speech Commands folds at α = 0.05 (before the rerun): mean test risk
  0.048 against 0.052 for window-level V0; 2 of 5 folds stay just above α (0.054, 0.056). It is an
  expectation bound, not a per-fold one. E2 should also report the per-unit-mean risk, the
  quantity this variant bounds.
- Headline certificate for the paper stays the cascade (ADR 0011), which is within α on 15/15
  Paderborn and 5/5 Speech Commands folds.
