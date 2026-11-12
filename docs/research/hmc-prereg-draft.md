# HMC confirmatory pre-registration — DRAFT (not frozen)

Drafted 2026-11-12. **Not frozen.** ADR 0028 §4 freezes this text after the remaining ADR 0028 runs
(Icentia11k, HAR, Speech Commands, Paderborn presets, E4/E7, reduced E3k) and **before** HMC is
downloaded. At freezing, the text moves into `docs/research/hypotheses.md` with its date. Every item
marked **[OPEN]** must be settled then. Any change made after seeing those runs is listed with its
reason and what had been seen. HMC itself must not be downloaded or read before freezing.

It follows the SHHS pre-registration of 2026-11-10 (`hypotheses.md`), which stays as a second
confirmation if SHHS access arrives. HMC is the same task as Sleep-EDF (sleep staging from one EEG
channel) but from another clinic, montage and scorer. It tests replication, not generality.

**Data.** HMC sleep staging v1.1 (PhysioNet, CC BY 4.0), all 151 subjects, one night each. The
channel is `EEG C4-M1` at 256 Hz. Each night is resampled to 102.4 Hz with
`resample_poly(x, 2, 5)`, then cut into 3072-sample windows on the 30-s scoring grid. Labels are
W / N1 / N2 / N3 / REM. The "Lights off" / "Lights on" markers are skipped; any other non-stage
annotation is an error. 30 min of wake are kept on each side of the sleep period. Samples are
quantised to int16 at ±250 µV full scale (`kai.data.hmc`). A night whose channel is missing or
unreadable is dropped and counted before fold assignment, and is not replaced.

**Protocol.** The unit is the subject (`SNxxx`). 5 leave-subjects-out folds, one repeat, training
seed 0, `make_l1_splits` defaults as for Sleep-EDF. f_raw is chosen per outer fold (ADR 0028 §7):
the candidates `wave` and `spec` are trained on that fold's `train` units. The one with the higher
macro-F1 on the fold's `ptrain` units wins; a tie goes to the smaller model. Every candidate's score
is recorded. `calib` and `test` never take part in the choice. α ∈ {0.02, 0.05, 0.10}, δ = 0.1,
the E3 grid of `configs/scenario/grid.yaml`, price preset `metered_cellular`, mission
`fidelity_first`. `test` is read once.

**Edge setting (S-H3′ only).** Every edge time × k and CPU power ÷ k, so energy per window is
unchanged (ADR 0027). **[OPEN]** k = 10 as for SHHS, unless the reduced E3k gives a reason to
change it. Any change is post-selection and must say so.

| ID | Statement | Metric and test | Threshold |
|---|---|---|---|
| **S-H3′** Cost on a slow edge | Kai (`kai_s1_v0`) is cheaper than both the best single representation (B4, uncertified) and f_raw on the edge (b14). | (a) `1 − total_Kai / total_B4` and (b) `1 − total_Kai / total_b14`, per fold, each over the grid cells feasible for both policies of that pair (`c_summary`; ADR 0028 §4: "revised S-H3′ on feasible cells"). Source: `kai exp e3k k_grid=[10] max_folds=5` (`e3_k10`, 5 folds). One-sided test of each threshold over folds; p = max(p_a, p_b). | (a) ≥ 0.20 and (b) ≥ 0.10 **[OPEN: keep or revise]** |
| **S-H2a** Certificate | As for SHHS: in expectation over calibration draws, under exchangeable subjects, V0 keeps the mean safe-set risk of certified decisions within α + 0.01 (non-inferiority, ADR 0022). | Safe-set risk per fold for each α; one-sided test that the fold mean is below α + 0.01; p = max over α. | mean ≤ α + 0.01 for every α |
| **S-H5** Heterogeneity | The full menu beats the best within-family controller. | `1 − total_Kai / min(total_compression-only, total_learned-only)` (E5, laptop edge timings, α = 0.05), per fold; one-sided test. | ≥ 0.10 |

**Tests, multiple testing, reporting.** As in the SHHS pre-registration, word for word:
- `one_sided_p`: a t-test over the 5 fold values, df = 4.
- The intersection–union p is the maximum of its component p-values.
- All 5 folds enter every test. No fold is re-run with a new seed.
- The Nadeau–Bengio p is a sensitivity check only.
- CIs: `bca_ci` over folds and `block_bootstrap_ci` by subject.
- Holm–Bonferroni over {S-H3′, S-H2a, S-H5} at family-wise 0.05, within HMC only.
- Supported = the threshold is met **and** the Holm-adjusted p < 0.05.

**Reported, not tested.** The same list as for SHHS, plus each fold's f_raw choice and all
candidates' `ptrain` scores.

**[OPEN] at freezing.**
1. k for S-H3′, and the S-H3′ thresholds.
2. A statement of what the remaining ADR 0028 runs showed, written before freezing.
3. The `theory-reviewer` pass over every sentence that states a guarantee.
