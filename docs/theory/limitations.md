# Limitations of the guarantee

These go into the paper's "Limitations" section (PRD §6.4). Each item names what the code does
about it and what it cannot do.

## 1. Exchangeability

The formal guarantee is for a test window exchangeable with the calibration windows.

- Windows from one bearing and one recording are autocorrelated. Calibration and test windows are
  non-overlapping (hop = N) and come from **different bearings** than training and
  predictor-training. That removes leakage; it does not make windows independent.
- A new bearing may be shifted relative to the calibration bearings. With the Paderborn data a fold
  has about five calibration bearings, so at the level that matters — the machine — the calibration
  sample is tiny. The window count (thousands) overstates the evidence.
- What is reported for the certificate experiment (E2): per-fold risk, never a pooled number
  alone; confidence intervals from a **bearing-block bootstrap**
  (`kai.analysis.stats.block_bootstrap_ci`), conditional on each fold's calibration set of about
  five bearings; H2 as an empirical finding across folds, never as a theorem about new machines.
  The scenario-grid tables (E3, E5) pool cells and folds and are broken down by α next to them.
- Fold-level intervals use the BCa bootstrap over five folds, which under-covers (about 85 %
  for a nominal 95 % interval in a normal simulation) and treats folds that share training
  bearings as separate units. Read them as rough.
- V1's binomial p-values assume i.i.d. windows and are therefore optimistic under autocorrelation.

## 2. Drift

- A probe-feature drift monitor (`kai.certify.drift`, per-feature two-sample KS test with a
  Bonferroni correction) switches the runtime to raw-only when recent windows stop looking like the
  calibration windows. It detects covariate shift in φ, not a change in `P(d | φ)`.
- Audit-in-the-loop: with probability `p_audit` the edge also sends raw, the cloud reports whether
  the chosen representation agreed, and Adaptive Conformal Inference moves the working level.
  The update runs on the audited subsequence rather than with inverse-probability weights, whose
  1/p spikes make the threshold oscillate. Every sent decision is audited with the same
  probability: a non-raw one by shipping a raw copy, a raw one for free (it agrees with itself),
  so the audited decisions are a uniform subsample and the threshold can leave raw-only again
  (ADR 0007). ACI tracks the **realised policy risk**, has no finite-sample guarantee, and its
  bytes are counted in every result.

## 3. Deadline misses are not disagreements

A late, deferred or dropped window is an SLO event. It is reported as `slo_violation` and through
the `deferred` / `dropped` rates, never folded into risk.

## 4. Predictor quality limits savings, not validity

A weak sufficiency predictor shrinks the safe set towards `{raw}`. The bound still holds; the
system simply saves nothing.

## 5. What the reference is

The certificate controls agreement with `f_raw`, the cloud's raw-data model — not correctness.
Where `f_raw` is wrong, a representation that agrees with it is "sufficient" and wrong too. Label
fidelity (macro-F1 against ground truth) is reported next to every agreement number.

With the Paderborn data `f_raw` is EnvNet (ADR 0006), a low-capacity model chosen because nothing
larger transferred across bearings. Its macro-F1 on unseen bearings is in
[results.md](../research/results.md); roughly a quarter of unseen bearings are misclassified as a
whole, and the certificate inherits every one of those errors.

**R6 runs the reference model's front-end on the edge.** The 16-byte supervised bottleneck is
`f_raw`'s fixed feature extraction plus a learned projection. It agrees with `f_raw` because it is
most of `f_raw`. Any claim that a 16-byte payload "is sufficient" must be read as "the edge computed
the cloud model's features"; the generic learned latents (R5, R7) and the engineered features (R3,
R4) do not come close to agreeing with EnvNet on unseen bearings.

**The cascade has no drift protection.** It skips the probe on windows its free set serves, so
the probe-feature monitor would see too few windows, and its safe mode could lock in. The cascade
refuses the monitor and ACI (ADR 0011); under a suspected shift, use the per-window certificate
with the audit loop.

## 6. Limits specific to this implementation pass

- **Edge hardware.** Timings were measured on a development laptop. Nothing has run on the
  Raspberry Pi yet; H7 and H6 compare the simulator with a real-socket run on the same machine.
- **Energy is modelled** (`P_cpu · t_edge + e_bit · bits`), with order-of-magnitude constants.
  No energy number in this pass is a measurement.
- **Cellular traces are synthetic.**
- **Head family.** Every feature and latent head is a standardised logistic regression or a
  LightGBM model chosen with the same search budget; the supervised bottleneck's own jointly
  trained head is not used on the cloud.
- **Prior-art baselines B9–B11** are conceptual re-implementations adapted to this setting.
