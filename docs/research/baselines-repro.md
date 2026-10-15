# Baselines — what was implemented, and how far it is from the cited work

Code: `src/kai/select/baselines.py`, wired into the experiments by `kai.bench.evaluate.policies`.
None of B9–B11 is the original authors' code; each is an adaptation of the *idea* to this
project's setting (one representation chosen per sensor window). Read a comparison against them as
"against this kind of method", not "against that paper's system".

Every baseline is charged only for the per-window work it really does (`overhead` in the table):
`none` runs nothing, `select` a table lookup, `probe` the 20 probe features, `full` the probe and
the sufficiency predictor.

## Internal baselines

| ID | Policy | How it chooses | Overhead |
|---|---|---|---|
| B1 | raw always | representation 0 | none |
| B2 | fixed compression | `r2a_wavelet` always | none |
| B3 | static features | `r3_spectral` always | none |
| B4 | best single | the cheapest representation whose **marginal** disagreement on the calibration bearings passes `(losses + 1) / (n + 1) ≤ α`, costed under the scenario's nominal state; fixed for the whole run | none |
| B5 | context-blind adaptive | the smallest member of Kai's per-window safe set, ignoring bandwidth, queue and load | full |
| B7 | oracle | knows each window's true disagreement; cheapest truly sufficient representation | select |
| B8 | profile table | offline lookup: bandwidth bin (two per decade, 10 kbit/s – 100 Mbit/s) × load level → best single among the marginally admissible, empty queue assumed | select |

## Prior-art adaptations

### B9 — adaptive-threshold data reduction (after Zhang et al. 2023)

- **Idea kept:** stay at the most compressed setting while the signal is steady, raise fidelity
  when it drifts; the threshold follows the signal's own running spread.
- **This implementation:** a ladder `r2b_wavelet → r2a_wavelet → r1_lossless`. The drift signal is
  the log RMS from the probe; an exponentially weighted mean and variance (β = 0.05) give a
  z-score, and the policy steps up one rung per 2 σ.
- **Differences from the paper:** the original reduces data inside one codec family with its own
  drift detector and reconstruction-error targets. Here the ladder is three of this project's
  codecs, the drift statistic is a single probe feature, and there is no reconstruction-error
  constraint. The constants (β, 2 σ) were set once and not tuned.
- **What it cannot do by design:** it has no notion of task sufficiency, so its disagreement with
  the raw decision is whatever its rungs happen to give.

### B10 — two-mode semantic / traditional switch (after Qu et al. 2026)

- **Idea kept:** a system with a "semantic" mode (compact, compute-heavy) and a "traditional" mode
  (bytes, compute-light) that switches on channel quality.
- **This implementation:** semantic = `r6_bottleneck`, traditional = `r1_lossless`. The switch is
  the latency break-even bandwidth of the two modes from PRD §6.6,
  `B* = 8 (S_trad − S_sem) / Δt`, evaluated at the current edge load.
- **Differences from the paper:** the original defines its own modes and switching criterion for
  its application; the break-even rule used here is this project's closed form, which is the most
  favourable switching rule the cost model allows. Two modes only, by construction.

### B11 — conformal model selection (in the style of Kalør & Ohtsuki 2025, arXiv 2506.11391)

- **Idea kept:** certify a set of *configurations* offline with a distribution-free bound, then
  pick among the certified ones at run time according to the system state.
- **This implementation:** a representation is admissible if a binomial test rejects "disagreement
  rate > α" at level δ / (K − 1) on the calibration bearings (Learn-then-Test with a Bonferroni
  correction, δ = 0.1). At run time the same degradation ladder and cost model as Kai choose among
  the admissible representations that meet the deadline.
- **Differences from the paper:** the original selects a feature-extractor + inference-model pair
  under conformal latency *and* accuracy guarantees for black-box models. Here the configurations
  are this project's representations, the loss is disagreement with the raw decision, latency comes
  from the cost model rather than a conformal bound, and the guarantee is marginal per
  representation.
- **The comparison this isolates:** B11 and Kai share the cost model and the ladder; they differ
  only in *marginal* admissibility versus Kai's *per-window* safe set.

### B13 — AdaEdge-style codec bandit (after Liu, Paparrizos & Elmore, ICDE 2024)

Added on 2026-10-14 after the first novelty sweep found AdaEdge to be the closest uncertified
neighbour of this project. The algorithm was taken from the full text (accepted manuscript,
<https://par.nsf.gov/servlets/purl/10613188>); the paper gives no pseudocode and no update
equation, so several details below are choices, marked as such.

- **Idea kept (online mode):** lossless first; when lossless does not fit the link, a
  multi-armed bandit over the lossy options whose reward is the agreement of the model's output on
  the decoded data with its output on raw data — the same quantity Kai certifies. Optimistic
  ε-greedy with ε = 0.01 and a constant step of 0.5, as the paper reports for its online and
  data-shift settings. No accuracy threshold and no guarantee: the paper calls itself best effort.
- **This implementation:** `r1_lossless` if it meets the deadline at the current link state;
  otherwise ε-greedy over the representations that do, by estimated agreement, ties to the cheaper
  one; if nothing fits it sends the cheapest anyway.
- **Differences from the paper:**
  - AdaEdge tunes every lossy codec to the *same* compressed size, so size drops out of its
    reward. Kai's representations have their own fixed sizes; here infeasible arms are masked and
    the bandit maximises agreement among the rest. This is an adaptation, not the paper's method.
  - AdaEdge runs the model on the device to compute the reward. Kai's edge does not hold the cloud
    model, so the reward arrives only for audited windows (2 % of sent windows), and the raw
    copies those audits need are charged to this baseline.
  - Choices the paper leaves open: the optimistic initial value (1.0), the update
    `Q ← Q + step · (reward − Q)`, masking as the way constraints enter, the tie-break.
  - The offline recoding mode (storage budget, LRU cascade) is not implemented.

## Not implemented

| ID | What | Why |
|---|---|---|
| B6 | Kai itself (S1 / S2 × V0 / V1 / V2) | the proposed method, listed in the PRD's baseline table for completeness |
| B12 | RL selector (PPO trained in the simulator) | a "Could" item; not started |

## Reporting rules for these baselines

- B4, B8 and B11 make no per-window sufficiency claim. In the result tables their disagreement is
  reported in the same column as Kai's, but only Kai's `certified` decisions carry the certificate.
- Any table that names B9–B11 says "adaptation" and points here.
