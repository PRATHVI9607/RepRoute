# ADR 0006 — Reference model `f_raw` becomes an envelope-spectrum classifier (EnvNet)

Date: 2026-10-14 · Status: accepted · Touches: `tasks/models.py`, `tasks/train.py`,
`tasks/pipeline.py`, `cli.py`, `configs/dataset/`, `scripts/model_search.py`

## Why

The reference model of ADR 0004 (SpecNet: log band-energy *shape* of the spectrum + logistic
regression) reached a mean test macro-F1 of 0.47 on unseen bearings, 0.29 in the worst fold. The
certificate controls agreement with `f_raw`, not its correctness, so a weak `f_raw` caps what the
whole system can claim. The project owner asked for the highest score that can be reached honestly.

## Protocol (fixed before any candidate was scored)

- Candidates are compared by **macro-F1 on held-out training bearings**: inside the `train`
  partition of each of the five L1 folds, bearings are split into inner folds that each hold out at
  least one bearing per class (two shuffles, so about six inner folds per outer fold).
- `ptrain`, `calib` and `test` are never read by the search.
- Selection rule: the highest mean over the five outer folds among candidates that fit the
  deployed system (window of 4096 samples, no side information). Candidates within 0.01 of the best
  are treated as tied — the inner-fold standard deviation is about 0.15 — and the tie goes to the
  one with the fewest inputs.
- Every candidate is appended to `experiments/model_search/results.jsonl`, losers included, and
  rendered to [docs/research/model-search.md](../research/model-search.md) by
  `scripts/model_search.py report`.

## What the search found

Scores are inner-CV macro-F1 (mean over outer folds / worst outer fold). The complete list is in
the generated report; the rows below are the ones the decision rests on.

| Candidate | Inputs | Mean | Worst fold |
|---|---|---|---|
| `band24` — SpecNet's front-end (the ADR 0004 model) | 22 | 0.43 | 0.34 |
| `mod6` — envelope spectrum per carrier band, relative to the mean envelope | 234 | 0.49 | 0.39 |
| `order6` — the same in the order domain (needs the shaft speed) | 288 | 0.50 | 0.40 |
| `kin6_fft256` — envelope level at the 6203's fault orders over the median (needs the speed) | 78 | 0.55 | 0.44 |
| `kinb_fft256` — the same evaluated at both rig speeds (speed-blind) | 156 | 0.56 | 0.47 |
| **`modfloor16` — envelope spectrum at 16 log-spaced modulation frequencies over its own median** | **96** | **0.575** | **0.47** |
| `modfloor16_order` — the same grid scaled by the true shaft speed | 96 | 0.57 | 0.46 |
| `band24+mod6`, `band24+kin6` — adding the stationary spectral shape back | 100–256 | 0.45–0.48 | 0.32–0.35 |
| one hidden layer (16 or 64 units) instead of a linear head | — | no gain | — |
| soft group-DRO over training bearings | — | −0.01 to −0.02 | — |
| `envcnn_w32` — 1-D CNN over the same envelope map, with / without a bearing-adversarial head | 6.6 k params | 0.49 / 0.48 | 0.36 / 0.32 |
| `modcnn_w8` / `modcnn_w16` — 2-D CNN on the log-STFT with each bin's time mean removed (with / without the adversarial head) | 6.4 k / 24 k params | 0.40–0.41 | 0.21–0.23 |

Four findings are consistent across the table.

1. **The stationary spectrum identifies the bearing, not the fault.** Every candidate that contains
   the band-energy shape loses on unseen bearings and wins on seen ones (finding 4).
2. **Normalising the envelope spectrum by its own median is what helps** (0.49 → 0.57). A
   peak-to-floor ratio is free of the carrier band's gain; a ratio to the mean envelope is not.
3. **Capacity hurts.** Linear heads with strong shrinkage beat hidden layers and CNNs trained on the
   same inputs; with 11–14 training bearings, anything flexible fits bearing identity.
4. **Knowing the shaft speed does not help at this window length.** The order-domain grid scores the
   same as the fixed Hz grid, so the wire frame needs no speed field. The 900 rpm condition is the
   hard one either way (0.44–0.46 against about 0.60 at 1500 rpm): a 64 ms window holds less than
   one shaft revolution at 900 rpm.

### Why not 0.95

`experiments/model_search/diagnostics_n4096.json` holds the numbers.

| Diagnostic (inner CV on training bearings) | `band24+mod6` | `modfloor16` |
|---|---|---|
| Held-out bearings (the honest protocol) | 0.48 | 0.575 |
| **Same bearings in fit and validation** (what a random-window split measures; diagnostic only) | **0.95** | 0.77 |
| Bearing-level majority vote over all windows of a held-out bearing | 45 % correct | 74 % correct |
| Binary healthy vs damaged, held-out bearings | 0.72 | 0.73 |

Scores around 0.95 on this dataset come from evaluating on windows of bearings that were also in
training; the same features drop to 0.48 when the bearing is new. The remaining error is not
window noise: about a quarter of the held-out bearings are classified wrongly *as a whole* (with
the order-domain front-end, for example, healthy `K002` is called an inner-race fault in 71 % of
its windows and `KI05` is called healthy in 56 %), and no per-window
model can fix that from 11–14 training bearings. Random-window evaluation is banned in this
project (`scripts/check_no_l0_splits.py`), so 0.95 is not a number this repository can report.

### What others report on unseen bearings

Collected by a literature subagent on 2026-10-14; "read" says how the number was obtained.

| Source | Protocol | Reported | Read |
|---|---|---|---|
| Lessmeier et al., PHME 2016, [Table 9](https://papers.phmsociety.org/index.php/phme/article/download/1577/542) | train on artificial damage, test on real damage, one operating condition, 3 classes, vibration | accuracy 62.3–65.9 % per classifier, 75.0 % for the ensemble | PDF |
| Knap, Jachymczyk, Lalik, [PHME 2026](https://papers.phmsociety.org/index.php/phme/article/download/4924/2931) | held-out bearings (K001, KA30, KI21, KB23), 4 classes, vibration | macro-F1 0.355 (logistic regression), 0.326 ± 0.008 (1-D CNN); artificial → real 0.634 / 0.637 | PDF |
| Vieira et al. 2025, [arXiv 2509.22267](https://arxiv.org/html/2509.22267v3) | bearing-wise against leaky splits, WDCNN | macro AUROC bearing-wise 54 % (time) / 60 % (frequency) / 74 % (envelope); segment-level leak 99.65–100 % | summary of the HTML, not checked line by line |
| Perminov & Korzun, [FRUCT-29](https://old.fruct.org/publications/acm29/files/Per.pdf) | real-damage bearings, one set of three held out, 2-D CNN | validation 99.7 % on seen bearings, test 74 % on unseen ones (9 % for another network width) | PDF |

These agree with the search on all three points: honest scores on this dataset sit between 0.3 and
0.75, envelope-domain inputs transfer best, and the same models read 99–100 % when windows of the
test bearings leak into training. The bearing geometry used by the `kin*` front-ends (8 rolling
elements, 6.75 mm, pitch circle 28.55 mm, 0° pressure angle, hence BPFO ≈ 3.05 × and BPFI ≈ 4.95 ×
shaft frequency) is Table 1 of the Lessmeier paper.

### Window length

Joining consecutive windows (study only; the system still uses N = 4096):

| Front-end | N = 4096 | N = 16 384 | N = 65 536 |
|---|---|---|---|
| `band24` | 0.43 | 0.43 | 0.43 |
| `order6` (needs speed) | 0.50 | 0.59 | 0.56 |
| `kin6h` (needs speed) | 0.52 | 0.64 | — |
| `kinb_fft256` (speed-blind) | 0.56 | 0.635 | — |
| `modfloor16` | 0.575 | 0.57 | — |

A 256 ms window lifts the fault-order features to about 0.64; the geometry-free front-end chosen
here does not benefit. Changing N changes the raw payload, the window period and every scenario, so
it is left as a decision for the group (see `worksplit.md`), with these numbers as the evidence.

## Decision

`f_raw` for the Paderborn data = **EnvNet**: `EnvFront` (`modfloor16`: STFT of length N/16 and hop
N/128, squared envelope in six octave bands, magnitude of its Fourier transform at 16 log-spaced
modulation frequencies from 1.28 to 40.96 cycles per window, log ratio to the median over those
frequencies; 96 inputs, no parameters) → standardisation → multinomial logistic regression. The L2
penalty is the only hyper-parameter and is chosen per fold from {0.1, 0.3, 1.0} (mean-loss scale)
by grouped inner CV on the training bearings, exactly as SpecNet chose its C. The production
solver (scikit-learn, `kai.tasks.train.train_envnet`) was checked against the search on the same
inner folds: 0.573 / 0.575 / 0.572 for the three penalties.

`modfloor16`, `modfloor24` and their penalties are tied under the rule (0.570–0.575); the tie went
to the 96-input front-end. It uses no bearing geometry and no shaft speed.

The dataset config selects the model (`configs/dataset/pu.yaml: f_raw: env`). The synthetic fixture
keeps SpecNet (`f_raw: spec`) so that the stored mini-paper references do not move; SpecNet and the
waveform CNN stay available as `kai folds --f-raw spec|wave`.

**Gate G1 is unchanged**: mean test macro-F1 ≥ 0.40 (ADR 0004).

## Disclosure

- The decision was frozen in this file before `f_raw` = EnvNet was evaluated on any `ptrain`,
  `calib` or `test` partition.
- It was made after all E1–E8 results of the SpecNet folds had been seen. No hypothesis, metric or
  threshold was changed; every experiment was re-run with the new folds and the earlier runs stay
  in the registry.
- The candidate list was designed by looking at inner-CV scores on the training partitions of all
  five folds. Each fold's test bearings are training bearings of other folds, so the *design* of
  the front-end has seen every bearing's inner-CV behaviour. The per-fold fit and the penalty
  choice are properly nested; the choice of front-end is not, and the test scores below should be
  read with that in mind.

## Further attempts after the decision (inner CV, same protocol; none adopted)

| Attempt | Mean | Worst fold | Against |
|---|---|---|---|
| Two stages: healthy vs damaged, then inner vs outer | 0.564 | 0.455 | 0.575 |
| Average of 15 linear heads fitted on bootstrap samples of the training bearings | 0.574 | 0.467 | 0.575 |
| Multi-label: two detectors (inner damaged? outer damaged?) | 0.550 | 0.449 | 0.575 |
| … with the three combined-damage bearings KB23 / KB24 / KB27 added to the fitting side | 0.549 | 0.460 | 0.550 |

The multi-label formulation that the PRD mentions and the three extra bearings change nothing, and
neither does variance reduction over bearings: the plateau near 0.575 at this window length is not
a matter of the head.

## Addendum (same day): the supervised bottleneck R6 is built on the reference model's front-end

**What happened.** The first three folds trained with EnvNet showed that no compact representation
could follow it. Disagreement with `f_raw` on the test bearings:

| Fold | R2a wavelet | R2b wavelet | R3 spectral | R4 summary | R5 autoencoder | R6 bottleneck (generic CNN) | R7 VQ |
|---|---|---|---|---|---|---|---|
| r0f0 | 0.007 | 0.031 | 0.49 | 0.54 | 0.60 | 0.58 | 0.54 |
| r0f1 | 0.005 | 0.027 | 0.38 | 0.52 | 0.70 | 0.62 | 0.61 |
| r0f2 | 0.002 | 0.016 | 0.34 | 0.37 | 0.49 | 0.37 | 0.37 |

Three-class chance is 0.67. With such a menu the safe set is `{raw, lossless, wavelets}` and the
smallest certifiable payload is a wavelet frame of 2–4 kB.

**How the fix was chosen** (stage D of the search; inner CV on training bearings, the first inner
fold of each outer fold; disagreement with an EnvNet fitted on the inner-training bearings, measured
through int8 quantisation):

| R6 encoder, 16 bytes | Mean | Worst fold |
|---|---|---|
| generic CNN, hard targets (the recipe of ADR 0004) | 0.49 | 0.57 |
| generic CNN, soft targets | 0.46 | 0.56 |
| generic CNN, soft targets, augmentation with teacher re-labelling | 0.47 | 0.73 |
| **EnvNet's front-end → learned 96 → 16 projection, soft targets** | **0.024** | **0.032** |

A CNN trained on 11–14 bearings does not learn to imitate an envelope-spectrum model on a new
bearing, for the same reason it does not learn the labels; relabelled augmentations do not change
that.

**Decision.** For the Paderborn data, R6 = the reference model's fixed front-end followed by a
learned 16-value projection and `tanh`, quantised to int8 (`kai.tasks.train.train_front_bottleneck`,
`configs/dataset/pu.yaml: bottleneck: teacher_front`). The projection is trained on `train` windows
so that a small head on the noisy latent reproduces `f_raw`'s class probabilities. This is
supervised compression as PRD §7 cites it — a bottleneck inserted after the teacher's early layers
and distilled from the teacher — and it keeps the PRD's payload (16 bytes, int8, ONNX). The generic
CNN stays available as `kai folds --bottleneck generic`; the three folds trained with it are kept in
`artifacts/folds_generic_r6/`.

**What this costs, stated plainly.**

- The edge now runs the reference model's front-end. R6 is close to running the classifier on the
  edge and sending its answer: the PRD's premise that `f_raw` lives only in the cloud holds for the
  final linear layer, not for the feature extraction. A reader should ask "why not classify on the
  edge?", and the answer here is only "the cloud keeps the decision rule and can change it without
  touching the edge's 96 → 16 projection being retrained" — a weak answer.
- With one 16-byte representation near 2 % disagreement, selection between representations is
  expected to matter little: Kai and "best single representation" will mostly coincide. H1 and H3
  are evaluated as registered and are expected to fail for this reason.
- R3, R4, R5 and R7 are unchanged and stay near chance agreement with EnvNet. They are in the menu
  as the PRD defines them; the experiments show what they cost.
- R6 was changed after its test-fold disagreement had been seen on three folds. The choice between
  encoders was made on the inner-CV table above, where the gap is twenty-fold.

## Consequences

- `f_raw` reads modulation in every octave band, including weak ones. Lossy waveform
  representations change its decision more often than they changed SpecNet's (R2b: 1.6–3.1 % on
  the first three folds, against 0.2–1.0 % with SpecNet).
- Gate G1 and every hypothesis keep their registered definitions and thresholds.

## Test results (read once, after the decision above)

Macro-F1 of `f_raw` against ground truth on the test bearings of the five L1 folds
(`experiments/model_search/results.jsonl`, stage `T-test`, from each fold's `metrics.json`):

| Reference model | r0f0 | r0f1 | r0f2 | r0f3 | r0f4 | Mean | Worst |
|---|---|---|---|---|---|---|---|
| SpecNet (ADR 0004) | 0.290 | 0.471 | 0.579 | 0.588 | 0.433 | 0.472 | 0.290 |
| **EnvNet (this ADR)** | **0.545** | **0.641** | **0.688** | **0.718** | **0.449** | **0.608** | **0.449** |

EnvNet is better in every fold, and the test mean is above the inner-CV estimate (0.575), as
expected with more training bearings than an inner fold has. Gate G1 (≥ 0.40) is met. The majority
baseline is 0.18 and the published scores on unseen bearings of this dataset are 0.33–0.75 (table
above); 0.61 is an honest number, and it still means that roughly four windows in ten are
misdiagnosed on a bearing the model has not seen.

Disagreement with EnvNet on the same test bearings, mean of five folds: R2a 0.5 %, R2b 2.3 %,
R6 (teacher front-end, 16 bytes) 0.7 %, R3 36 %, R4 44 %, R7 45 %, R5 52 %.
