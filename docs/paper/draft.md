# Paper draft — certify-then-optimize for edge sensor offloading

**The manuscript now lives in LaTeX (2026-11-09):** `main-arxiv.tex` (article, with appendix:
full proof and the earlier-code datasets) and `main-ieee.tex` (IEEEtran conference, no appendix)
share `sections/*.tex`, `macros.tex` and `refs.bib`. Build with
`latexmk -pdf main-arxiv.tex` / `latexmk -pdf main-ieee.tex`. Every number carries a `% src:`
comment naming its report. Open items are `\todo{…}` (red in the PDF): authors, Icentia11k
(§7.8), the release URL, and bibliography entries marked VERIFY in `refs.bib`. This file stays
as the outline.

Status: working draft, 2026-11-01 (after the ADR 0020 / 0021 rerun). **No number is typed here.**
Every quantitative statement points to a section of [results.md](../research/results.md)
(Paderborn) or a transfer report — [results-har.md](../research/results-har.md),
[results-speechcmd.md](../research/results-speechcmd.md),
[results-speechcmd-logmel.md](../research/results-speechcmd-logmel.md) (exploratory),
[results-sleepedf.md](../research/results-sleepedf.md); HUST, UORED and PTB-XL are from earlier
code — which `make report` regenerates. When a
figure or table goes into the manuscript, it is copied from there at submission time. Claims are
worded to match [novelty-log.md](../research/novelty-log.md) and PRD §3.3 as revised on 2026-10-14.

## Working title

*Certified offloading: a sufficiency certificate for what an edge sensor sends, and when choosing
per window is worth it*

## Abstract (shape, to be filled from the report)

An edge node that offloads vibration windows must decide what to send: raw samples, a compressed
waveform, engineered features or a learned latent. We certify that choice. Conformal risk control
on the event that the cloud's decision would change yields a *safe set* of representations whose
expected rate of certified decisions that disagree with the raw-data decision, per offered window,
is at most α for **any** selector inside it when calibration and test units are exchangeable, and
two such sets — one fixed, one per window, each at α/2 — combine by a union bound into a *cascade*
that pays for per-window scoring only when it can pay for itself (both added after the first
results, so exploratory). Evaluated strictly on units never seen in training — bearings, people,
speakers and sleepers — the cascade's and the fixed set's mean disagreement stays below α on every
dataset rerun with the final code, at a small fraction of the raw payload (results §E2 of each
report). The simulator used for the cost study meets its pre-set latency threshold against a real
socket runtime on three of four datasets and in a Docker testbed (§E7, §E7d). Adaptive choice
fails its pre-registered value test on every dataset (H3); on three of four it costs more than the
best single representation, which carries no guarantee, because when the certificate refuses the cheap latent the
next certified payload is too large for a slow link. A mixed menu beats every single family on
Sleep-EDF, on PTB-XL and, exploratory, with log-mel on speech; on these datasets the menu holds a
small representation close to the raw decision. We report the negative findings as such and
release the pipeline.

## 1 Introduction — the argument in five steps

1. Representation choice for offloading is made by heuristics or learned policies with no
   statement about what it does to the downstream decision; AdaEdge (ICDE 2024) already selects
   among codecs by agreement with the raw-data model, best effort.
2. Contribution: the *certificate* — a safe set with a finite-sample bound, in expectation, on agreement with the
   raw decision that holds for every selector inside it (Proposition 1), so the optimizer can be
   swapped without re-proving anything; and the *cascade*, which spends edge compute on
   per-window scoring only when the cost model says it could pay.
3. An honest evaluation needs a reference model that generalises to unseen machines. We searched
   for one under a strict leave-bearings-out protocol and record every candidate; scores near 0.95
   reported on this dataset come from leaky splits.
4. Findings: on the four datasets rerun with the final code the cascade's and the fixed set's
   mean risk is below α at every α (on UORED, earlier code, the cascade's mean was above α); the
   per-window certificate lands on α on Speech Commands, stays below it on Paderborn, and is pushed
   over α by unit shift on HUST, UORED and HAR at α = 0.1; the simulator meets its latency threshold
   on three of four datasets; adaptive choice fails H3 everywhere, is cheaper than best-single only
   on HAR, and paid under CPU pricing only while the edge was slow.
   These are the regimes a practitioner needs to know.
5. Contributions: (C1, narrowed) the certificate, the any-selector statement and the cascade;
   (C4) a leakage-safe benchmark with a recorded model search; (C5) an empirical map of when
   adaptivity pays, with negative results reported as such.

## 2 Related work

Use PRD §3.3 (items 1–13) and `lit-matrix.csv`. Three groups: conformal guarantees at the edge for
one payload family (rows 18–23); cascade and routing with risk control (rows 25–28: "bound one
quantity, optimise the other"); codec / representation selection without guarantees (AdaEdge,
VergeDB, ZipMate, row 13–15). The differentiation is the per-window certificate over a
heterogeneous menu and its independence of the selector; not "heterogeneous menu" and not
"agreement with the raw model", both of which AdaEdge has.

## 3 Problem and definitions

Glossary terms as in `docs/glossary.md`. Decision-agreement loss, safe set, safe-set loss, flags.
State plainly that the bound is on `E[d_π · 1{certified}]` per offered window.

## 4 Method

- 4.1 Safe sets and Proposition 1 (corrected statement and proof, `docs/theory/proposition1.md`).
- 4.2 Variants: V0, V1 (LTT), V2 (policy-specific; nominal cost vector only), **V0 with constant
  scores** (one safe set for every window; ADR 0008), and the **cascade** (union of the two at α/2
  with a value-of-information gate on the probe; ADR 0011) — the deployment we recommend.
- 4.2b Group CRC (exploratory, ADR 0020): one exchangeable sample per unit; needs at least
  ⌈1/α⌉ − 1 calibration units, which at α = 0.05 only Speech Commands has (Sleep-EDF has enough
  at α = 0.1); it bounds the expected mean loss of a new exchangeable unit.
- 4.3 Selection inside the set: greedy, MPC; the degradation ladder.
- 4.4 Drift: audit-in-the-loop ACI (no finite-sample guarantee); the probe-feature monitor and why
  it fails here (ADR 0008 §5).

## 5 System

Select-then-encode; wire frame with contract hash; immutable bundle; numpy/ONNX-only edge; the
discrete-event simulator, with timings measured at each node's duty cycle and a calibrated runtime
overhead (ADR 0008); real-socket and Docker backends.

## 6 Experimental protocol

- Leave-bearings-out folds with four bearing-disjoint partitions; three repeats × five folds;
  leave-condition-out folds for drift (results header, `docs/research/hypotheses.md`).
- The reference model: EnvNet, chosen by inner CV on training bearings only (ADR 0006); why scores
  near 0.95 in the literature come from leaky splits (ADR 0006, "Why not 0.95").
- The supervised bottleneck built on the reference model's front-end, and what that implies
  (ADR 0006 addendum) — this must be in the main text, not a footnote.
- Pre-registered hypotheses H1–H7 and the verdict rule (threshold met **and** Holm).

## 7 Results (each subsection = one section of results.md)

- 7.1 The menu (results §E1): sizes, timings at three duty cycles, agreement of every
  representation with the reference model on unseen bearings.
- 7.2 The certificate (results §E2): risk per α across fifteen folds with bearing-block CIs, the
  exchangeability control, V0 versus the constant-score variant versus V1 and V2.
- 7.3 The system (results §E3, price presets, §E5): Kai versus baselines B1–B13 and the oracle;
  breakdown by sensors per node and by bandwidth; what the ablations say about the predictor.
- 7.4 Shift (results §E6): leave-condition-out; the audit loop; the drift monitor.
- 7.5 Theory and reality (results §E4, §E7, §E7d): sim-to-real meets its threshold (H7); the
  closed-form break-evens fail (H6), and under a strict noise rule no measured crossing is resolved
  at all (ADR 0022, exploratory); a median-SE rule or a finer grid was not tried.
- 7.6 Transfer (results-har.md, results-speechcmd.md, results-sleepedf.md; HUST, UORED and PTB-XL
  from earlier code): the same pipeline and hypotheses on four more datasets; log-mel on Speech
  Commands as an exploratory menu (ADR 0018).
- 7.7 Privacy (results §E10): compact payloads re-identify machines.

## 8 Limitations

Copy from `docs/theory/limitations.md`; add: laptop timings and the 32-sensor sensitivity; the
measured traces are downlink shape; the reference model's absolute quality; fold-level p-values
are optimistic because folds of different repeats share test units, and Holm is applied within a
dataset, not across datasets; group CRC needs more units than Paderborn and HAR have at any α, and Sleep-EDF at α = 0.05; the
Sleep-EDF real-socket test replays 60 windows per link rate (ADR 0021); the edge is a laptop, not a
Raspberry Pi, so H4 is untested where it matters.

## 9 Conclusion (shape)

A certificate is cheap to add and decouples guarantees from optimisation. The union-bound cascade
and the static set stay below α on average on every rerun dataset; window-level V0 lands near α and
misses H2a where the units are not exchangeable. Whether *per-window* certification pays depends on the menu: with one compact
representation that follows the reference model, a certified static set is the right deployment;
with a menu of genuinely competing representations the per-window predictor has something to do.
Building such a menu for machines the model has not seen is the open problem.

## Open items before submission

- Every hypothesis verdict re-read against the final report.
- The decision on whether the edge may run the reference model's front-end (worksplit.md) — it
  changes the framing of 7.1–7.3.
- A human check of Proposition 1.
- The novelty sweep repeated at T−14 and T−2 days.
- SHHS (access to be requested on 2026-11-02): enough units for group CRC, and the first unseen test of the
  exploratory fixes agreed on 2026-11-01 (lossless fallback, static fallback, noise-aware H6
  crossings, H2a non-inferiority; results in `updates.md` §7).
