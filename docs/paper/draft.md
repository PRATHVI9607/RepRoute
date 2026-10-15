# Paper draft — certify-then-optimize for edge sensor offloading

Status: working draft, 2026-10-14. **No number is typed here.** Every quantitative statement points
to a section of [results.md](../research/results.md) (Paderborn) or
[results-hust.md](../research/results-hust.md) (transfer), which `make report` regenerates. When a
figure or table goes into the manuscript, it is copied from there at submission time. Claims are
worded to match [novelty-log.md](../research/novelty-log.md) and PRD §3.3 as revised on 2026-10-14.

## Working title

*Certified offloading: a sufficiency certificate for what an edge sensor sends, and when choosing
per window is worth it*

## Abstract (shape, to be filled from the report)

An edge node that offloads vibration windows must decide what to send: raw samples, a compressed
waveform, engineered features or a learned latent. We certify that choice. Conformal risk control
on the event that the cloud's decision would change yields a *safe set* of representations whose
expected disagreement with the raw-data decision is at most α for **any** selector inside it, and
two such sets — one fixed, one per window, each at α/2 — combine by a union bound into a *cascade*
that pays for per-window scoring only when it can pay for itself. Evaluated strictly on bearings
never seen in training, across two datasets, the fixed certified set holds its bound in every fold
at a small fraction of the raw payload (results §E2). Whether per-window choice pays depends on
what is priced: under byte or energy pricing one compact representation suffices and the fixed set
wins; when edge CPU time is priced, per-window choice is markedly cheaper than any single
representation (results §E3, price presets). We report the negative findings as such and release
the pipeline.

## 1 Introduction — the argument in five steps

1. Representation choice for offloading is made by heuristics or learned policies with no
   statement about what it does to the downstream decision; AdaEdge (ICDE 2024) already selects
   among codecs by agreement with the raw-data model, best effort.
2. Contribution: the *certificate* — a safe set with a finite-sample bound on agreement with the
   raw decision that holds for every selector inside it (Proposition 1), so the optimizer can be
   swapped without re-proving anything; and the *cascade*, which spends edge compute on
   per-window scoring only when the cost model says it could pay.
3. An honest evaluation needs a reference model that generalises to unseen machines. We searched
   for one under a strict leave-bearings-out protocol and record every candidate; scores near 0.95
   reported on this dataset come from leaky splits.
4. Findings: the fixed certified set holds on unseen bearings on both datasets; the per-window
   certificate holds on average on one and fails under the second dataset's bearing shift; per-window
   choice pays only when edge CPU is the scarce resource. These are the regimes a practitioner
   needs to know.
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
  with a value-of-information gate on the probe; ADR 0011).
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
- 7.5 Theory and reality (results §E4, §E7, §E7d): break-evens and sim-to-real.
- 7.6 Transfer (results-hust.md): the same pipeline on a second dataset.
- 7.7 Privacy (results §E10): compact payloads re-identify machines.

## 8 Limitations

Copy from `docs/theory/limitations.md`; add: laptop timings and the 32-sensor sensitivity; the
measured traces are downlink shape; the reference model's absolute quality.

## 9 Conclusion (shape)

A certificate is cheap to add, valid in the sense it promises, and decouples guarantees from
optimisation. Whether *per-window* certification pays depends on the menu: with one compact
representation that follows the reference model, a certified static set is the right deployment;
with a menu of genuinely competing representations the per-window predictor has something to do.
Building such a menu for machines the model has not seen is the open problem.

## Open items before submission

- Every hypothesis verdict re-read against the final report.
- The decision on whether the edge may run the reference model's front-end (worksplit.md) — it
  changes the framing of 7.1–7.3.
- A human check of Proposition 1.
- The novelty sweep repeated at T−14 and T−2 days.
