# ADR 0028 — Audit fixes, slow-edge crossover, memory-tight Kai, bigger f_raw, HMC

Date: 2026-11-10. Status: accepted 2026-11-10 (decisions in §6).
Everything in A–D is **exploratory** (decided after the ADR 0025–0027 results were seen). The
pre-registered H1–H7 verdicts of record and the failed H3 stay as they are; corrected
re-measurements are reported next to them, never in their place.

Inputs: nine read-only reviews of 2026-11-10 (simulator/policies, harness/report, certificate/
training/splits, loaders/codecs, profiler/runtime/traces, theory review of the slow-edge model,
Paderborn diagnosis, pretrained-f_raw feasibility, HMC + wav2vec2 feasibility).

## 1. Fixes (bug = wrong number today; risk = can produce one)

Simulator and policies (`link/des.py`, `select/`, `bench/evaluate.py`)
1. `CostModel.sendable`: the bandwidth test can never fail (estimator ≥ 1 b/s, frozen during
   outages), so `certified_late` never defers on a dead link. Test sendability on the estimated
   delivery time of the frame (≤ cap·D) instead. *Affects the ADR 0025 late variants.*
2. Raw deferral charges no I/O or raw encode, an r1 deferral charges both: charge `edge_ms[0]`.
   *Affects the registered `kai_s1_v0` (H3 re-measurement).*
3. Bytes still queued at the end of a run are never charged: charge `backlog_bytes` at the end.
4. Committee edge time omits the decode `t_dec[c]`; `required_bytes` omits the non-member reps the
   committee/latent policies can send (and r1 for deferral), and `predictor_z`/r6 for latent ones.
5. ACI feedback: route the free raw "agree" through the same delivery path as non-raw audits.
6. S2 (MPC) is charged greedy's measured `t_sel`: measure it or label it.

Harness and report (`bench/experiments.py`, `scripts/report.py`)
7. One RNG feeds all bootstrap CIs in insertion order, so exploratory rows shift pre-registered
   CIs (H2a CI moved 0.1105 → 0.1125): one Generator per (block, α, variant).
8. `SLOW_POLICIES` lacks `b1_raw` (fresh e3s cannot reproduce the published B1 keys).
9. A policy missing on some folds makes `x26_summary` raise before the H1/H3 summary is written:
   exploratory reductions only over folds where every policy has rows, else a "missing" key.
10. Provenance: add a simulator code hash and per-fold `table.npz` hashes; validate `add_policies`
    names; keep `required_bytes` when the column is absent.
11. H6 "verdict of record = first E4 run": take it from the first registry row, as the text says.
12. Tests for `fill_required_bytes`, `x26_summary`, `e3s`, `budget_sweep`.

Runtime, profiler (`edge/runtime.py`, `profile/profiler.py`)
13. With shaping, the runtime serialises edge compute and sending (the DES overlaps them): a
    sender task with a queue. *Affects E7/H7: re-measure.*
14. Profiler pins only the calling thread: pin all threads (or start under `taskset`); choose a
    P-core explicitly; record core, frequency cap, turbo, `cpu.max`, throttle counts.

Training and data (`tasks/`, `data/`) — fixed in code; see §6 decision D2 for existing folds
15. r5 int8 scale from the first 4096 train windows in loader order → from all train windows.
16. Codec cache key hashes every 97th window → hash all of `ws.x` (or the fold fingerprint).
17. GPU determinism: set `CUBLAS_WORKSPACE_CONFIG`, `warn_only=False`, device in `recipe()`.
18. Sleep-EDF resamples each epoch separately → resample the whole night, then cut.
19. `quantise` rejects non-finite input; labels ≥ 0 asserted; `frame.pack` range checks.

Known limitations stated, not fixed for existing datasets (§6 D2–D3): f_raw selected on one fold's
train units (not nested per outer fold), so f_raw's test F1 is unbiased only for r0f0; HAR uses
acceleration magnitude (static postures barely separable); PTB-XL label noise from likelihood-100.

## 2. Slow-edge methodology (replaces ADR 0027's comparison)

- **Feasibility.** For each cell and policy, edge utilisation ρ = edge time per window / window
  period. Cells with ρ ≥ 1 for a policy are *infeasible* for it and get no cost. Two policies are
  compared only over cells feasible for both; the share of infeasible cells is reported.
- **Cost terms.** Cost is reported split into bytes, energy, CPU and lateness, per policy and k.
- **Cap sensitivity.** Lateness cap factor 5, 10 (default), 20 reported.
- **Energy.** Default: energy per operation unchanged (`p_cpu_w ÷ k`); sensitivity e ∈ {0.3, 1, 3}.
- **CIs.** Every per-fold value is reported; a CI narrower than 1e-3 is labelled *structural*.

## 3. The four exploratory additions

**C. Crossover k\*.** k ∈ {1, 2, 3, 5, 10, 20, 30, 100}, policies `kai_s1_v0`, `b14_edge_inference`,
`b4_best_single`, `b1_raw`, `kai_s1_v0_classical` (A), first 5 folds, all six datasets.
Δ_f(k) = 1 − C_Kai,f / C_b14,f over cells feasible for both, same folds at every k. k\*_f = the
smallest grid k with Δ_f(k′) > 0 for every k′ ≥ k, interpolated linearly in log k; "> 100" if none,
"≤ 1" if Δ_f(1) > 0. Every sign change is reported. Uncertainty: fold bootstrap of the crossing of
the mean curve, plus the per-fold range. Descriptive; no test.

**A. Memory-tight Kai.** `kai_s1_v0_classical`: the V0 certificate with the menu restricted to the
classical codecs r0–r4 (no model file); same ĝ, λ̂ recalibrated on that menu (calib only). Required
memory = predictor + probe + peak runtime RSS measured once per dataset (not file bytes alone).
Budget sweep over required memory, including f_raw (b14). Fixed menu, no tuning.

**D. Measured slow edge.** E1 profiled on one P-core pinned with all threads, `scaling_max_freq` =
400 MHz, turbo off (the user runs the two sudo commands), cloud process on other cores at full
speed, nothing else running. Written to a separate profile (`KAI_PROFILE=slow400`); edge-run times
(including b14's f_raw and committee heads) come from it, cloud times from `local.json`. The
measured per-operation slowdown distribution is reported, never assumed. E3 on the C policies with
that profile. The CPU-quota variant is dropped (systemd stretches a 1 ms period to 10 ms at 10 %,
which models a shared edge, not a slow one).

**B. Bigger f_raw on Speech Commands.** wav2vec2-base self-supervised weights (LibriSpeech 960 h,
no Speech Commands fine-tune), fine-tuned per fold on that fold's `train` speakers only; ONNX
export with parity check. **Acceptance:** kept only if its macro-F1 on held-out speakers within
`ptrain` (grouped CV, never `calib` or `test`) beats the current f_raw with a paired lower CI bound
> 0; both candidates recorded. If accepted, every d_r, head, ĝ and λ̂ is refit and the certificate
becomes agreement with the new f_raw; b14 is reported with both f_raws. Stack: torchaudio
`WAV2VEC2_BASE` if it loads under torch 2.14 (tested first in a throwaway environment), else
`transformers` (needs approval). Speech Commands only.

**E. Paderborn.** Diagnosis recorded (ĝ AUC 0.52–0.56; r6 alone has d = 0.007 at 16 B; 5 calib
bearings). No Paderborn result is used to choose a menu, cap, price, k or variant.

## 4. HMC confirmatory test (replaces waiting for SHHS)

HMC sleep staging v1.1 (PhysioNet, CC BY 4.0, 151 subjects, never touched in this repo), `EEG
C4-M1` 256 Hz → 102.4 Hz (`resample_poly(x, 2, 5)`, whole night), stages W/N1/N2/N3/R, lights
markers skipped. Its pre-registration (revised S-H3′ on feasible cells, S-H2a, S-H5) is written
**after** the runs above and **before** HMC is downloaded; the 2026-11-10 SHHS entry is superseded
in the log and kept as a second confirmation if SHHS access arrives. f_raw for HMC is selected
nested (inner CV within each outer fold's train units).

## 5. Order (one run, nothing in between)

1. Fixes 1–19 + tests; full nox. 2. Smoke: Sleep-EDF, mini grid, 1 fold, every new path.
3. Main run (unattended): E2/E3/E5 + presets + E4/E7 on all six datasets on the fixed code;
   A and C. 4. D alone (turbo off, nothing else running). 5. B on the GPU, then E2/E3 for
   Speech Commands (both f_raws). 6. Reports, analysis, HMC pre-registration, HMC run.
Estimated compute: ~1–1.5 days for 3–5; HMC ~8 h.

## 6. Decisions

Decided 2026-11-10, before any code or run:
- **D1** At 400 MHz, energy per operation is unchanged: `p_cpu_w` = 1 W ÷ s, where s is the median
  over profiled operations of (time at 400 MHz) / (time in `local.json`).
- **D2** Existing folds are rebuilt only if r5's int8 latents clip (|q| = 127) on more than 1 % of
  `test` windows of some dataset, measured on the stored folds before the main run. Otherwise fixes
  15–18 apply to new builds (HMC, B) and are stated as limitations for the existing folds.
- **D3** f_raw selection stays as it is for the existing datasets and is documented; HMC and B use
  nested selection.
- **D4** Accepted: fixes and tests, nox, smoke run, report, then the single full run.

## 7. D2 outcome and additions (2026-11-10, before any run)

**D2 measured** on the stored folds (share of `test` windows with any r5 latent at |q| = 127, mean /
max over folds): Paderborn 28.3 % / 100 %, Sleep-EDF 14.1 % / 22.2 %, Icentia11k 6.6 % / 8.2 %, HAR
3.7 % / 4.9 %, Speech Commands (both) 0.7 % / 1.2 %. The rule fixed in §6 is met, so **every existing
fold is rebuilt** with fixes 15–17 (r5 scale from a seeded sample of the fold's train windows; full
cache key; deterministic CUDA). As the folds are rebuilt anyway, Sleep-EDF also gets fix 18
(whole-night resampling). E1 is re-profiled, since the menus change. f_raw keeps the architecture
chosen before (D3) and is retrained per fold. The r5 scale stays the 0.999 quantile of the latent
magnitude by design; only its sample changes.

**B, cloud timing.** The cloud runs the wav2vec2 f_raw on the GPU: its cloud-side head time is
measured on the RTX 2050 with batch 1 (warm-up, median + MAD); its edge-side time (b14, D) on the
CPU like every other edge time.

**D, I/O calibration.** With `KAI_PROFILE` naming a slow-edge run, E1's I/O calibration builds its
bundle with the edge times it has just measured; the cloud process runs on other cores
(`KAI_CLOUD_CPUS`).

**Nested f_raw selection, made precise (before any run).** A choice that averages over outer folds
uses units that are test units of other folds, so every f_raw choice is made **per outer fold**, from
that fold's own units only: each candidate is trained on the fold's `train` units and scored by
macro-F1 on the fold's `ptrain` units (never `calib` or `test`). B: wav2vec2 replaces the current
f_raw in a fold only if its `ptrain` macro-F1 minus the current one has a unit-bootstrap (speakers,
2000 resamples, seeded) 95 % lower bound > 0. HMC: the candidate with the highest `ptrain` macro-F1
(ties → the smaller model); every candidate's score is recorded. The architecture may therefore
differ between folds; the fold's metrics name it. `ptrain` is also the predictor's training set:
choosing f_raw on it does not touch `calib` or `test`, so the certificate's guarantee is unchanged.

## 8. Fixes found in the B smoke run and the 2026-11-11 run (2026-11-12)

Bookkeeping and execution fixes. None changes a rule fixed in §6–§7 or a pre-registered verdict.

1. **wav2vec2 inference device.** wav2vec2 was trained on the GPU but ran its `ptrain` inference
   on the fold's `--device` (CPU), which took too long. Its inference now runs on the GPU too
   (`resolve_device("auto")`), whatever `--device` says. This covers the probe parity check,
   `raw_decision` and the cloud GPU timing. Other f_raws are unchanged.
2. **wav2vec2 validation batch.** The validation predict in `train_w2v` uses batch 16. The optimizer
   state is still on the 4 GB GPU at that point, and batch 64 hit an out-of-memory retry. The
   training itself is unchanged.
3. **Cloud head time of waveform reps.** For a fold with a measured cloud-GPU f_raw time
   (the wav2vec2 f_raw, B), that time now applies to every waveform representation that the cloud
   decodes and runs through f_raw: raw, lossless and the wavelets. Before, it applied to raw only.
   Folds without a cloud-GPU time are unaffected. Test: `tests/unit/test_committee.py`.
4. **One f_raw per E1 profile.** E1 profiles fold 0's f_raw for every fold. With a per-fold choice
   (§7), folds may differ. `f_raw_choice` now raises if they do, instead of profiling the wrong
   model for some folds. The E1 summary names the chosen f_raw.
5. **`kai folds --stale`.** This retrains only folds built with another recipe or other windows
   and keeps the rest. Before, the only choices were an error or `--force`, which retrains all.
6. **Report.** `scripts/report.py` treated the slow-edge sweeps `e3_slow<k>` and `e3_k<k>` as
   price presets and crashed on their missing H1 keys. It now skips them there. The report also
   lists every experiment summary older than E1's as stale. E1 is rerun after every fold rebuild,
   so an older summary was computed on folds that no longer exist.
