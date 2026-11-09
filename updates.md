# Updates — 2026-10-30

What changed on `main` since `303c1c9` ("new datasets used and got good results for now"), why,
and what the new results say. Nothing here is pushed; `main` was updated locally only.

## 1. Branches reviewed and merged into local `main`

Two branches appeared on `origin`. Both were reviewed line by line, merged in a scratch worktree,
and checked there (ruff clean, `mypy --strict` clean on `contracts/ certify/ select/`, **158 tests
pass**, up from 132) before `main` was fast-forwarded to the merge (`40d602a`).

### `edge` (Ryan, 12 commits, 2026-10-21 → 10-24) — merged, code better than before

| Change | Verdict |
|---|---|
| `Edge._send`: token-bucket wait with a precise `time.sleep` instead of `loop.call_later` (≈1 ms timer quantum, larger than a 44-byte frame's serialisation time), and the wait subtracted from `edge_ms` (ADR 0013) | **Better.** Removes a constant offset the DES does not model. Trade-off: the sleep blocks the event loop, so ack reading pauses while a frame waits; harmless for a single-sensor emulation, documented in the docstring |
| `EdgeConfig.__post_init__` + `_validate_selector`: unknown selector names and cascade+ACI/drift now fail at construction instead of silently running greedy | **Better.** Fail-closed; the old fallthrough ran `GreedySafe` on a typo |
| `CostModel`: zero bandwidth → infinite latency instead of a divide warning | **Better**, no numeric change otherwise |
| E7 / E7d summaries record bundle id, git sha, profile hash and the selector | **Better** (provenance) |
| ADR 0013 (pacer), ADR 0014 (pin the `performance` CPU profile for timings), ADR 0015 (small-frame rate estimation), `docs/edge-work.md`, `docs/edge-results.md` | Kept as written |
| 4 new unit-test files (pacing, selector config, profile selection, metadata) | Kept |
| **Experiment outputs** (`experiments/e*/summary.json`, figures, `results.md`) | **Not better as a final state:** they are a **5-fold** rerun on Ryan's 16-CPU laptop and replace the 15-fold Paderborn study on `main`. Code merged as is; the Paderborn suite is rerun on 15 folds with the merged code (section 4) so the report is again 15 folds from one machine and one timing regime |

Ryan's findings worth keeping in mind (his machine, 5 folds, `performance` profile): H6 passes
(4.7 % boundary error), loopback E7 fails H7 (20.5 % and 42.3 % p50 gap in two runs — measurement
variance), Docker E7d passes H7 (2.2 %).

### `feature/cloud-telemetry-integration` (simran-s-patil, 2 commits, 2026-10-28) — merged

| Change | Verdict |
|---|---|
| `HeadZoo.decide_timed` → (decision, t_dec_ms, t_head_ms); `decide` now calls it | Neutral, no behaviour change |
| `CloudServer(on_decision=…)` callback; `kai.cloud.server --telemetry-out file.jsonl` writes one JSON line per decision | **Better:** measured cloud decode/head times per frame, opt-in, off by default; a failing callback is logged and never drops the reply |
| `cloud_process` kills the server if `terminate` does not stop it within 10 s | **Better** (no hung CI) |
| `realrun.parse_telemetry_jsonl` / `correlate_telemetry` | Fine. Two cautions: (1) the sink opens in **append** mode and the parser **raises on a duplicate (window, rep)**, so reusing one file across runs fails — pass a fresh path per run; (2) nothing in `bench/experiments.py` uses it yet, so E7 does not report measured cloud times |

### Not part of either branch

`src/kai/bench/bundle.py` has an **uncommitted local edit** (present before this session's work)
that removes the cascade certificate (`cascade.json`) from bundles and goes back to a fixed
`_building` temp dir. It breaks `s1_cascade` on the edge: with it, the cascade loopback test **hangs** instead of failing
(it passes on the committed file), and `ruff` flags its import order. It was left untouched; revert it with `git checkout src/kai/bench/bundle.py` unless it was intentional.

## 2. New datasets from the team note ("RepRoute: New Datasets and Screening Script")

The note's screening script is now `scripts/screen_dataset.py`. Changes from the note:
- it reads repo WindowSets as well as the note's `X / y / unit` files;
- it passes a `Generator` instead of using a module-level RNG;
- it counts raw as int16, as stored;
- it writes the verdict to `experiments/screening/<name>.json`.

Four datasets from the note were added, each following the PTB-XL pattern: a loader in `src/kai/data/`, `scripts/download_*.py`, `scripts/prepare_*.py`, `configs/dataset/*.yaml`, a unit test, and leave-units-out splits with 5 folds × 3 repeats. Readers are numpy/scipy only (no `mne`, `wfdb` or `soundfile` installed); every download is checksum-verified.

| Dataset | Units | Windows | Window | Classes |
|---|---|---|---|---|
| Sleep-EDF Expanded (cassette), EEG Fpz-Cz | 78 subjects, 153 nights | 195 469 | 30-s epoch resampled to 3072 samples at 102.4 Hz | W / N1 / N2 / N3 / REM |
| Icentia11k (segment 0 of the first 1000 patients) | 980 patients | 8 998 (balanced, ≤ 6 per patient and class) | 4096 samples at 250 Hz | normal / ≥ 1 PAC / ≥ 1 PVC |
| Speech Commands v0.02, 10 keywords | 2 492 speakers | 35 103 | 1 s at 16 kHz, zero-padded to 16 384 | yes … go |
| UCI HAR raw recordings (HAPT), accelerometer magnitude | 30 users | 2 509 | 256 samples at 50 Hz | 6 activities |

Not added: **SHHS** needs a data-access request, which the team must file. **CAPTURE-24** and **MIMII** were left for later: MIMII has only 7 machines per type, the same problem as the bearing datasets.

Notes:
- In the new datasets a unit has windows of several classes, so splits are leave-units-out and stratified only by a per-unit summary (Icentia11k: the patient's highest class) or not at all.
- `kai.tasks.train` hard-coded 3 output classes. Every trainer now sizes its output from the labels (`n_classes`), with a test in `tests/unit/test_n_classes.py`. Without this fix, no 5-, 6- or 10-class dataset could run.

### Screening results (20 000 windows max per dataset, 30 % of units held out)

| Dataset | f_raw F1 (chance) | Medium-good reps (2–25 % disagreement) | Reps used ≥ 10 % by the cheapest-first picker | Hardness AUC | More-units curve | Verdict |
|---|---|---|---|---|---|---|
| Paderborn (baseline) | 0.61 (0.33) | — | — | 0.52 | — | **FAIL**, as the note predicted |
| HUST | 0.43 (0.33) | R3 | R7, R4, R3 | 0.47 | — | FAIL |
| UORED | 0.44 (0.33) | R2, R3, R4, R5, R7 | R7 87 %, R4 12 % | 0.74 | flat | **MAYBE** |
| PTB-XL | 0.48 (0.33) | R3, R4, R5, R7 | R7 92 % | 0.81 | 22 % → 19 % | FAIL |
| Icentia11k | 0.51 (0.33) | R3, R4, R5, R7 | R7 87 % | 0.67 | 28 % → 23 % (10 → 686 patients) | FAIL |
| Sleep-EDF | 0.56 (0.20) | R3 | R7 54 %, R4 33 % | 0.64 | 60 % → 58 % | FAIL |
| Speech Commands | 0.54 (0.10) | none | R7 40 %, R3 25 %, R4 24 % | 0.59 | 89 % → 88 % | FAIL |
| UCI HAR | 0.58 (0.17) | none | R4 39 %, R7 37 %, R3 12 % | 0.56 | 77 % → 73 % | FAIL |

What the screening says:

1. **More units do not open up the menu.** The more-units curve is close to flat everywhere: Icentia11k goes from 28 % to 23 % between 10 and 686 patients. By the note's own rule, *the representations are the problem, not the unit count*. The idea that the unit count is the main lever is not supported.
2. **The note's picker collapses for a structural reason, not a data one.** Once one representation is both the cheapest (R7, 8 bytes) and good enough on most windows, a cheapest-first picker uses it almost always. On PTB-XL and Icentia11k four representations are "medium-good", yet one takes 87–92 % of windows. Paderborn's R6 at 99.5 % is the same effect. A dataset can only pass if no small representation is good on most windows, and then the small representations are mostly bad, as on Speech Commands and HAR.
3. **Hardness is predictable where the certificate is meant to work.** AUC 0.81 on PTB-XL and 0.67 on Icentia11k, but chance on Paderborn and HUST. That matches where the per-window certificate held (PTB-XL) and where it did not (the bearing sets).
4. **Caveats.** The screening's R5/R7 are PCA on raw samples, which is very weak for speech and EEG; the repo's trained autoencoder, VQ and bottleneck are much stronger. Per-window z-normalisation removes the gravity level that separates sitting, standing and lying in HAR. So FAIL here means "the note's cheap stand-ins do not diversify", not "the full pipeline cannot".

## 3. Paderborn rerun: 15 folds, merged code, pinned `performance` CPU profile

The full suite was run on this laptop (12 CPUs, `intel_pstate`), following ADR 0014: E1, E2, E3, E5, E5pi, E6, E8, E10, E4, E7, the `wired_gateway` preset, the figures and `results.md`. E7d needs Docker and was not rerun, so `results.md` shows Ryan's E7d (other machine).

| | Before (`303c1c9`, `powersave`) | Now (`performance`) |
|---|---|---|
| Probe + predictor + selector, 1 sensor | 2.28 ms | **0.59 ms** |
| R6 encoder | 2.65 ms | **0.70 ms** |
| f_raw macro-F1, 15 folds | 0.604 | 0.604 (deterministic) |
| E2 at α = 0.05: V0 / cascade / static risk; folds within α | 0.039 / 0.019 / 0.030; 12, 15, 15 of 15 | unchanged (does not depend on timing) |
| H3: SLO violations vs context-blind, bursty traces | +30.0 % | +25.1 % |
| H5: full menu vs best single family | +5.6 % | +6.1 % (need ≥ 10 %) |
| H6: break-even boundaries | 4 of 12 match, worst 4620 % | 2 of 3 match, worst 54 % — fails |
| H7: loopback sim-to-real p50 gap | 10.1 % (fail) | **3.6 % (meets the threshold)**, with Ryan's pacer fix |
| **`wired_gateway`** (edge CPU priced): oracle's choices | r4 73 %, raw 25 % (two reps) | **r4 98 %** (one rep) |
| `wired_gateway`: Kai V0 / cascade / static vs best-single | 61 % / 72 % / 78 % cheaper | **2.4× / 2.2× / 1.37× more expensive** |

**The most important correction in this update.** The headline "when edge CPU time is priced, per-window choice is far cheaper than any single representation" **does not hold** once timings are pinned. With an edge about 4× faster, compute is too cheap to trade against bytes, and one representation (R4) wins almost every cell. The earlier saving was an artifact of the `powersave` timing regime. README, `docs/presentation.md` and `docs/paper/draft.md` now say so. A Raspberry Pi is much slower than this laptop, so the regime where adaptivity pays may well exist there, but only a Pi run can show it.

## 4. Speech Commands: full pipeline (the best candidate from the screening)

Chosen because it has the most units (2 492 speakers) and is the only dataset where the screening's picker spread across three representations. The reference model was chosen by inner CV on fold r0f0's training speakers: the waveform CNN won with 0.892 macro-F1 (SpecNet 0.467, EnvNet 0.361; recorded in `experiments/model_search/speechcmd-v1.jsonl`). 5 leave-speakers-out folds, the same pre-registered hypotheses and thresholds. Report: [docs/research/results-speechcmd.md](docs/research/results-speechcmd.md). E6 needs a second operating condition, so it does not apply here; E7d needs Docker.

| Representation | Label macro-F1 | Disagreement with f_raw (test) |
|---|---|---|
| r0 raw = f_raw | **0.917** | 0 |
| r2a / r2b wavelet | 0.916 / 0.902 | 0.7 % / 4.3 % |
| r6 bottleneck (16 B) | 0.733 | 26.6 % |
| r3 spectral | 0.594 | 40.1 % |
| r7 VQ / r4 summary / r5 AE | 0.475 / 0.333 / 0.329 | 53–66 % |

| | Result |
|---|---|
| Certificate at α = 0.05: V0 / cascade / static risk; folds within α | 0.052 / **0.027** / **0.029**; 2, **5**, **5** of 5 (rerun: 0.0515 / 0.0273 / 0.029, same fold counts) |
| Exchangeability control | 0.050 at α = 0.05 (exact) |
| H1: oracle's main choices | r6 in 100 % of cells; best-single 15.8 % above the oracle (need 25 %). **Not supported** |
| H3: Kai V0 vs best-single (cost) | 25 % **more** expensive (first run: 31 %); static 44 %, cascade 35 %. Not supported |
| H3: SLO violations vs context-blind (bursty traces) | 84 % **more** (first run: 39 %). Not supported |
| H4: encode cost changes the choice | 2.8 % of high-load cells (first run: 16.0 %, which met the 15 % threshold). The first run's wavelet encode times were inflated by BLAS threading (ADR 0017); with honest timings the encode cost rarely changes the choice. Not supported |
| H5: mixed menu vs best family | +5.2 % (need 10 %). Not supported |
| H6: closed-form break-even | 1 measured argmin change, matched within 14.7 % (need 20 %). Threshold met (descriptive) |
| H7: loopback sim-to-real | First run: p50 gap 99.5 %, agreement 70.7 %, partly a **bug** (the pacer stamped frames after the wait, so the rate estimate read ~10 Mbit/s everywhere). After both fixes (ADR 0013 addendum, ADR 0017): gap 17.3 % (need 10 %), agreement 84.0 % (need 85 %). Still fails, now narrowly |

What it means:
Every number above is from the rerun after the two bug fixes in §5, unless marked "first run". The verdicts did not change, except H4, which lost its threshold once the timings were honest. Rerun report: [docs/research/results-speechcmd.md](docs/research/results-speechcmd.md).

- **The certificate result holds.** The static set and the cascade stay within α in every fold. The per-window certificate averages 0.052 at α = 0.05, essentially on α but 2 of 5 folds just above.
- **Cost: Kai loses fairly here.** The best single representation sends windows whose decision disagrees with raw only 3.3 % of the time (under α) and is still cheaper (8.55e7 against 10.7e7) with fewer SLO violations (0.32 against 0.44). Kai loses because when the certificate rejects R6 for a window, the next certified option is near-lossless (several KB; raw is 32 KB), and those frames are late. The menu has nothing in between; ADR 0016 adds mid-size wavelets as an exploratory fix.
- **The menu still collapses onto one representation for the oracle (R6)**, because the oracle prices bytes and ignores disagreement. Only the certified policies use the wavelets.

## 5. Fixes after the first Speech Commands run, and an exploratory menu

### Two bugs fixed (both in code the edge runs)

| Bug | Effect | Fix |
|---|---|---|
| The pacer stamped a frame's send time *after* its token-bucket wait | The edge's goodput estimate read ~10 Mbit/s at every shaped rate, so it picked frames too large for slow links (E7 agreement 70.7 % at 256 kbit/s) | Stamp when serialisation starts (`edge/runtime.py` `_send`; ADR 0013 addendum; test `test_a_paced_frame_is_stamped_when_its_serialization_starts`). Estimates now read 299 k / 905 k / 3.3 M at 256 k / 1 M / 5 M |
| `np.linalg.norm` and `np.dot` above ~10 k elements start OpenBLAS threads | The wavelet encode timed 65.9 ms at load 0 but 4.6 ms at load 0.5: the profile table, and every cost built on it, was wrong for the wavelets | Element-wise sums in `reps/classical.py` and `reps/features.py` (ADR 0017). Payloads byte-identical on 300 windows for every codec; probe features within 3e-12 relative. Wavelet encode now 2.3 ms |

The pre-registered Speech Commands run was redone with both fixes (§4). Paderborn E7 still passes after the pacer fix (gap 4.2 %).

### Exploratory menu: mid-size wavelets (ADR 0016) and a 64-byte R6

Not pre-registered. The question: does Kai lose on cost because the menu has nothing between R6 and a several-KB wavelet? Same 5 folds and seeds, `kai folds --extra-wavelets --bottleneck-dim 64`, work dir `transfer/speechcmd_x`. Report: [docs/research/results-speechcmd-x.md](docs/research/results-speechcmd-x.md).

| Representation | Bytes | Disagreement with f_raw |
|---|---|---|
| r2b wavelet (pre-registered) | 2 190 | 4.3 % |
| r2c wavelet (new, eps 0.15) | 914 | 17.0 % |
| r2d wavelet (new, eps 0.30) | 460 | 33.4 % |
| r6 bottleneck, 64-d (was 16-d) | 64 | 24.1 % (was 26.6 %) |

| | Pre-registered menu (rerun) | Exploratory menu |
|---|---|---|
| V0 risk at α = 0.05; folds within α | 0.0515; 2/5 | 0.051; 2/5 |
| Static / per-rep V1; folds within α | 0.029; 5/5 | 0.029 / 0.025; 5/5 |
| H1 best-single above oracle | 15.8 % | 20.0 % |
| H3 Kai V0 vs best-single (cost) | 25 % more | 25.5 % more |
| H5 mixed menu vs best family | +5.2 % | +2.2 % |
| H7 p50 gap / agreement | 17.3 % / 84.0 % | 12.7 % / 80.7 % |

The new wavelets did not fix the cost gap. The certificate holds with the larger menu (it was calibrated on the same rules), but the mid-size wavelets disagree too often (17 % and 33 %) to enter many safe sets. The oracle's main choice moved from r6 to r4 only because r6 grew from 16 to 64 bytes and r4 (24 bytes) became the cheapest representation.

### Where the cost gap comes from

On static links (exploratory breakdown in `results-speechcmd-x.md`; the pre-registered rerun shows the same pattern), Kai holds back most windows on slow links, because nothing in their safe set fits the deadline:

| Link | Best-single cost | Kai V0 cost | Windows Kai deferred |
|---|---|---|---|
| 64 kbit/s | 1.93e5 | 3.33e5 | 94.9 % |
| 256 kbit/s | 6.29e4 | 9.41e4 | 61.5 % |
| 1 Mbit/s | 3.97e3 | 3.35e4 | 35.9 % |
| 5 Mbit/s | 3.73e3 | **3.29e3** | 0.7 % |
| 20 Mbit/s | 3.73e3 | **3.17e3** | 0 % |

Kai is cheaper than best-single from 5 Mbit/s up. Below that it loses, and this is the honest reason: on Speech Commands every representation small enough for a slow link disagrees with raw on 24–65 % of windows, so the certificate refuses them, and deferral is priced as an SLO miss. Best-single sends anyway with no certificate.

The existing `liveness_first` variant (`kai_s1_v0_live`, exploratory) sends an uncertified frame when nothing certified fits. In the pre-registered rerun it costs 1.86e7, against 8.55e7 for best-single (4.6× cheaper), with SLO violations of 0.069 against 0.317. The price: 26.5 % of windows go out uncertified, and the disagreement of what was sent is 12.3 %. It is not a certified result, and it does not count towards H3.

### What would fix it, and what it costs

- A representation that is small *and* close to raw on audio, for example a log-mel spectrogram head (new rep, ADR, about 6 h of work and compute). This is the only change that attacks the cause.
- Report Speech Commands as the dataset where the certificate holds and adaptation does not pay. The pre-registration requires this anyway; nothing above is tuned away.

## 6. Review fixes and the full rerun on five work dirs (2026-11-01)

A whole-codebase review found correctness, robustness and speed problems. All of them were fixed before the rerun (ADR 0020, ADR 0021; 200 tests pass, lint and `mypy --strict` clean). The fixes that change numbers:

| Fix | Where | Effect on results |
|---|---|---|
| Edge pacing no longer blocks the event loop; arrival time is the scheduled time, and the deadline is reduced by the lag; `edge_ms` times only decide + send | `edge/runtime.py` | The H7 sim-to-real gap (below) |
| Audit feedback only for `certified` decisions, in the edge and in the DES | `edge/runtime.py`, `link/des.py` | ACI and live variants; the certificate covers certified decisions only |
| Decode and head timed in one call, as the server runs them | `profile/profiler.py` | Cloud-side timings in E1 |
| Group CRC: one exchangeable sample per unit (bearing, speaker, subject) | `certify/crc.py`, new policy `kai_s1_v0_group` (ADR 0020, exploratory) | New rows in E2 / E3 |
| H1(b), H7 with unmeasured rows, E2 bootstrap blocks, E8 timed in 30 blocks (median + MAD), reuse refused on a provenance mismatch | `bench/experiments.py`, `bench/evaluate.py` | Report columns; no verdict depends on them |
| E7 capped at 30 min of real time per link rate | `bench/experiments.py` (ADR 0021) | Sleep-EDF E7 replays 60 windows per rate, not 300; every other dataset keeps 300 |

The replication unit stays the fold, as pre-registered. Folds of different repeats share test bearings, so the fold-level p-values are optimistic; every report says so. Holm is applied within each report (8 hypotheses), not across the five datasets.

Then everything was rerun in one queue on the pinned `performance` profile: HAR, Speech Commands, Speech Commands with log-mel (ADR 0018, exploratory), Sleep-EDF and Paderborn, then E7d in Docker on this laptop. Reports: [results.md](docs/research/results.md) (Paderborn), [results-har.md](docs/research/results-har.md), [results-speechcmd.md](docs/research/results-speechcmd.md), [results-speechcmd-logmel.md](docs/research/results-speechcmd-logmel.md), [results-sleepedf.md](docs/research/results-sleepedf.md).

### Reference models on unseen units

| Dataset | Folds | f_raw macro-F1 (majority) | Closest compact representation |
|---|---|---|---|
| Paderborn | 15 | 0.604 (0.184) | r6, 16 B, 0.7 % disagreement |
| UCI HAR | 5 | 0.779 (0.053) | r6, 16 B, 13.7 % |
| Speech Commands | 5 | 0.917 (0.018) | r6, 16 B, 26.6 % |
| Speech Commands + log-mel | 5 | 0.917 (0.018) | smallest log-mel, r10b, 146 B, 11.4 % (r10a, 258 B: 10.2 %) |
| Sleep-EDF | 5 | 0.672 (0.105) | r6, 16 B, 22.9 % |

### The certificate at α = 0.05 (mean risk; folds within α, descriptive)

The cascade (ADR 0011) and the static set (ADR 0008) were added after the first results and are exploratory; only V0 and V1 carry pre-registered hypotheses (H2a, H2b). Group CRC: risk pooled over windows / mean over units (the quantity it bounds).

| Dataset | V0 | Cascade (exploratory) | Static (exploratory) | Group CRC (exploratory) |
|---|---|---|---|---|
| Paderborn | 0.039; 12/15 | **0.019; 15/15** | 0.030; 15/15 | raw only |
| UCI HAR | 0.049; 3/5 | **0.025; 5/5** | 0; 5/5 | raw only |
| Speech Commands | 0.052; 2/5 | **0.027; 5/5** | 0.029; 5/5 | 0.048 / 0.050; 3/5 |
| Speech Commands + log-mel | 0.052; 2/5 | **0.027; 5/5** | 0.029; 5/5 | 0.046 / 0.051; 3/5 |
| Sleep-EDF | 0.041; 4/5 | **0.023; 4/5** | 0.041; 5/5 | raw only (works at α = 0.1: 0.032; 5/5) |

V0 bounds the *expected* risk, so some folds above α are consistent with an expectation bound (the ≈ 47 % figure assumes window-level exchangeability, which leave-units-out splits do not give). Group CRC bounds the expected mean loss of a new unit exchangeable with the calibration units; with class-stratified bearings, units are exchangeable only within a stratum (ADR 0020). Its finite-sample correction needs at least ⌈1/α⌉ − 1 calibration units: 19 at α = 0.05, 9 at α = 0.1. Speech Commands has enough at every α, Sleep-EDF only at α = 0.1, and Paderborn and HAR at none; there it returns raw-only, which is correct but saves nothing. The unit floor comes from the method's own +1 term, so more units is the only remedy; SHHS (≈ 5 800 subjects) is the planned test. On Speech Commands the per-unit mean (0.050, and 0.051 with log-mel) sits on α.

### Hypothesis verdicts

"met" = threshold met but Holm does not reject; **supported** = both.

| | Paderborn | HAR | Speech Cmd | Speech Cmd + log-mel (exploratory) | Sleep-EDF |
|---|---|---|---|---|---|
| H1 diversity | no | no | no | no | no |
| H2a V0 risk ≤ α | no | no | no | no | met |
| H2b V1, ≥ 90 % folds | no | no | no | no | no |
| H3 adaptation value | no | no | no | no | no |
| H4 encode cost matters | no | no | no | no | no |
| H5 heterogeneity pays | no (+5.8 %) | met (+10.9 %) | no (+5.0 %) | **supported (+15.9 %)** | **supported (+21.0 %)** |
| H6 closed-form boundaries | no | no | no | no | no |
| H7 sim-to-real (p50 gap) | met (4.2 %) | met (5.6 %) | no (12.5 %) | met (8.8 %) | met (6.1 %) |
| H7 in Docker (E7d) | met (1.0 %) | — | — | — | — |

### What changed against the previous runs (§3, §4)

| | Before | Now | Why |
|---|---|---|---|
| Speech Commands H7: p50 gap / agreement | 17.3 % / 84.0 % | 12.5 % / 98.7 % | Non-blocking pacing and lag-aware deadlines. Agreement now passes; the gap still fails |
| Paderborn H7 (loopback) | 3.6 % | 4.2 % | Within run-to-run noise; still met |
| Paderborn E7d | Ryan's machine, 2.2 % | this laptop, 1.0 % | First Docker run here |
| Paderborn H6 | 2 of 3 changes matched, worst 54 % | 5 of 16 matched, worst 8 716 % | More measured argmin changes after the pacing fix. Most are between near-tied compact representations; checking this is an agreed next step |
| Speech Commands H6 | 1 change, 14.7 % (met) | 5 changes, 2 matched, worst 12 132 % | Same pattern: H6 lost its threshold |
| Paderborn E2 (V0 / cascade / static) | 0.039 / 0.019 / 0.030 | unchanged | Calibration does not depend on timing |
| Speech Commands H3 cost vs best-single | 25 % more | 26 % more | Unchanged within noise |

H6 losing its threshold on Speech Commands is recorded as a loss, not tuned away.

### What it means

- **The certificate is the result that holds (exploratory for the cascade and the static set).** The cascade's mean risk is below α on every rerun dataset (0.019–0.027 at α = 0.05). Descriptively, 29 of 30 folds over the four datasets are within α (the log-mel run reuses the Speech Commands folds and is not counted; the folds are not independent). The union bound only gives E ≤ α; the margin is the slack of the α/2 + α/2 split. On Paderborn its cheapest certified payload is 0.2 % of raw.
- **H3 fails on every dataset.** On HAR, Kai V0 is 26.4 % cheaper than best-single (CI [+23.3 %, +32.4 %]; H3(a) met) but misses H3(b): SLO reduction +1.7 % against the 30 % target. On Paderborn, Speech Commands and Sleep-EDF Kai costs more than best-single. On Paderborn (E3, mean over α and cells), Kai V0 costs 3.1× best-single (9.46e7 against 3.04e7 per fold), the cascade 2.0× and the static certified set 1.4×. Kai V0 holds back 28.6 % of windows: their safe set excludes r6, the next certified options are KB-sized wavelets or raw, these do not reach the cloud before the deadline, and each held-back window is priced as an SLO miss (SLO violations 0.30 against 0.19). Best-single sends r6 on those windows anyway, uncertified: 22.5 % of its frames are outside the safe set, and its marginal screening on calib bounds nothing once it picks among several representations (ADR 0022). On HAR, best-single is inside the safe set on every window. The static set has the fewest SLO misses of the four (0.14). The per-window predictor has no signal on Paderborn (AUC < 0.5; `docs/research/predictor-search.md`), so a fixed set beats every adaptive variant.
- **A better representation is what moves the needle, not more tuning.** Log-mel gives Speech Commands a 146-byte representation at 11 % disagreement, and with it H5 is met (+15.9 %) and H7 is met (8.8 %). It is exploratory (ADR 0018): a second family on the same speakers, built after seeing the first run, so not confirmatory.
- **H1, H4 and H6 fail everywhere.** One compact representation wins most cells. Encoders take microseconds on this laptop, so load rarely changes the choice; only a Raspberry Pi run can test H4 as intended.

### Next (agreed 2026-11-01; each fix ADR-first, exploratory, original verdicts unchanged)

1. H3: send `r1_lossless` instead of raw where raw is the only certified option (decision-identical by construction). Expected gain is small: lossless is still KB-sized, so it helps only on links fast enough for lossless but not for raw.
2. H3: fall back to the static set when the predictor's `ptrain` AUC is near 0.5, and report cost at the same guarantee.
3. H6: count a measured crossing only when the latency gap exceeds the MAD noise, re-analysing the stored E4 measurements.
4. H2a: add a pre-registered non-inferiority test (risk ≤ α + margin) alongside the CI.
5. SHHS: the user registers for access on 2026-11-02. It is the dataset with enough units for group CRC and the first unseen test of the fixes above.

## 7. Exploratory fixes after the rerun (ADR 0022, 2026-11-02)

All four rules were fixed in ADR 0022 **before** they were computed. The test data had already been read, so every number below is **exploratory**. The pre-registered verdicts of §6 stand; the E4 re-measurement does not replace H6's verdict of record (see below). The first unseen test is SHHS.

| Fix | Result |
|---|---|
| 1. Lossless fallback | **Not needed.** `r1_lossless` is already in every menu and never disagrees with raw (0.0 on every dataset). The ladder already sends the cheapest certified representation that fits. Windows are held back only when even lossless (0.4–19 KB) misses the deadline |
| 2. Best-single's certificate (checked) | B4 picks the cheapest representation whose marginal calibration disagreement passes the CRC-style check (n·R̂ + 1)/(n + 1) ≤ α. That check is a valid expected-risk bound only for one representation named before calibration. B4 applies it to every representation and keeps the cheapest that passes, so its choice carries no guarantee (multiple testing: if several cheap representations sit just above α, one of them passes by chance). Its frames are uncertified. H3 compares Kai with a cheaper but uncertified choice; the certified fixed-choice comparator is `kai_s1_v0_static`. An earlier draft of this section said the opposite; the theory review showed it wrong |
| 2. `kai_s1_auto` (static set when the ptrain CV AUC < 0.55, else V0) | **Picks the cheaper branch, but does not rescue H3.** Cost against best-single: Paderborn −115 % (static in 12 of 15 folds; V0 alone −211 %, static alone −41 %); HAR +26.4 %, Speech Commands −26.0 %, Sleep-EDF −63.3 % (V0 in every fold on these three, and V0 is the cheaper branch there: static is +0.8 %, −44.9 %, −102.6 %). The only wrong picks are 3 Paderborn folds with AUC 0.555–0.567, just above the cut. Where V0 lost to best-single, both branches lose |
| 3. H6 noise-aware | **No boundary survives** on any of the four re-measured datasets: every measured argmin change is within 2·√(MAD² + MAD²) of a tie. The noise is the per-window MAD, about 2.7× a 2-SE rule on the difference of two medians of 25 iid windows, and the rule also drops real crossings whose break-even lies near a grid point. It is strict for declaring a boundary, and so biased towards "tied". The rule was fixed in the ADR and is kept; a test with the median's standard error or a finer grid was not run |
| 4. H2a non-inferiority (mean V0 risk < α + 0.01, max p over α) | **Rejects** on Paderborn (p = 0.032), Speech Commands (0.040) and log-mel (0.034). **Does not reject** on HAR (0.338; V0 is 0.104 at α = 0.1) or Sleep-EDF (0.084). Fold p-values are optimistic; not in the Holm family |

### E4 was re-measured (needed for fix 3)

The pre-registered H6 was recomputed on the new measurements:

| | §6 | Re-measured |
|---|---|---|
| Paderborn | 5 of 16 matched, worst 8 716 % | 1 of 10 matched, worst 128 %; fails |
| HAR | 3 of 10, worst 3 536 % | 5 of 11, worst 3 979 %; fails |
| Speech Commands | 2 of 5, worst 12 132 %; fails | 1 of 3, 0.6 %; **threshold met** |
| Sleep-EDF | 2 of 8, worst 388 % | 1 of 11, worst 43 %; fails |

The pre-registered H6 verdict of record stays the §6 run: **fail on every dataset, Speech Commands included.** Speech Commands' H6 has gone met (14.7 %, §4), fail (12 132 %, §6) and met (0.6 %) across three measurements; the third was triggered by an exploratory fix, and the noise-aware count drops its one matched crossing. A verdict that changes between measurements of the same thing is not counted as met. The regenerated report shows the latest measurement and says so. Every run stays in the registry.

### What it means

- **Consistent with the bound, not stronger evidence for it.** V0's fold-mean risk is below α + 0.01 at p < 0.05 on Paderborn and on Speech Commands (two menus; log-mel shares the Speech Commands test units). On Speech Commands the point risk is above α at every α; on Paderborn the bearing-block CI touches α + 0.01 at α = 0.1. The margin is half of α at α = 0.02.
- **H3 is not rescued by gating.** The gate picks the cheaper of V0 and the static set on every dataset but three Paderborn folds, and on three of four datasets both cost more than best-single. Best-single is cheaper than every Kai variant tried, but it is uncertified.
- **H6 crossings are not resolved by this measurement.** Under the ADR 0022 rule no crossing survives. That rule is strict, so this does not show that H6 cannot be tested; a median-SE rule or a finer grid might.
