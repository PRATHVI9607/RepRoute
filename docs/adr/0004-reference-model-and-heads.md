# ADR 0004 — Reference model, distilled heads, and other implementation decisions

Date: 2026-10-10 · Status: accepted · Touches: `tasks/`, `cloud/heads.py`, `link/des.py`, `data/pu.py`

## 1. The reference model `f_raw` is a low-capacity spectral-shape classifier

**What happened.** The first leave-bearings-out fold (repeat 0, fold 0) was trained with the
planned ResNet1D on raw waveforms. It memorised its training bearings and did not transfer:

| Partition (bearings never seen in training, except `train`) | Accuracy |
|---|---|
| train | 1.000 |
| ptrain | 0.633 |
| calib | 0.465 |
| test | 0.219 (macro-F1 0.159; majority baseline 0.182) |

The LightGBM feature heads behaved the same way (R3: train 0.997, test 0.291). A reference model at
chance makes "agreement with the raw decision" meaningless, so the model was replaced.

**How the replacement was chosen.** By bearing-grouped cross-validation **inside the training
partition of that fold only** (11 bearings; ptrain, calib and test were not used):

| Candidate | Inner-CV macro-F1 (mean, range) | Protocol |
|---|---|---|
| ResNet1D, BatchNorm input (the original) | 0.18 (0.15–0.22) | 2 × 2-fold grouped, 8 epochs |
| ResNet1D, per-window normalisation | 0.26 (0.24–0.29) | same |
| … plus gain / shift / noise augmentation | 0.22 (0.19–0.26) | same |
| STFT front-end + CNN | 0.18 (0.18–0.19) | same |
| … plus augmentation | 0.22 (0.19–0.26) | same |
| Scale-free time statistics + logistic regression | 0.32 (0.11–0.48) | 10 × leave one bearing per class out |
| Order-domain envelope spectrum + logistic regression | 0.54 (0.27–0.82) | same |
| Log band-energy **shape** of the spectrum + logistic regression | 0.58 (0.27–0.86) | same |
| Welch log band shape, `n_fft` ∈ {512, 1024, 2048}, 24 / 48 bands | 0.39–0.47 | 12 × leave one bearing per class out |
| … same features, 16-unit MLP | 0.33 (0.08–0.55) | same |

Three things are consistent across the table: features that remove the window's overall level beat
those that keep it, linear models beat trees and networks, and the spread across held-out bearings
is very large. The inner folds train on a single healthy bearing, so they are harsher than a real
fold (two or three).

**Decision.** `f_raw` = `SpecNet`: a fixed, parameter-free front-end (Welch power spectrum with
`n_fft = N/4`, 24 log-spaced bands, log energies minus their mean) followed by standardisation and
a multinomial logistic regression. The regularisation strength is the only hyper-parameter and is
picked per fold by grouped inner CV on the training bearings. The fitted weights are copied into a
torch module so that it exports to ONNX like every other model. The order-domain envelope features
scored similarly but need the shaft speed as a side input, which the wire frame does not carry.

The ResNet1D stays in the code as `--f-raw wave` so that the comparison can be reproduced.

**Gate G1 threshold (fixed here, before any fold of the new model was evaluated).** G1 passes if
the mean test macro-F1 of `f_raw` over the folds run is **≥ 0.40**; the majority baseline is ≈ 0.18
and three-class chance ≈ 0.33.

**Disclosure.** Test-partition numbers of fold r0f0 for the *discarded* model — label accuracy and
the marginal disagreement of each representation — were seen before this decision. No certificate,
selection or cost experiment had been run on Paderborn data at that point.

## 2. Heads and the supervised bottleneck are distilled from `f_raw`

Every feature / latent head is trained to reproduce `f_raw`'s **decision** on the training windows,
not the ground-truth label, and so is the task-aware bottleneck R6. The certificate measures
agreement with `f_raw`, so this trains directly for what is certified; it is also how supervised
compression is usually done (the compact student mimics the teacher).

The fairness rule of PRD §7 is kept: every representation gets the same Optuna budget (trials,
folds, rows) over the same two families — standardised logistic regression and LightGBM — scored by
bearing-grouped inner CV. Label fidelity is still reported against ground truth.

Deviation from the PRD: its text says "LightGBM/MLP head"; the MLP family is replaced by logistic
regression because the study above showed higher-capacity models transfer worse across bearings.

## 3. Smaller decisions

| Decision | Reason |
|---|---|
| int16 full scale = ±16 (signal units), not ±50 | The peak over the subset is 10.9; a ±50 scale left the top two bits always zero and gave the lossless codec a free 12 %. |
| Five of the twenty recordings per bearing and condition (36 052 windows) | Keeps a fold trainable on a laptop; configurable in `scripts/prepare_pu.py`. One published recording (`N15_M01_F10_KA08_2.mat`) is unreadable and is skipped and counted. |
| Lossless codec = delta → byte-plane shuffle → zstd | High bytes of int16 samples compress far better on their own. |
| Per-window overhead (probe, predictor, selector) is divided by `1 − ρ` too | They share the edge CPU with the encoder; the PRD formula left them unscaled. |
| The Disagreement Table is computed from the **exported ONNX models** | What is certified is then exactly what the cloud runs; the closed-loop test asserts equality. |
| ACI updates on the audited subsequence, without inverse-probability weights | 1/p spikes make the threshold swing between "raw only" and "everything". |
| Deferred raw windows are forwarded only when a raw frame fits between two arrivals, or once per `probe_s` on an idle link | Forwarding them as soon as the queue was momentarily empty starved live traffic on a saturated link; the periodic probe lets the bandwidth estimate recover after an outage. |
| The backlog the selector sees subtracts the estimated progress of the frame on the wire | Counting the whole in-flight frame made a link at ~100 % utilisation defer every second window. |
| Goodput samples on the edge remove the round trip for frames sent into an idle link and skip frames under 1 kB | In the first Docker run the estimate read 177 kbit/s on a 1 Mbit/s link: 156-byte frames were measuring the 10 ms delay, not the rate. |
| Docker shaping uses `netem rate`, not a token bucket (`tbf`) | A token bucket passes any frame smaller than its burst at line speed, so it does not behave like a link for single frames. |
| V2 calibration breaks cost ties towards the lower menu index | That is what the selector's `argmin` does; R6 and R7 both cost 16 bytes. Found by a property test. |
| The classical-feature cache is keyed on the data and on the codec / probe versions | A stale cache after a feature change would silently mix versions. |
| Probe statistics use products instead of `**`, cached band edges; the selector evaluates cost and feasibility in one pass | The first profile measured 1.06 ms for the probe — as much as an encoder. |

## 4. The timing profile is measured at the deployment duty cycle

**What happened.** The first profile timed each encoder in a tight loop (warm-up, 30 repetitions,
median), as PRD §9.6 prescribes. The first sim-to-real run (E7) then showed real p50 latency up to
78 % above the simulator. The cause: an edge handles one window per 64 ms period, and after each
idle gap the caches are cold and the CPU has clocked down (the laptop's governor is `powersave`
and cannot be pinned without root). The same code is 2.7–6.4× slower that way:

| | tight loop | one call per 64 ms |
|---|---|---|
| probe | 0.14 ms | 0.90 ms |
| R3 spectral encode | 0.42 ms | 1.53 ms |
| R2b wavelet encode | 0.74 ms | 2.02 ms |

**Decision.** `kai.profile.profiler.time_ms` takes a `period_ms` and E1 profiles every encoder,
decoder, head and the per-window overhead at the dataset's window period. E2, E3, E5, E7 and E8
were re-run with that profile; the earlier runs remain in the registry and are listed in the
results' run history. After the change the loopback gap fell to at most 16 %.

**Two findings that follow from measuring this way.**

1. *Processor sharing does not describe a duty-cycled edge.* With busy co-tenants pinned to the same
   core, sub-slice bursts are **not** slowed by `1/(1 − ρ)`: a task that wakes from sleep is scheduled
   promptly and finishes inside its slice, and the busy core is clocked up. The fitted load exponents
   are 0–0.24 instead of 1. The simulator uses the fitted values, so edge load ρ has little effect in
   the scenario grid. The real-socket phase-diagram run (E4) nevertheless measured latencies about
   35 % higher on average at ρ = 0.8 than the model predicts: the micro-benchmark under-states the
   effect on the full per-window path. The load model needs a capacity term, not an exponent.
2. *The per-window overhead is not negligible on this hardware.* Probe + predictor + selector take
   about 2.3 ms cold, more than any learned encoder (≈ 1.6 ms). It is 3.7 % of a 64 ms window period,
   but it is not "much cheaper than encoding".

**Still unmodelled** (visible in E7 / E7d): TCP/IP headers (≈ 52 bytes per packet, a third of a
156-byte frame) and about 1 ms of event-loop and socket time per window.
