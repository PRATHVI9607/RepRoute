# ADR 0009 — What was deliberately not built, and what would change that

Date: 2026-10-14 · Status: accepted (each row is a recommendation; the group can overrule it)

The PRD lists items as Could or Stretch, and the work split added more. After the second and third
passes the evidence is against spending time on several of them. This record says which, why, and
what result would reopen each. Nothing here was attempted and abandoned silently: where something
was tried, the numbers are cited.

| Item | Decision | Evidence | Reopen if |
|---|---|---|---|
| **R8, JEPA-style predictive latent** (PRD kill-switch) | **No-go** | A latent only matters if a head on it repeats the reference model on unseen bearings. Three generic learned latents trained on 11–14 bearings do not (R5 52 %, R7 45 %, a CNN bottleneck 46–49 % disagreement, with or without relabelled augmentation — stage D of [model-search.md](../research/model-search.md)). A predictive pretext task adds no bearings. | A generic latent reaches < 10 % disagreement in inner CV, e.g. after pretraining on a much larger pool of unlabelled bearings |
| **R9, MiniRocket-style features** | Not built | Random convolutional features are a strong baseline for *label* accuracy; here a representation must repeat EnvNet, whose decision rests on a peak-to-floor ratio of envelope spectra. R3's 128 bytes of engineered envelope and spectrum features disagree in 36 % of windows. Adding a 0.3–1 kB representation changes the menu, the cache and every fold. | The group wants a learning-free *label* baseline next to `f_raw` |
| **R3 redesigned to follow EnvNet** | Not built | Making R3 "per-band envelope-spectrum ratios" would make it EnvNet's own front-end in float16 (192 bytes): the same idea as R6 at twelve times the size. | The decision on whether the edge may run `f_raw`'s front-end goes against R6 — then R3 is the place to discuss it, not a third representation |
| **`lz4` variant of R1** | Not built | R1 is never the cheapest safe choice in any cell of the grid; a faster lossless codec changes no decision. | Lossless becomes competitive (raw-dominated regimes on a slow edge) |
| **S3, bandit on cost-model residuals** | Not built | Selection hardly matters in this menu: with one 16-byte representation near 1 % disagreement, the greedy and MPC selectors already make the same choice in every cell. A learner that corrects latency predictions cannot change an argmin that is not close. | A menu with real trade-offs (several representations within a factor of two in cost) |
| **S4 / B12, RL selector** | Not built | Same reason; the PRD wants it only to answer "is RL needed?", and with this menu the answer is visible without it. | As for S3 |
| **C6 / E9, escalation across families** | Not built | Escalation pays when a cheap representation is *sometimes* insufficient and the cloud can tell. Here the cheap representation that works disagrees in 0.7 % of windows, and the others are near chance. | A representation with 5–20 % disagreement that the cloud can flag from its own confidence |
| **"Edge inference" as its own policy** | Not built | R6 already is the reference model's front-end; running the last linear layer on the edge saves 15 bytes per frame. The report states the equivalence next to the profile table. | — |
| **All 20 recordings per bearing** | Not built | The error is at the level of whole bearings (about a quarter of unseen bearings misclassified in every window), and four times more windows of the same 29 bearings add no bearings. Extracting them needs about 18 GB of intermediate files. | A model that is limited by windows per bearing, not by bearings |
| **Feature-weighted (covariate-shift) calibration** | Not built | With the certified static safe set the calibration is a handful of marginal rates over five bearings; there is no per-window score to reweight. For the per-window predictor, the probe features do not separate a new condition from a new recording ([ADR 0008](0008-simulator-timing-static-certificate-traces.md) §5), so weights estimated from them would be noise. | A probe that separates operating conditions |
| **Cheaper probe (time statistics only)** | Not built | Replaced by a stronger remedy: the static safe set needs no probe at all, and the duty-cycle profile removed most of the modelled cost on busy nodes. | The per-window predictor becomes informative again |
| **arm64 image build on this machine** | Not done | Needs QEMU binfmt handlers registered with a privileged container; the CI job does it on a runner. Base images are pinned by digest to the multi-arch index, so the arm64 build uses the same pin. | — |

## Not possible here

- **Raspberry Pi runs and measured energy** — no access to the Pi, no power meter.
- **Turning CI on, branch protection** — needs a push, which the owner does.
- **DVC remote** — needs a storage location and credentials.
- **Human review of Proposition 1** — an automated review was done (ADR 0007); it is not a substitute.
- **Licence choice, role assignment, deadline, venue** — the group's decisions.
- **Measured uplink capacity traces** — the public 4G dataset used has downlink throughput only
  (ADR 0008 §4).
- **20 repeats × 5 seeds** — about 17 hours of fold training and several days of grid simulation on
  this laptop. Three repeats and three seeds were run.
