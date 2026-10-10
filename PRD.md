# Kai — Product & Research Requirements Document

*Sufficiency-certified representation selection for edge–cloud industrial IoT*
*(project name: **Kai**; the Python package is `kai`)*

| | |
|---|---|
| **Project** | Representation-Aware Adaptive Edge-Cloud Offloading, v2 ("Kai") |
| **Document version** | 2.0 — supersedes the *Project Review* docx (v1, evidence cutoff 30 Sep 2026) |
| **Date** | 10 Oct 2026 |
| **Team** | 5 members (roles A–E, §14) |
| **Planning assumption** | Kick-off Mon **12 Oct 2026**, 16 working weeks → **29 Jan 2027**. If your real deadline is shorter, cut by the MoSCoW list in §15.3, never by cutting tests, calibration, or the evaluation protocol. |
| **Status** | Plan reviewed (red-teamed, §2), numerically sanity-checked (§6.3), dependency set resolved on 4 platforms (§10.2) |
| **Honesty clause** | Nobody can honestly promise "nobody has ever done this." This PRD instead builds a *defensible, checkable* novelty claim (§3) and a protocol to re-verify it before submission (§19). |

> **How to read this:** §0 is the 2-minute version. §2 explains what was wrong with v1 and how it was fixed. §5–§6 are the architecture and the theory. §10–§13 are the tooling. §14–§16 are who does what, when, and what to download on Day 0. §21 holds copy-paste starter files.

---

## 0. TL;DR

**Problem.** Industrial edge nodes must decide, for every sensor window, *what to send* — not just *whether* to send. The same window can be shipped as raw samples, losslessly compressed, wavelet-compressed, engineered features, summary statistics, or a learned latent. Existing work tunes the compression of *one* family, or switches between two modes. Nobody (that we could find) offers a runtime that picks among a **heterogeneous menu** and can **certify** the choice.

**Idea (Kai).** Split the decision into two layers:

1. **Certify.** A cheap *sufficiency predictor* scores, per window, how likely each representation is to make the cloud model *disagree with what it would have decided from raw data*. A *conformal risk-control* calibration turns those scores into a **safe set** of representations with a distribution-free guarantee: `E[disagreement with raw decision] ≤ α`.
2. **Optimize.** *Any* cost-aware policy (greedy, MPC, bandit, RL) then chooses the cheapest representation **inside the safe set** given live bandwidth, queue backlog, edge load, energy, and deadline.

Because the guarantee holds for *every* member of the safe set, the optimizer can be arbitrarily clever or learned **without voiding the guarantee**. We call this **certify-then-optimize**. It is adaptive-bitrate streaming (ABR) for machine perception, with a statistical certificate replacing "perceptual quality."

**Headline deliverables.**
- `kai` — open-source runtime (edge, cloud, link-sim, selector, certifier).
- **Kai-Bench** — reproducible benchmark + simulator where any representation plugs in via one contract.
- **Representation Phase Diagram** — a map of which representation is optimal as bandwidth × deadline × edge load vary, with closed-form break-even thresholds validated against measurements.
- A paper with pre-registered hypotheses, bearing-wise leakage-safe evaluation, and a real Raspberry Pi validation.

---

## 1. Goals, non-goals, success criteria

### 1.1 Goals
- **G1** Make representation choice a first-class runtime decision across ≥ 8 representation families.
- **G2** Provide a *statistical sufficiency certificate* (risk control) for the choice, valid on held-out machines (bearings).
- **G3** Account honestly for *all* costs: probe, encode, queueing, transmission, decode, head inference, energy, storage.
- **G4** Produce a regime theory (break-even thresholds) that predicts measured behavior.
- **G5** Ship a reproducible, version-pinned, one-command-setup codebase that runs identically on Linux, WSL2, macOS (Docker), and Raspberry Pi (aarch64).

### 1.2 Non-goals (explicit, to protect scope)
- Not inventing a new encoder, a new wireless protocol, or a new RL algorithm.
- Not claiming multi-task or multi-modality generality from one task. Images/video are out.
- Not claiming privacy. Privacy leakage is an *optional measured extension* (E-priv), never a headline.
- JEPA/TS-JEPA is an **optional** candidate, kill-switched at Week 11. It is never in the novelty claim.

### 1.3 Success criteria ("done" means all Must items true)
| # | Criterion | Measured by |
|---|---|---|
| S1 | Certificate holds: mean held-out disagreement ≤ α (+CI) for α ∈ {0.02, 0.05, 0.10} | Exp E2 |
| S2 | Kai beats best-single-representation baseline on total cost at matched risk | H3 |
| S3 | Regime map reproduced on real Pi + netem within tolerance | H6, H7 |
| S4 | Every number in the paper regenerates from `make paper-artifacts` from a clean clone | CI nightly |
| S5 | ≥ 85 % line coverage on `certify/` and `select/`; mypy-strict clean on `contracts/ certify/ select/` | CI |
| S6 | Negative results reported honestly (pre-registered hypotheses, §4) | Review |

---

## 2. Review of the v1 plan: what stays, what was broken, how it is fixed

### 2.1 Strengths of v1 (kept and amplified)
| Keep | Why it is good | How Kai amplifies it |
|---|---|---|
| Representation, not protocol, is the research object | Correct, narrow, defensible | Formalised as a *Representation Contract* (§5.4) |
| Conservative novelty stance | Prevents embarrassing over-claims | Added quantitative novelty-kill criteria and a weekly novelty log (§19) |
| A/B/C decomposition (generate / select / act) | Clean organising principle | Becomes the code layout and team split |
| Minimum-sufficient-representation concept | The core insight | Made *operational and certifiable* via risk control |
| Software-first, public datasets, netem | Feasible for students | Added discrete-event sim + one real Pi for sim-to-real |
| Constrained optimisation over weighted scores | More interpretable | Hard constraints = safe set + deadlines; weights only from physical *price presets* |
| JEPA is optional | Avoids a fragile premise | Kept, time-boxed with a hard kill date |
| Go/no-go + pivot table | Good risk hygiene | Made *quantitative* (§9.7) |

### 2.2 Flaws found in v1 and the fixes (the red-team ledger)
Severity: **C** = would invalidate results, **H** = would weaken the paper, **M** = quality/efficiency.

| ID | Sev | Flaw in v1 | Consequence | Fix in Kai |
|---|---|---|---|---|
| F1 | **C** | Fidelity `F_r` is *measured after* sending, but the selector must choose *before* sending. v1 never says how the edge knows fidelity at decision time. | The selector is un-implementable as drawn, or silently cheats with labels (oracle leakage) | **Sufficiency predictor** `ĝ_r(x)` on cheap probe features + conformal calibration (§6) |
| F2 | **C** | "Fidelity relative to raw baseline" is ambiguous (label accuracy vs agreement with raw model). | Two incompatible definitions get mixed in plots | Two named quantities: **label fidelity** (F1/AUROC vs ground truth) and **decision agreement** (same output as raw-reference model). Certificates use agreement; reports show both |
| F3 | **C** | Dataset split unspecified. CWRU/Paderborn windows from the same recording leak across train/test; a 2026 study shows segment- and condition-wise splits inflate results | Inflated accuracy → meaningless "fidelity" → bogus savings | **Bearing-wise (leave-bearings-out) splits only**, plus cross-operating-condition test; CWRU demoted to smoke test (§8) |
| F4 | **C** | Architecture draws "generate *all* representations, then select." | Wastes edge compute; the selector's own cost contradicts the thesis | **Select-then-encode**: only a cheap *probe* runs on every window; exactly one encoder runs (§5.2) |
| F5 | **H** | Time-series windows are autocorrelated; conformal/IID assumptions ignored | Certificate would be invalid or overstated | Exchangeability unit = *bearing/run block*; block-bootstrap diagnostics; adaptive conformal (ACI) for drift; limitations stated (§6.4) |
| F6 | **H** | Decisions treated as independent. A large payload now delays later windows. | Understated latency, wrong selection under bursts | **Queue-aware state** (backlog bytes) and MPC lookahead; DES models the queue (§5.5) |
| F7 | **H** | Cloud head/decoder per representation not versioned. Re-training an encoder silently breaks the cloud. | "Version issues" at runtime | **Contract hash + version negotiation** in every frame (§5.4) |
| F8 | **H** | Baselines too weak: no "static profile-table selector," no adaptive-compression baseline, no two-mode switch. | Reviewers say "straw men" | Added B8–B11 (§9.1) |
| F9 | **H** | Missing two directly relevant 2025 papers (conformal model selection for edge AI; conformal edge-cloud cascade) | Novelty claim could be attacked | Added to prior-art table with explicit differentiation (§3.3) |
| F10 | **H** | Cost is a weighted sum with arbitrary weights. | One weight vector dominates conclusions | Weights come only from **physical price presets** (J, ms, bytes→$), and results are reported on Pareto fronts (§5.6) |
| F11 | **H** | Hypotheses are qualitative, no pre-registration; no statistics plan. | Cherry-picking risk | **H1–H7 with numeric thresholds**, committed before experiments; paired bootstrap CIs, ≥ 5 seeds (§4, §9.6) |
| F12 | **M** | Energy "measured or calibrated model" with no plan. RAPL is unavailable in VMs/containers. | Energy numbers unusable | Pi + inline USB power meter for calibration; everywhere else *labelled modelled* (§5.6, §16) |
| F13 | **M** | Reproducibility is "MLflow/W&B/local JSON" — not a decision, and MLflow's dependency tree is heavy. | Version conflicts | `uv.lock` + Docker + Hydra + DVC; MLflow/DVC installed as isolated `uv tool`s (§10) |
| F14 | **M** | Team plan assumes 4 members; no ownership, interfaces, or timeline. | Integration chaos | 5 roles, CODEOWNERS, interface contracts, 16-week schedule (§14–15) |
| F15 | **M** | No plan for non-IID cloud model drift ("task shift") beyond a bullet. | Weak robustness section | Controlled shift experiments: operating-condition holdout + head re-train with version bump (§9.2) |
| F16 | **M** | Learned-latent list lacks a *task-aware vs task-agnostic* contrast and a discrete (byte-cheap) option. | Misses the cleanest scientific comparison | Added supervised bottleneck (R6) vs unsupervised AE (R5) vs VQ codebook (R7) |
| F17 | **M** | "Fallback" when nothing meets the deadline is undefined. | Undefined behavior at the worst moment | Explicit **degradation ladder** and an *uncertified-rate* metric (§5.3) |
| F18 | **M** | No mention of token/AI-assist workflow, code quality gates, or dev environment. | Slow, inconsistent code | §12–§13 |

### 2.3 Strengths added beyond v1 ("the goat upgrades")
1. **Certify-then-optimize** decoupling (the central new idea).
2. **Representation Phase Diagram** with closed-form break-evens, validated on hardware.
3. **ABR-for-machines** framing: borrows mature MPC/ABR control ideas rather than reinventing RL.
4. **Contract-based plug-in benchmark** (anyone adds a representation in ~50 lines).
5. **Leakage-safe evaluation as a contribution**, not a footnote.
6. **Sim-to-real validation** on a Raspberry Pi with measured power.
7. **Pre-registered, falsifiable hypotheses** with honest pivot paths.

---

## 3. Novelty thesis (defensible, narrow, checkable)

### 3.1 Contribution list
| ID | Contribution | Type | Strength |
|---|---|---|---|
| **C1** | **Certify-then-optimize**: safe-set construction over a *heterogeneous* representation menu with a conformal-risk-control guarantee on *decision agreement with the raw-data decision*, valid for *any* downstream cost-aware selection policy | Method + proposition | Primary |
| **C2** | **Select-then-encode** runtime that charges the probe, predictor, selector, and encoder to the same budget as transmission | System | Primary |
| **C3** | **Representation Phase Diagram**: regime map + closed-form break-even thresholds with encode-cost and queueing, validated against measurements | Analysis | Primary |
| **C4** | **Kai-Bench**: contract-based plug-in benchmark, leakage-safe protocol, trace-driven link emulation, DES + real-Pi validation | Artifact | Medium–Strong |
| **C5** | Empirical study: when are summaries/features sufficient vs when do learned latents pay for their encode cost (task-aware vs task-agnostic vs discrete) | Empirical | Medium |
| **C6** | *(Stretch)* Sufficiency-gated **escalation** across heterogeneous families (send cheap first; send residual only on cloud-side uncertainty) | Method | Optional |

### 3.2 Claims we will NOT make
- "First semantic / task-oriented offloading." (false)
- "Task-aware compression is new." (false)
- "JEPA at the edge is new." (false — TS-JEPA, CHARM exist)
- "Adaptive offloading is new." (false)
- "Conformal prediction at the edge is new." (false — see §3.3)
- "Learned representations are private." (unsupported)
- "Guarantee under arbitrary distribution shift." (false — see §6.4)

### 3.3 Closest prior art and differentiation
Items 1–10 come from v1. **Items 11–12 are new** and are the most dangerous threats to C1; verified against their arXiv abstracts on 10 Oct 2026.

| # | Work | What it does | Kai differentiation |
|---|---|---|---|
| 1 | Chen et al. 2024/2026 (semantic multi-task offloading, QoE) | Joint semantic factor + resources | One semantic family; no heterogeneous menu; no certificate |
| 2 | Li et al. 2024 (graph-RL dynamic semantic compression) | Adaptive learned feature compression | Single family, vision |
| 3 | Zheng et al. 2025 (computation-aware offloading) | Compression ratio ↔ compute split | Single family |
| 4 | Lyu 2024; Xie 2023 (information bottleneck) | Task-relevant info under channel noise | Inside one learned architecture |
| 5 | Lin 2025; Yang 2026 (industrial semantic sensing) | Industrial semantic pipelines | Image / single pathway |
| 6 | Amiri 2022 (SIC-EDGE) | Progressive wavelet layers until sufficient | One hierarchy, ECG; **closest to C6** |
| 7 | Zhang 2023 (adaptive IoT data reduction) | Concept-drift adaptive threshold | Reconstruction-oriented, one family |
| 8 | Girgis 2026 (TS-JEPA); Dutta 2026 (CHARM) | Predictive sensor latents | A candidate (R8), not a selector |
| 9 | Qu 2026 (semantic/traditional switch + caching) | Two-mode switching | Two modes, no guarantee |
| 10 | Matsubara 2022; Shao & Zhang 2020 | Supervised/learned feature compression | Encoder design, not runtime selection |
| **11** | **Kalør & Ohtsuki 2025**, *Black-Box Edge AI Model Selection with Conformal Latency and Accuracy Guarantees* (arXiv 2506.11391) | Chooses a **feature-extractor + inference-model combination** under deadline/loss bounds via conformal risk control; image classification | They select **models** and do not choose the **payload representation** from a heterogeneous menu; we certify **agreement with the raw decision** per window and decouple certification from any optimizer; time-series IIoT |
| **12** | **Huang, Park, Paoletti, Simeone 2025**, *Reliable Inference in Edge-Cloud Model Cascades via Conformal Alignment* (arXiv 2510.17543) | Edge model returns prediction sets; conformal alignment decides when to offload to the cloud model | They decide **whether to offload / which model answers**; we decide **what data representation to send** (and how) with system-state-aware costs |

**Honest assessment.** Conformal guarantees at the edge are *not* novel; applying them to the **payload-representation choice across heterogeneous families**, with the **safe-set/any-optimizer decoupling** and **encode-cost-aware phase analysis**, is the defensible gap. Items 11–12 must be cited and compared *conceptually* (and, if feasible, via an adapted baseline B11).

### 3.4 Novelty-kill criteria (checked weekly by Role E, §19)
The C1 claim is weakened to "application of known tools" if a paper appears that (a) certifies per-window sufficiency for a **payload representation** (b) over **≥ 3 heterogeneous families** (c) in an edge-cloud offloading loop. If found: pivot to **C3 + C4 + C5** (regime theory + benchmark + empirical study), which do not depend on C1.

---

## 4. Research questions and pre-registered hypotheses

Commit this section to `docs/research/hypotheses.md` **before running any evaluation experiment**. Changes after that require an ADR with a dated rationale.

**RQ.** Can an edge runtime pick, per window, the cheapest representation from a heterogeneous menu that is *statistically certified* to preserve the cloud task decision, while meeting deadlines under changing bandwidth, queue, and compute?

| ID | Hypothesis (falsifiable) | Threshold | If false → pivot |
|---|---|---|---|
| **H1** Diversity | Over the scenario grid (§9.2) the oracle selects ≥ 3 distinct representations each in ≥ 10 % of cells; best-single (B4) costs ≥ 25 % more than the oracle at matched risk | as stated | Reframe as benchmark/regime-analysis paper (C3–C5) |
| **H2a** Certificate (expected) | Held-out mean disagreement ≤ α for α ∈ {0.02, 0.05, 0.10}; CRC variant | mean ≤ α, 95 % block-bootstrap CI upper bound ≤ α + 0.01 | Diagnose exchangeability (§6.4); fall back to per-bearing calibration |
| **H2b** Certificate (PAC) | LTT variant: in ≥ 90 % of leave-bearings-out folds, disagreement ≤ α (δ = 0.1) | ≥ 90 % folds | Report as limitation; keep H2a |
| **H3** Adaptation value | Kai reduces total cost ≥ 20 % vs B4 at the same α, and cuts SLO violations ≥ 30 % vs context-blind B5 on bursty traces | as stated | Descriptive trade-off paper |
| **H4** Encode-cost matters | Ablation "ignore encode/probe cost" changes the chosen representation in ≥ 15 % of high-load cells and raises SLO violations | as stated | Drop heavy learned reps |
| **H5** Heterogeneity pays | Heterogeneous menu beats best *within-family* adaptive controller (compression-only; learned-only) by ≥ 10 % cost at matched α | as stated | Narrow to the families that matter |
| **H6** Theory predicts reality | Closed-form break-even thresholds locate measured phase boundaries within ±20 % (bandwidth axis) | ±20 % | Report empirical map only |
| **H7** Sim-to-real | DES vs Pi+netem: p50 latency within 10 %, selection agreement ≥ 85 % | as stated | Report gap; recalibrate cost model |

Null results are publishable outcomes here, as long as the pre-registration exists.

---

## 5. System architecture

### 5.1 Layered view (A/B/C of v1, made executable)

```
                         ┌──────────────────────────── LAB PLANE (offline, torch allowed) ───────────────────────────┐
                         │ data → windows → splits → train encoders/heads/predictor → profile → CALIBRATE (CRC/LTT)  │
                         │                                   │                                                        │
                         │                                   ▼                                                        │
                         │                       ┌──────────────────────┐                                            │
                         │                       │  CALIBRATION BUNDLE  │  (hashes, λ̂(α), predictor, profiles,       │
                         │                       └──────────┬───────────┘   ONNX encoders, lockfile hash, git SHA)   │
                         └──────────────────────────────────┼────────────────────────────────────────────────────────┘
                                                            │ signed, versioned
 ┌───────────────────────────── EDGE PLANE (numpy + onnxruntime only; NO torch) ───────────────────┐        ┌───────── CLOUD PLANE ─────────┐
 │  sensor/replay ─► window ─► PROBE φ(x) ─► SUFFICIENCY PREDICTOR ĝ_r ─► SAFE SET S_λ(x)           │        │  frame decode + version check │
 │      (always)      (always)  (~cheap)         (K scores)               {raw} ∪ {r : ĝ_r ≤ λ̂}     │        │  head zoo  h_r  (per rep)     │
 │                                                                              │                    │  link  │  raw reference model f_raw    │
 │  state z_t = {bandwidtĥ, RTT̂, backlog, load ρ, energy, storage, deadline D}  ▼                   │ ─────► │  (audit + escalation replies) │
 │                                                         SELECTOR π (greedy | MPC | bandit)        │ netem/ │  telemetry sink               │
 │                                                              │ picks ONE r ∈ S_λ(x)               │  DES   └───────────────────────────────┘
 │                                                              ▼                                    │
 │                                         ENCODER (only the chosen one runs) ─► ACTION EXECUTOR     │
 │                                         {transmit | defer/cache | escalate | drop(policy-gated)}  │
 │  audit sampler (p≈1–3 %: also send raw) ─► feedback for drift-aware recalibration (ACI)          │
 └────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Two key design rules
1. **Select-then-encode.** Only the probe and predictor run for every window. Exactly one encoder runs. Their cost is added to the budget (`t_probe + t_pred + t_sel` is charged in every latency/energy figure).
2. **Certify-then-optimize.** The safe set is decided by the certificate; the cost-minimizing policy is a *separate, swappable* module that can only pick inside the safe set.

### 5.3 Degradation ladder (explicit behavior when things go wrong)
Evaluated top to bottom; the first rule that yields a feasible action wins. Mission mode (`fidelity_first` | `liveness_first`) is a config.

| Step | Condition | Action | Flag |
|---|---|---|---|
| 1 | ∃ r ∈ S_λ(x) meeting deadline D and storage/energy budgets | send `argmin cost` | `certified` |
| 2 | none, mission = `fidelity_first` | cache/defer raw in local buffer, retry when link recovers (store-and-forward) | `deferred` |
| 3 | none, mission = `liveness_first` | send cheapest **deadline-feasible** rep even if outside S_λ | `uncertified` (log ĝ_r) |
| 4 | buffer full & policy allows | drop oldest low-priority window | `dropped` |

`uncertified_rate`, `deferred_rate`, `dropped_rate` are first-class metrics. The guarantee (§6) covers only `certified` decisions; the system SLO covers the rest.

### 5.4 Contracts (what kills "version issues")
Every representation implements one Python `Protocol` (full stub in §21.3):

```
rep_id, version, contract_hash          # hash(code_version, params, quantization, dtype, endianness, ONNX sha256)
fit(train)            [lab only]
encode(x) -> bytes    [edge-safe: numpy/onnxruntime only, deterministic]
decode_for_head(b)    [cloud]
size_model()          # fixed bytes OR (mean, q95) for variable-size codecs
```

- **Frame format** (identical overhead for all representations ⇒ fair accounting):
  `magic(2) | proto_ver(1) | rep_id(1) | contract_hash(4) | window_id(4) | ts_ns(8) | len(4) | payload | crc32(4)` = **28 B** overhead.
- **Version negotiation**: the cloud rejects frames whose `contract_hash` it does not hold a head for and replies `NEEDS_BUNDLE`; the edge refuses to start with a bundle whose lockfile hash or ONNX hashes mismatch.
- **Calibration bundle** (`bundle/<id>/`): `manifest.json` (pydantic schema, SHA-256 of every file), `predictor.npz`, `lambda.json` (λ̂ per α, per variant), `profiles.parquet` (size/time per rep, per load), `encoders/*.onnx`, `env.lock` hash, git SHA, dataset+split ids, seeds. Immutable; new calibration ⇒ new bundle id.

### 5.5 Link, load and timing model
Per-window end-to-end latency for representation *i* (all terms measured by the profiler or the telemetry; none assumed):

```
L_i = t_probe + t_pred + t_sel + t_enc,i(ρ) + (Q + S_i + 28)/B + RTT/2 + t_dec,i + t_head,i
```
- `Q` backlog bytes ahead of this window; `B` current throughput estimate; `S_i` payload bytes (q95 for variable codecs).
- `t_enc,i(ρ) ≈ t⁰_enc,i / (1 − ρ)` (processor-sharing approximation, edge load `ρ ∈ [0,1)`); the profiler *validates or replaces* this with a fitted curve per representation on throttled cgroups and on the Pi.
- **Bandwidth estimator**: harmonic mean of the last *k* goodput samples (robust, standard in ABR/MPC), plus EWMA RTT.
- **Three execution backends, one codebase**:
  1. **DES** (SimPy): 10⁶+ decisions/min, used for sweeps and the phase diagram.
  2. **Docker + `tc netem/tbf`**: real sockets, real time, trace-driven (Linux; Docker Desktop on Mac/Win runs a Linux VM, so `tc` works inside containers with `NET_ADMIN`).
  3. **Raspberry Pi** edge + laptop/server cloud: validates DES (H7) and calibrates energy.
- **Traces**: public cellular throughput traces (3G/4G/5G sets are published by several groups — *[VERIFY availability/licensing in Week 1]*) rescaled to the sensor-data regime, plus a Markov-modulated bursty generator for controlled experiments. Optional Mahimahi for trace replay (Linux-only).

### 5.6 Cost model and deployment presets (no arbitrary weights)
```
cost(r | z) = p_byte·(S_r+28) + p_J·E_r + p_cpu·t_edge,r + p_late·max(0, L_r − D)
```
Prices come from **named presets** in `configs/profile/`, each grounded in a physical story, never hand-tuned per experiment:

| Preset | Story | Dominant price |
|---|---|---|
| `battery_node` | solar/battery sensor, LoRa-ish/cellular uplink | `p_J` high |
| `metered_cellular` | pay-per-MB private LTE/5G | `p_byte` high |
| `wired_gateway` | plant Ethernet, shared edge CPU | `p_cpu` high |
| `deadline_critical` | closed-loop alarm path | `p_late` dominant |

All headline results are reported as **Pareto fronts** (bytes × latency × energy × risk); scalar cost is only used by the runtime. Energy is `measured` on the Pi (USB power meter) and `modelled` elsewhere (`P = P_idle + k·util`, calibrated on the Pi); every plot carries the label.

---

## 6. Mathematics

### 6.1 Definitions (put in `docs/glossary.md`; the whole team uses these words identically)
- Window `x ∈ ℝ^N`. Cloud **reference model** `f_raw` (trained on raw windows).
- Representation `r` with encoder `e_r`, decoder/head `h_r`; decision `ŷ_r(x) = h_r(e_r(x))`. Raw `r₀` has `ŷ_{r₀}=f_raw`.
- **Decision agreement loss** `d_r(x) = 1[ŷ_r(x) ≠ f_raw(x)]` (classification). For regression use `1[|ŷ_r − f_raw| > τ]` with τ fixed a priori.
- **Label fidelity** = F1/AUROC of `ŷ_r` against ground truth (reported, but *not* what the certificate controls).
- **Probe** `φ(x)`: cheap features always computed (RMS, kurtosis, crest, 8 log-band energies, …).
- **Sufficiency predictor** `ĝ_r(φ) ≈ P(d_r = 1 | φ)`, trained on a split disjoint from calibration.
- **Safe set** `S_λ(x) = {r₀} ∪ { r : ĝ_r(φ(x)) ≤ λ }`.
- **Safe-set loss** `L(x; λ) = max_{r ∈ S_λ(x)} d_r(x) ∈ {0,1}`.

### 6.2 Proposition 1 (certify-then-optimize) — *to be proved formally by Role C, checked by Role E*
Let calibration windows `(x_i)_{i≤n}` and a test window `x_{n+1}` be exchangeable. Define
`λ̂ = sup{ λ ∈ Λ : (n/(n+1))·R̂_n(λ) + 1/(n+1) ≤ α }`, with `R̂_n(λ)=(1/n)Σ L(x_i;λ)`.
Then for **any** (possibly state-dependent, learned, or randomized) policy `π` with `π(x,z) ∈ S_{λ̂}(x)`:

`E[ d_{π(x_{n+1}, z)}(x_{n+1}) ] ≤ E[ L(x_{n+1}; λ̂) ] ≤ α.`

*Sketch.* `L(·;λ)` is non-decreasing in λ (the safe set grows) and bounded by 1, so Conformal Risk Control (Angelopoulos et al., ICLR 2024) gives the second inequality. The first holds pointwise because `d_π ≤ max over S_λ̂`. State `z` may depend on anything *except* the labels `d_r`. ∎
*Requirements*: deterministic encoders/heads (or stochasticity folded into `d`), `n ≥ ⌈1/α⌉−1` calibration windows, and a conservative handling of the exchangeability unit (§6.4).

**Variants compared in experiment E2/E5:**
| Variant | Idea | Guarantee | Trade-off |
|---|---|---|---|
| **V0** (core) | single λ, worst-case over safe set (above) | expected risk ≤ α, **any** policy | most conservative |
| **V1** | per-representation thresholds `λ_r` via **Learn-then-Test** with Bonferroni / fixed-sequence | high-probability risk ≤ α w.p. ≥ 1−δ | less conservative, needs more calibration data |
| **V2** | policy-specific CRC (calibrate loss of the *actual* chosen rep under nominal states) | valid for that policy & state distribution only | tightest, least robust |

### 6.3 What the numerical sanity check showed (done during planning, 10 Oct 2026)
A synthetic simulation (6 representations, imperfect predictor with noise σ=0.08, correlated disagreement, 300 repeated calibrations, 40 000 test windows each) gave:

| α | n_cal | mean test risk | frac. of trials with risk > α | mean cost vs raw (synthetic) |
|---|---|---|---|---|
| 0.02 | 200 / 1000 | 0.0191 / 0.0190 | 0.41 / 0.36 | 0.47 |
| 0.05 | 200 / 1000 | 0.0483 / 0.0480 | 0.40 / 0.36 | 0.43 |
| 0.10 | 200 / 1000 | 0.0984 / 0.0990 | 0.46 / 0.47 | 0.39 |

**Interpretation.** (i) The *expected*-risk guarantee holds in every setting. (ii) **A single calibration exceeds α in ~40 % of draws** — this is expected for an *in-expectation* guarantee and is exactly why V1/LTT (high-probability) exists and why H2a and H2b are separate. (iii) The cost numbers are synthetic and are **not evidence**; they only show the mechanism works. This script becomes `tests/property/test_crc_validity.py`.

### 6.4 Honest limitations of the guarantee (go in the paper's "Limitations")
1. **Exchangeability.** Windows from one bearing/run are autocorrelated; windows from *new* bearings may be shifted. The formal guarantee is for windows exchangeable with the calibration set. Mitigations: non-overlapping calibration/test windows; calibration on **held-out bearings**; leave-bearings-out folds; block-bootstrap CIs; report H2 empirically across folds, never as a theorem about new machines.
2. **Drift.** Handled by (a) a probe-feature drift monitor (two-sample test on φ) that tightens α or enters *safe mode* (raw only), and (b) **audit-in-the-loop**: with probability `p_audit ≈ 1–3 %` the edge also sends raw so the cloud can observe `d_r`; inverse-probability-weighted risk estimates feed **Adaptive Conformal Inference** (Gibbs & Candès, NeurIPS 2021) to adjust λ online. This is approximate (sparse feedback), reported as an empirical result, and its overhead is counted.
3. **Deadline misses are not disagreements.** A late or dropped frame is an SLO event, not a model error; the two are reported separately (fixes v1's "bandwidth = partial accuracy loss" conflation).
4. **Predictor quality limits savings, not validity.** A bad predictor makes the safe set tiny (everything falls back to raw) but never voids the bound.

### 6.5 Selection problem and policies
Per window: `r* = argmin_{r ∈ S_λ̂(x) ∩ Feas(z)} cost(r | z)`, `Feas(z) = {r : L_r ≤ D, E_r ≤ E_budget, S_r ≤ storage_free}`.

| Policy | Description | Phase |
|---|---|---|
| **S0** static | best fixed rep from validation (= B4) | W6 |
| **S1** greedy-safe | formula above — the core runtime | W6 |
| **S2** MPC-safe | enumerate `K^h` sequences (h=3–5, K≤9, ≤ 6 561 candidates; trivial) with persistence assumption on S_λ; accounts for queue growth | W8 |
| **S3** bandit | LinUCB / Thompson on *cost-model residuals* (online correction of latency/energy models) | W11, Could |
| **S4** RL | PPO/DQN **only as a comparison**, trained in DES; included to answer "is RL needed?" | W12, Could |
| **S\*** oracle | knows `d_r` and true future link; upper bound, never deployed | W6 |

### 6.6 Closed-form regime theory (Contribution C3)
Take representations *i* (small payload, heavier encode) and *j* (larger payload, lighter encode), `S_j > S_i`. Define
`Δt(ρ) = (t⁰_enc,i − t⁰_enc,j)/(1−ρ) + (t_dec,i + t_head,i) − (t_dec,j + t_head,j)`.
Equal latency when `(S_j − S_i)/B = Δt(ρ)` ⇒

**`B*_{ij}(ρ) = (S_j − S_i) / Δt(ρ)`** (valid when `Δt>0`): *for `B < B*` the compact representation i wins on latency; for `B > B*` the cheaper-to-encode j wins.*

Because `Δt` grows with ρ, **B\* shrinks as edge load rises** (heavy encoders pay off only on ever-slower links). Energy analogue: with per-bit radio energy `e_bit` and edge power `P_cpu`, i wins on energy iff `e_bit > P_cpu·Δt_enc / (8·ΔS)`. Add deadline feasibility (`L ≤ D`) and the safe-set mask (`ĝ_r ≤ λ̂`), and the **Phase Diagram** is the partition of the (B, D, ρ) space by the argmin representation. H6 tests that these formulas predict measured boundaries within ±20 %.

---

## 7. Representation menu (v1 of Kai-Bench)

All sizes below are **measured by the profiler, not assumed**. Dimensions are config values.

| ID | Family | Edge computation | Payload | Cloud side | Role |
|---|---|---|---|---|---|
| **R0** | Raw | none (int16 as acquired) | N·2 B | `f_raw` | reference / upper bound |
| **R1** | Lossless | delta + zstd (lvl configurable; lz4 as speed variant) | variable (q95 tracked) | decode → `f_raw` | strong non-learned baseline |
| **R2a/b** | Bounded-lossy wavelet | PyWavelets `db4` level 5, threshold to ε ∈ {1 %, 5 %} rel. L2, int8/16 quantize, zstd | variable | decode → `f_raw` | conventional lossy |
| **R3** | Engineered spectral | Hilbert envelope → envelope-spectrum band energies (log bands; bearing-defect bands as refinement) + time stats, float16 | ~64–128 B | LightGBM/MLP head | domain-feature baseline |
| **R4** | Statistical summary | RMS, peak, crest, kurtosis, skew, P2P, shape/impulse/margin factors, ZCR, spectral centroid/entropy | ~24 B | LightGBM/MLP head | cheapest informative |
| **R5** | Learned latent (task-agnostic) | 1D-conv autoencoder → latent dim 32, int8 (ONNX) | ~32 B + scale | MLP head | tests "learning pays?" |
| **R6** | Learned latent (task-aware) | supervised bottleneck trained with head (cf. supervised compression) → dim 16–32, int8 (ONNX) | ~16–32 B | MLP head | task-aware vs R5 |
| **R7** | Discrete latent | VQ-VAE code indices (e.g., 16×8 bit) | ~16 B | embed + MLP head | byte-cheapest learned |
| **R8** | *Optional* predictive latent | TS-JEPA-style encoder (ONNX) | ~32 B | MLP head | only if W11 kill-switch passes |
| **R9** | *Stretch* | quantized MiniRocket-style features | ~0.3–1 KB | ridge/logistic head | strong learning-free baseline |

**Fairness rules** (fix v1's "decoder/head incompatibility"):
1. Every head gets the **same hyper-parameter search budget** (Optuna, equal trials) on the same inner CV.
2. Head compute (`t_head`) and decode (`t_dec`) are in the latency equation for every representation.
3. Lossy-decoded representations first use the *unmodified* `f_raw`; a "decoder-aware fine-tuned" variant is an ablation, not the default.
4. Encoders are exported to **ONNX with a pinned opset** and parity-tested against torch (`max|Δ| ≤ 1e-4` float32; int8 payloads bit-exact).

---

## 8. Data, tasks, and the leakage-safe protocol

### 8.1 Datasets (decision-ready)
| Role | Dataset | Why | Caution |
|---|---|---|---|
| **Primary** | **Paderborn University bearing dataset** (KAt-DataCenter) | Many physical bearings (artificial + real damage), 4 operating conditions, high-rate vibration — enables *bearing-wise* splits | Large `.rar` downloads and MATLAB `.mat` files; check exact sizes/license on the download page *[VERIFY]*; build a decimated `processed/` subset once and share via DVC remote |
| Smoke test | CWRU bearing data | tiny, ubiquitous | **never used for reported results**; random-window accuracy is meaningless (leakage) |
| Transfer | One of XJTU-SY / HUST / MFPT, chosen in W2 by checklist: license ✓, ≥ 10 bearings ✓, raw waveform ✓, labels ✓ | tests selector over-fit to a workload | different sampling rates require re-profiling |
| Stretch | NASA C-MAPSS | RUL, different modality | cycle-level, simulated; only usable by R3/R4-type reps; **not used for core claims** |

### 8.2 Task
- **T1 (primary):** fault diagnosis, 3 classes (healthy / inner-race / outer-race) from windows of `N = 4096` samples (config), plus optional multi-label reformulation (as recommended by the 2026 realistic-evaluation study).
- **T2 (stretch):** one *different* task on the transfer dataset (e.g., anomaly score or RUL) to test task lock-in; never claimed as generality.

### 8.3 Split protocol (non-negotiable)
| Level | Rule | Reporting |
|---|---|---|
| L0 | random-window split | **forbidden** in any reported number (CI test greps for it) |
| **L1** | **leave-bearings-out**: no physical bearing appears in two of {train, predictor-train, calibration, test} | Primary results; 5 folds × 20 repeated bearing-assignments for CIs |
| **L2** | leave-operating-condition-out | Robustness / task-shift section |
| L3 | cross-dataset | Transfer experiment |

- Test and calibration windows use **hop = N** (non-overlapping); training may overlap 50 %.
- Hyper-parameters tuned **only** via inner CV on training bearings; test bearings are touched once per final run. A `splits/` manifest (bearing IDs per fold, hashed) is committed; a test asserts no ID overlap.
- Class imbalance: report macro-F1 and per-class recall; no resampling on calibration/test.

---

## 9. Experimental design

### 9.1 Baselines
| ID | Baseline | Purpose |
|---|---|---|
| B1 | Raw always | cost/fidelity reference |
| B2 | Fixed compression (R1 or R2) | simple compression |
| B3 | Static features (R3 or R4) | low-compute baseline |
| B4 | **Best-single** representation chosen on validation | does adaptation matter? |
| B5 | Context-blind adaptive (sufficiency-aware, ignores bandwidth/queue/load) | isolates system-awareness |
| B6 | **Kai (S1/S2, V0/V1/V2)** | proposed |
| B7 | Oracle | upper bound |
| **B8** | **Static profile-table selector** (pick by regime lookup, no per-window predictor) | the "obvious engineer's" baseline |
| **B9** | **Adaptive-threshold data reduction** (drift-driven, after Zhang et al. 2023) | prior-art re-implementation |
| **B10** | **Two-mode semantic/traditional switch** (conceptual re-implementation after Qu et al. 2026) | prior-art |
| **B11** | **Conformal model-selection style** (adaptation of Kalør & Ohtsuki 2025 to representation choice, if feasible) | closest prior art |
| B12 | RL selector (PPO in DES) | "is RL needed?" (Could) |

### 9.2 Scenario grid (DES full factorial; real subset on Docker/Pi)
| Axis | Levels |
|---|---|
| Bandwidth | static {64 kbps, 256 kbps, 1 Mbps, 5 Mbps, 20 Mbps}; traces {3G, 4G, 5G-rescaled, Markov-bursty} |
| Deadline D | {50, 100, 250, 1000} ms |
| Edge load ρ | {0, 0.5, 0.8} |
| Sensors per node | {1, 8, 32} (changes offered load; raw @64 kHz int16 ≈ 1 Mbps per channel) |
| Energy budget | normal / constrained |
| Storage | ample / 90 % full |
| Task regime | stable / operating-condition shift |
| α | {0.02, 0.05, 0.10} |

### 9.3 Metrics
Label F1 (macro) · **decision agreement/risk** · certified / uncertified / deferred / dropped rates · bytes per decision · latency p50/p95/p99 (end-to-end, incl. probe/enc/queue/tx/dec/head) · SLO-violation rate · edge CPU-ms & peak RSS · energy per decision (*measured* | *modelled*) · storage footprint · selector overhead · switching rate (flapping) · regret vs oracle · audit overhead.

### 9.4 Core experiments
| ID | Experiment | Tests |
|---|---|---|
| E1 | Representation profiling (size, enc time vs ρ, fidelity, agreement) | feeds everything; G1 |
| E2 | Certificate validity: V0/V1/V2 across α, folds, bearings; coverage plots | H2a, H2b |
| E3 | Main comparison B1–B11 vs Kai across grid | H1, H3, H5 |
| E4 | Phase Diagram: empirical vs closed-form B\*(ρ) | H6 |
| E5 | Ablations (below) | H4, mechanism |
| E6 | Drift & task shift: L2 splits, audit-in-loop ACI vs static λ | robustness |
| E7 | Sim-to-real: DES vs Docker/netem vs Pi | H7 |
| E8 | Selector overhead & scalability (decisions/s on Pi) | feasibility |
| E9 | *(Could)* escalation (C6) vs SIC-EDGE-style single hierarchy | C6 |
| E10 | *(Could)* privacy leakage probe (operating-condition / machine-ID attribute inference from latents) | honest privacy scoping |

### 9.5 Ablations
Remove heterogeneity (learned-only; compression-only) · remove sufficiency predictor (B8) · remove certificate (use raw ĝ threshold, no calibration) · remove compute cost · remove queue awareness · remove audit/ACI · swap V0/V1/V2 · remove learned latents · add storage/energy constraints · predictor with vs without probe features of high-cost type.

### 9.6 Statistics and reproducibility protocol
- ≥ 5 training seeds × leave-bearings-out folds × repeated bearing assignments; all runs logged, none discarded.
- Paired bootstrap (BCa) 95 % CIs over folds; **Holm–Bonferroni** across H1–H7; report effect sizes, not just p-values.
- Each figure/table has a `make` target that regenerates it from the run registry; a CI nightly job rebuilds a small "mini-paper" from a tiny fixture dataset to catch breakage.
- Determinism: fixed seeds, `torch.use_deterministic_algorithms(True)` where possible, single-thread profiling mode for timing, CPU frequency governor pinned to `performance` on the Pi, ≥ 30 warm repetitions with median + MAD for any timing.

### 9.7 Quantitative gates
| Gate | Week | Pass condition | If fail |
|---|---|---|---|
| **G0** Foundations | 1 | clean-clone setup works on Linux + WSL2 + Mac(Docker); leakage-safe splits tested; hypotheses committed | extend W1 — no exceptions |
| **G1** Diversity | 4 | ≥ 6 reps profiled; bearing-wise macro-F1 of `f_raw` well above majority baseline (threshold fixed by ADR in W2); napkin phase diagram has ≥ 2 distinct Pareto-optimal reps in ≥ 3 regimes | shrink menu / reframe as benchmark |
| **G2** MVP | 6 | closed-loop e2e (edge→netem→cloud) with S1 + V0; synthetic coverage test green | cut stretch items; fix integration |
| **G3** Signal | 8 | on dev folds: H2a holds; H1 trending; H3 ≥ 10 % (half target) | pivot to C3–C5 paper |
| **G4** Complete | 12 | all Must experiments done; novelty re-sweep logged; draft §1–§6 | scope freeze, drop Coulds |
| **G5** Freeze | 14 | clean-clone artifact reproduces all figures; internal red-team done | slip into buffer W15–16 |

---

## 10. Technology stack and compatibility ("no version issues" by construction)

### 10.1 Compatibility principles
1. **One blessed environment.** Everyone develops in the same locked environment (`uv.lock` + devcontainer). "Works on my machine" is a bug in the setup, not in the machine.
2. **Python 3.12 everywhere**, installed by `uv` (never the system Python — Raspberry Pi OS versions differ: 3.11 on Bookworm, 3.13 on Trixie). Rationale: widest binary-wheel coverage on x86_64 *and* aarch64 for the scientific stack; boring is compatible. Re-evaluate only after the project ends.
3. **The edge never imports torch.** Learned encoders run via ONNX Runtime. The edge install is 14 pinned packages (resolved below), small enough for a Pi and trivially auditable. An `import-linter` contract enforces it in CI.
4. **Everything is locked**: `uv.lock` (exact versions + hashes), Docker base images pinned by digest, ONNX opset pinned, seeds pinned, dataset hashes pinned.
5. **Heavy tools are isolated** (`uv tool install dvc`, `mlflow`) so their dependency trees can never conflict with the research environment.
6. **Upgrades are PRs**, never ad-hoc: `chore(deps)` PR → full CI → merge. Freeze upgrades after Week 12.

### 10.2 Pinned baseline (resolved 10 Oct 2026 with `uv pip compile --python-version 3.12` on four targets)
Targets that **resolved successfully**: Linux x86_64 (manylinux_2_28), Linux aarch64 (manylinux_2_28 — Pi, Docker on Apple Silicon), macOS arm64, Windows x86_64. The lockfile you generate on Day 0 is authoritative; this table is the expected result and the *reason* each package exists.

| Package | Version | Plane | Why |
|---|---|---|---|
| Python | 3.12.x (uv-managed) | all | compat |
| numpy | 2.5.3 | edge+lab | arrays (NumPy 2 semantics; avoid legacy 1.x idioms) |
| onnxruntime | 1.31.0 | edge+cloud | run ONNX encoders/heads without torch |
| PyWavelets | 1.10.0 | edge+lab | wavelet codec R2 |
| zstandard | 0.25.0 | edge+lab | lossless/lossy entropy stage R1/R2 |
| lz4 | 4.4.5 | lab (opt. edge) | fast-codec variant of R1 |
| msgpack | 1.2.3 | edge+cloud | telemetry/control messages |
| pydantic | 2.14.0 | all | contracts, bundle manifest schema |
| psutil | 7.2.2 | edge | load ρ, RSS, CPU time |
| torch (CPU build) | 2.14.1 | lab only | train encoders/heads, ONNX export |
| onnx | 1.23.2 | lab only | export/validation |
| scikit-learn | 1.9.1 | lab | predictor, metrics, CV |
| lightgbm | 4.7.0 | lab/cloud | feature heads |
| scipy | 1.18.1 | lab | stats/signal (not on edge) |
| pandas | 3.0.6 | lab | analysis tables (pandas 3 semantics: copy-on-write, string dtype — do not copy old tutorials blindly) |
| pyarrow | 26.0.0 | lab | Parquet run registry |
| statsmodels | 0.15.0 | lab | tests, bootstrap helpers |
| optuna | 5.0.0 | lab | equal-budget head search |
| hydra-core | 1.3.7 | lab | composable configs |
| simpy | 4.1.2 | sim | discrete-event simulator |
| matplotlib | 3.11.2 | lab | publication figures (plotly 7.1.0 optional for exploration) |
| typer / rich | 0.27.3 / 15.0.0 | all | CLI and logs |
| **Dev**: ruff 0.17.0 · mypy 2.4.0 · pytest 9.1.1 · hypothesis 6.168.5 · pre-commit 4.6.2 · nox 2026.8.17 | | dev | quality gates |
| **Isolated tools**: dvc 3.67.1 · mlflow 3.17.0 (optional) | | tool | data versioning / optional UI |
| *Optional cross-checks*: mapie 1.5.0 (CRC reference), codecarbon 3.3.1 (CPU energy model sanity), aeon 1.6.0 (MiniRocket, lab only) | | opt | verification |

*Caveat:* "latest" moves. Do not copy this table into `pyproject.toml` by hand — run `uv lock` on Day 0 and commit the result. Week-1 task D1 re-verifies resolution and runs the import-smoke matrix (§16).

### 10.3 Platform matrix
| Platform | Role | Notes |
|---|---|---|
| Linux x86_64 (Ubuntu 24.04) | primary dev/CI | everything works natively incl. `tc netem` |
| **Windows** | dev | **WSL2 Ubuntu 24.04 mandatory** (no native Windows dev; Docker Desktop with WSL2 backend) |
| macOS arm64 | dev | native for lab work; **Docker for netem** experiments |
| **Raspberry Pi 4B** (64-bit OS, ≥ 4 GB) | reference edge validation | uv-managed Python 3.12; edge group only; governor `performance` |
| GPU | **not required** | CPU-only torch keeps determinism and installs small; use Colab/Kaggle only for optional JEPA (R8) with exported weights committed to DVC |

### 10.4 Project-level decisions (so nobody re-debates them)
| Topic | Decision | Alternative rejected & why |
|---|---|---|
| Package/env manager | `uv` (lockfile, Python mgmt, dependency groups) | conda/poetry: slower, weaker cross-platform lock story for aarch64 |
| Config | Hydra + pydantic-validated dataclasses | ad-hoc YAML: no composition/validation |
| Tracking | Hydra output dir + DVC pipeline + Parquet run registry; MLflow *optional UI* | W&B: account/network dependency |
| Data versioning | DVC with a shared remote (Drive/university server/NAS) | git-lfs: poor pipeline semantics |
| Transport | asyncio TCP with the 28-byte frame (§5.4); TLS optional and measured separately | gRPC/HTTP: heavy deps and uneven per-rep overhead |
| Notebooks | `jupytext` `.py:percent` only; no outputs committed | `.ipynb` in git: diff noise, token waste |
| Docs | Markdown in-repo + ADRs; paper in LaTeX (or Typst) in git | Word/Docs: unreviewable diffs |
| Timing | `time.perf_counter_ns`, warm-up, median+MAD, pinned CPU governor | single-shot timings |
| Containers | 3 images (`edge`, `cloud`, `lab`), multi-arch `linux/amd64,linux/arm64`, base pinned by digest | one fat image |

---

## 11. Repository layout (final)

```
kai/
├── PRD.md                      # this file (living; changes via PR)
├── README.md  CLAUDE.md  CONTRIBUTING.md  CHANGELOG.md  LICENSE  CITATION.cff
├── pyproject.toml  uv.lock  .python-version  Makefile  noxfile.py
├── .pre-commit-config.yaml  .editorconfig  .gitignore  .gitattributes  .dvcignore
├── .devcontainer/devcontainer.json
├── .claude/                    # AI-assist config (§13)
│   ├── settings.json           # permissions.deny for data/, artifacts/, lockfile noise
│   ├── agents/                 # theory-reviewer.md  test-writer.md  lit-scout.md
│   ├── commands/               # new-rep.md  profile-rep.md  calibrate.md  run-exp.md  novelty-sweep.md
│   └── skills/                 # project skills (e.g., kai-conventions)
├── .github/
│   ├── workflows/ci.yml  nightly.yml  release.yml
│   ├── CODEOWNERS  PULL_REQUEST_TEMPLATE.md  ISSUE_TEMPLATE/
├── configs/                    # Hydra
│   ├── dataset/ (pu.yaml cwru.yaml transfer.yaml)
│   ├── representation/ (r0_raw … r9_minirocket.yaml)
│   ├── predictor/ certify/ (v0.yaml v1.yaml v2.yaml)
│   ├── selector/ (s0_static … s4_rl.yaml)
│   ├── scenario/ (grid.yaml traces/ presets/)
│   ├── profile/ (battery_node.yaml metered_cellular.yaml wired_gateway.yaml deadline_critical.yaml)
│   └── experiment/ (e1_profile.yaml … e10_privacy.yaml)
├── data/                       # gitignored, DVC-tracked
│   ├── raw/  interim/  processed/  splits/          # splits/ manifests ARE committed (small)
├── docs/
│   ├── glossary.md  architecture.md
│   ├── adr/ (0001-python-3-12.md, 0002-certify-then-optimize.md, …)
│   ├── theory/ (proposition1.md  regime-theory.md  limitations.md)
│   ├── research/ (hypotheses.md  lit-matrix.csv  novelty-log.md  baselines-repro.md)
│   └── paper/ (main.tex  refs.bib  figures/)
├── src/kai/
│   ├── contracts/              # Representation Protocol, Frame, Bundle schema (pydantic)  [mypy-strict]
│   ├── data/                   # loaders (pu, cwru, transfer), windowing, leakage-safe splitters
│   ├── reps/                   # r0_raw … r9, each: encoder, fit, size_model, tests
│   ├── tasks/                  # f_raw, heads, training, metrics
│   ├── certify/                # probe, predictor, crc, ltt, aci, drift_monitor, audit  [mypy-strict, ≥85 % cov]
│   ├── select/                 # base, static, greedy_safe, mpc, bandit, rl, oracle, baselines/  [mypy-strict]
│   ├── edge/                   # runtime loop, encoder bank (ONNX), executor, telemetry, energy   [NO torch]
│   ├── cloud/                  # server, head zoo, reference model, audit/escalation handlers
│   ├── link/                   # framing, DES (simpy), netem scripts, trace loaders
│   ├── profile/                # profiler harness (size, time-vs-load, energy)
│   ├── bench/                  # experiment runner, registry, scenario grid
│   └── analysis/               # stats, bootstrap, plots, phase_diagram, tables
├── tests/
│   ├── unit/  property/  integration/  e2e/  golden/  fixtures/ (tiny synthetic dataset)
├── docker/ (edge.Dockerfile cloud.Dockerfile lab.Dockerfile compose.yaml netem/)
├── scripts/ (bootstrap.sh  download_data.py  build_bundle.py  pi_setup.sh  verify_env.py)
├── notebooks/                  # jupytext .py only
├── experiments/                # run manifests (small JSON/Parquet committed; big stuff in DVC)
└── artifacts/                  # bundles, models, ONNX (DVC)
```

**Import rules (enforced by `import-linter`):** `contracts` imports nothing from `kai`; `edge` may import only `contracts`, `reps` (edge-safe modules), `certify` (runtime parts), `select`, `link`; **nothing under `edge` may import torch, sklearn, pandas, scipy, lightgbm**; `analysis` and `bench` may import anything.

---

## 12. Code quality system

### 12.1 Tooling and gates
| Area | Tool | Gate |
|---|---|---|
| Lint + format | **ruff** (rules: E,F,I,UP,B,SIM,C4,PL,RUF,NPY,PERF,PT; line length 100) | pre-commit + CI, zero warnings |
| Types | **mypy --strict** on `contracts/ certify/ select/`; standard mode elsewhere | CI |
| Tests | **pytest** + **hypothesis** (property) + `pytest-cov`, `pytest-xdist` | coverage ≥ 85 % on `certify/ select/`, ≥ 70 % overall |
| Architecture | **import-linter** (edge isolation), **deptry** (unused/missing deps) | CI |
| Security/supply chain | `pip-audit`, `uv lock --check`, `gitleaks` pre-commit | CI |
| Hooks | **pre-commit** (ruff, mypy-changed, nbstripout/jupytext check, gitleaks, check-added-large-files 500 kB) | local + CI |
| Task runner | **nox** sessions (`lint`, `types`, `tests`, `edge_smoke`, `docs`) + `make` shortcuts | one command each |
| Commits | Conventional Commits (`feat:`, `fix:`, `exp:`, `docs:`, `chore:`), squash-merge | PR title check |
| Profiling | `py-spy`, `pyinstrument`, `memray` (Linux) | used in E8 |

### 12.2 Mandatory property/golden tests (these *are* the correctness spec)
1. **CRC validity**: promoted from the §6.3 simulation; asserts mean risk ≤ α + tolerance over repeats.
2. **Safe-set monotonicity**: `S_λ ⊆ S_λ'` for `λ ≤ λ'`; `L(x;λ)` non-decreasing.
3. **Policy containment**: for any selector, a `certified` action ∈ safe set (fuzzed with hypothesis).
4. **Codec laws**: lossless round-trip identity; lossy error ≤ ε; payload size ≤ declared q95 in ≥ 95 % windows.
5. **Frame**: encode/decode round-trip, CRC corruption detected, contract-hash mismatch rejected.
6. **Splits**: no bearing ID in two partitions; test windows non-overlapping; a repo-wide check bans L0 splits in reported experiments.
7. **ONNX parity**: torch vs onnxruntime within tolerance; int8 payloads bit-exact.
8. **DES conservation**: bytes offered = bytes delivered + backlog + dropped; Little's law sanity.
9. **Determinism**: same seed + config ⇒ identical run registry hash.
10. **Golden figures**: a tiny-fixture run reproduces reference numbers within tolerance (guards silent regressions).

### 12.3 Definition of Done (every PR)
Code + tests + types · docstring on public API · config + docs updated · no new warnings · registry/schema migration if contracts changed · ADR if a decision changed · reviewed by a CODEOWNER **other than the author** · CI green · PR ≤ ~400 changed lines (split otherwise).

### 12.4 Git workflow
Trunk-based: protected `main`, short-lived branches `role/topic` (e.g., `C/ltt-variant`), required checks, 1 approval (2 for `contracts/` and `certify/`), weekly tagged milestone (`w06-mvp`). Experiments run from a **clean tree at a tagged commit**; the run registry stores git SHA + lock hash, and a dirty tree aborts a "final" run.

---

## 13. AI-assisted development and token efficiency

**Principle: measure first, optimize native settings second, add tools third.** Reported savings from community tools vary from ~0 % to ~98 % depending on workload and are mostly vendor/blog benchmarks; adopt a tool only if it wins an A/B on *your* repo.

### 13.1 Layer 0 — native hygiene (do all of this before installing anything)
- **`CLAUDE.md` ≤ ~150 lines**: commands (`uv sync`, `nox -s tests`), architecture map (§5/§11 condensed), conventions, "don't touch" paths, glossary pointer. Stable guidance only; workflows go into skills/commands.
- **`.claude/settings.json` deny rules** for heavy/noisy paths: `data/**`, `artifacts/**`, `*.ipynb`, `uv.lock`, `docs/paper/figures/**`, `experiments/**/*.parquet`. (There is no `.claudeignore`; use permission deny rules plus `.gitignore` hygiene.)
- **Session discipline**: `/clear` between unrelated tasks; `/compact` with explicit instructions ("keep: current module, failing test, decisions"); check `/cost` regularly; Plan mode for anything touching `contracts/` or `certify/`.
- **Model routing**: a mid-tier model for routine edits/tests; the strongest model for theory review, architecture, and debugging hard failures.
- **Subagents** for noisy work (literature scouting, log triage, full-test-suite analysis) so the main context stays clean.
- **MCP discipline**: only servers you use; each connected server's tool schemas cost context every turn. Prefer plain CLI tools where they suffice.
- **Task template for every AI prompt**: `Goal · Files in scope · Acceptance tests (must pass) · Do-not-touch · Output format`. Small, test-anchored tasks beat long conversations.

### 13.2 Layer 1 — tools (adopt via A/B; owner: Role E)
| Tool | What it does | Verdict for Kai | Caveat |
|---|---|---|---|
| **RTK** (rtk-ai) | Rust CLI proxy that compresses noisy shell output (pytest, git, docker logs) before it hits context; rewrites only Bash tool calls | **Adopt after Layer 0** — our workflow is terminal-heavy (pytest, docker, netem logs) | Doesn't touch built-in Read/Grep/Glob; needs hook governance |
| **ccusage** (CLI) *[VERIFY name/install]* | Reads local Claude Code logs → per-session/day token and cost report | **Adopt** — the measurement backbone for the A/B | — |
| **context-mode** | MCP that keeps large tool outputs in a searchable local store | **Candidate** (choose *one* of this / token-savior) | Elastic License v2 — fine for an academic project, read it |
| **token-savior** | MCP for symbol-level lookup instead of whole-file reads | **Candidate** (A/B vs context-mode) | — |
| **Serena** (oraios) | MCP using language servers for semantic, symbol-level code navigation/edit | **Candidate** for `src/kai` once the codebase > ~50 files | Needs LSP setup; measure |
| claude-mem | Persistent cross-session memory | Optional | Can inject stale context |
| caveman-style prose compressors | Shorten assistant prose | **Skip for code/docs work** | Can hurt clarity in reviews |
| token-optimizer (alexgreensh) | Hooks for delta re-reads | **Avoid** | PolyForm Noncommercial licence |
| claude-token-optimizer (nadimtuhin) | CLAUDE.md scaffolder | **Avoid** | Unverified claims |

**A/B protocol (Week 2, ~2 hours, Role E):** three fixed tasks (e.g., "add representation R4 with tests", "debug failing DES conservation test", "write the profiler report script"); run each baseline vs +tool; record tokens (ccusage), wall time, tests passing, human-rated diff quality. Adopt iff **≥ 15 % token reduction with no quality loss**; record in `docs/adr/00xx-ai-tooling.md`.

### 13.3 Project commands, agents, skills (create in Week 1–2)
- Commands: `/new-rep <id>` (scaffolds encoder, config, contract test, size-model test), `/profile-rep`, `/calibrate`, `/run-exp <id>`, `/novelty-sweep`.
- Agents: `theory-reviewer` (read-only; checks Proposition 1 steps and every statistical claim), `test-writer` (hypothesis-first), `lit-scout` (arXiv/Scholar sweep → appends `lit-matrix.csv`, `novelty-log.md`).
- Skill `kai-conventions`: glossary, naming, units (always bytes, ms, J), RNG discipline, how to add a metric.

### 13.4 Research-quality tooling
Zotero + Better BibTeX (shared group library → `refs.bib`) · arXiv RSS/email alerts for `cs.NI`, `cs.LG`, `eess.SP` with keyword filters · Semantic Scholar / Google Scholar alerts on the novelty-kill phrases (§3.4, §19) · `lit-matrix.csv` (paper × {A,B,C layer, families, task, guarantee?, code?}) · pre-registered `hypotheses.md` · ADRs for every decision · LaTeX with `latexmk` in CI, `make figures` regenerating every plot.

---

## 14. Team of five: roles, ownership, interfaces

Assign names to letters in the Week-1 kickoff. Every role has a **primary owner and a named backup** (bus factor ≥ 2) and owns directories via `CODEOWNERS`.

| Role | Title | Owns (dirs) | Core responsibilities | Backup |
|---|---|---|---|---|
| **A** | **Data & Classical Representations** | `data/ reps/(r0–r4,r9) configs/dataset configs/representation` | Dataset ingestion, windowing, **leakage-safe splitters**, codecs (R0–R2), engineered/summary features (R3–R4, numpy-only Hilbert via FFT), codec-law tests, transfer dataset | B |
| **B** | **Learning (cloud models + learned reps)** | `tasks/ reps/(r5–r8) cloud/heads` | `f_raw`, per-rep heads with **equal search budgets**, AE / supervised bottleneck / VQ / (JEPA), ONNX export + parity tests, builds the **Disagreement Table** with A | C |
| **C** | **Certification & Decision (theory lead)** | `certify/ select/ docs/theory` | Probe, sufficiency predictor, CRC/LTT/ACI, audit sampler, drift monitor, selectors S0–S4, oracle, **Proposition 1 proof**, closed-form regime theory | E |
| **D** | **Systems & Emulation** | `edge/ cloud/server link/ profile/ docker/ scripts/` | Edge runtime, 28-byte frame protocol, asyncio transport, DES, netem + trace replay, profiler, **Raspberry Pi + power measurement**, containers | A |
| **E** | **Evaluation, Reproducibility & Paper** | `bench/ analysis/ docs/research docs/paper .github/ .claude/` | Experiment runner + registry, baselines B1–B12, statistics, figures, CI/pre-commit/tooling, AI-assist A/B, **literature + novelty log**, paper lead | D |

### 14.1 Interfaces between roles (frozen in Week 1–3 as pydantic schemas in `contracts/`)
| Artifact | Producer → Consumer | Schema (key columns) |
|---|---|---|
| `WindowSet` + `splits/` manifest | A → all | window_id, bearing_id, condition, class, fold, partition ∈ {train, ptrain, calib, test} |
| **Disagreement Table** | B (+A) → C, E | window_id × rep_id → `d_r`, correct, logits-hash, payload_bytes |
| `profiles.parquet` | D → C, E | rep_id, load ρ, `t_enc` median/MAD/q95, bytes mean/q95, energy (measured/modelled flag) |
| **Calibration Bundle** | C (+B encoders) → D | §5.4 |
| Run registry | all → E | run_id, git_sha, lock_hash, config_hash, seed, metrics |

### 14.2 RACI (R = does, A = accountable, C = consulted)
| Activity | A | B | C | D | E |
|---|---|---|---|---|---|
| Data loaders / splits | **R/A** | C | C | | C |
| Representations R0–R4 | **R/A** | C | | C | |
| Learned reps R5–R8, heads | C | **R/A** | C | | |
| Predictor + calibration (CRC/LTT/ACI) | | C | **R/A** | | C |
| Selectors, oracle | | | **R/A** | C | C |
| Edge/cloud runtime, DES, netem, Pi | C | | C | **R/A** | |
| Experiments, stats, figures | C | C | C | C | **R/A** |
| Proposition 1 proof | | | **R** | | **A** (independent check) |
| CI / tooling / env | C | | | **R** | **A** |
| Literature & novelty log | C | C | C | | **R/A** |
| Paper | C | C | **R** (§method) | **R** (§system) | **A** |

### 14.3 Rituals (lightweight, mandatory)
- **Mon 30 min** planning: the week's deliverable per role, blockers, risks.
- **Wed async** (3 lines each): done / doing / blocked.
- **Fri 45 min** *integration demo*: something runs end-to-end or the week is "red". Demos are on the tagged `main`.
- **Red-team hour** (biweekly, rotating): one member attacks another's component with adversarial tests/questions (e.g., D attacks the certificate, C attacks the latency model).
- **Decision log**: any irreversible choice → ADR (≤ 1 page) the same day.
- **Weekly novelty sweep** (E, 30 min) → `novelty-log.md`.
- **Cross-review rule**: nobody merges their own change in `contracts/` or `certify/`.

---

## 15. Plan: 16 weeks, step by step

*Assumes kick-off Mon 12 Oct 2026. Re-map to your academic calendar (exams/holidays) in Week 1; use W15–W16 as buffer first, then MoSCoW (§15.3).*

### 15.1 Weekly schedule
| Wk | Dates (Mon–Fri) | A — Data/Classical | B — Learning | C — Certify/Decide | D — Systems | E — Eval/Paper | Gate |
|---|---|---|---|---|---|---|---|
| **1** | 12–16 Oct | repo skeleton, PU subset downloaded→`processed/`, loaders, windowing, **leakage-safe splitter + tests** | torch CPU env, `f_raw` skeleton on tiny fixture | `glossary.md`, Prop 1 draft, §6.3 sim → property test | devcontainer, 3 Dockerfiles, `uv.lock`, **env-resolution matrix**, CI skeleton | `hypotheses.md` committed, lit-matrix, novelty-log, CLAUDE.md, pre-commit, token baseline (`/cost`, ccusage) | **G0** |
| **2** | 19–23 Oct | R0–R2 + codec-law tests; pick transfer dataset by checklist | `f_raw` (ResNet1D-small) bearing-wise baseline; ADR for F1 threshold | probe φ(x) + predictor skeleton | frame protocol + asyncio loopback + profiler harness | AI-tool A/B; bootstrap/stat lib; registry schema | |
| **3** | 26–30 Oct | R3–R4 (numpy-only), size models | heads R0–R4 with equal Optuna budget | CRC + LTT implementations, **Disagreement-Table builder** with B | DES v0 (link, queue, encode-cost model); netem scripts | experiment runner; baselines B1–B4 | |
| **4** | 2–6 Nov | profiling runs R0–R4 across loads | Disagreement Table v1 complete | **napkin Phase Diagram** + closed-form B\* | profiler under cgroup throttling | G1 report (E1) | **G1** |
| **5** | 9–13 Nov | transfer-dataset loaders | R5 (AE) + R6 (supervised bottleneck) training | predictor ĝ trained; **V0 calibration**; coverage tests on real table | edge runtime loop in Docker (probe→select→encode→send) | E2 harness (coverage plots) | |
| **6** | 16–20 Nov | R7 (VQ) integration | ONNX export + parity tests R5–R7 | **S1 greedy-safe**, S0, oracle | **closed loop e2e** with netem + bundle loading | B5, B8 baselines; first E3 | **G2** |
| **7** | 23–27 Nov | data QA; R9 (Could) | heads tuned; bundle integration | **V1 (LTT)**, V2 | **Pi setup**, edge-only install check, power-meter calibration | traces ingest + scenario-grid generator | |
| **8** | 30 Nov–4 Dec | L2 (operating-condition) splits | decoder-aware fine-tune ablation | **S2 MPC** | DES-vs-netem validation (small grid) | E3 prelim on dev folds; G3 report | **G3** |
| **9** | 7–11 Dec | transfer dataset pipeline | retrain on transfer | drift monitor, **audit sampler, ACI** | Pi runs for E7 (pass 1) | B9, B10 re-implementations | |
| **10** | 14–18 Dec | support / data QA | **R8 JEPA go/no-go decision starts (time-box 1 wk)** | E2 final (V0/V1/V2) | E8 overhead & scalability on Pi | B11 attempt; E4 phase-diagram code | |
| **11** | 21–25 Dec | E5 ablation runs | R8 decision **kill-switch at end of week** | S3 bandit (Could) | E7 complete | E6 drift runs | |
| **12** | 28 Dec–1 Jan | final runs from tagged commit | final runs | final runs; theory text | final runs | **novelty re-sweep #2**; draft §1–§6 | **G4** |
| **13** | 4–8 Jan | figures review | | | artifact packaging | paper draft complete; Could: E9/E10 | |
| **14** | 11–15 Jan | internal review & fixes | | | | **clean-clone reproduction**; red-team round 2 | **G5** |
| **15** | 18–22 Jan | buffer / camera-ready | | | | repo release prep (licence, README, archive DOI) | |
| **16** | 25–29 Jan | submission / presentation | | | | demo video; retrospective | |

### 15.2 Step-by-step build order (what to build first, and why)
1. **Make the foundation boring and reproducible** (W1): env, CI, schemas, splits. Nothing scientific is trusted until the split test passes.
2. **Measure before modelling** (W2–W4): profile every representation (bytes, time vs load, agreement). The napkin Phase Diagram from *measured* numbers tells you early whether the project has signal (G1).
3. **Certify** (W3–W6): Disagreement Table → predictor → CRC. Validate on the synthetic simulation first, then real data.
4. **Optimize** (W6–W8): greedy-safe, then MPC. Oracle and baselines exist from the start so every result is comparative.
5. **Close the loop** (W5–W6): real sockets + netem as soon as the first selector exists; do not leave integration to the end.
6. **Harden** (W7–W11): Pi, drift, audit, transfer, ablations.
7. **Freeze and write** (W12–W16): final runs from a tagged commit; paper; clean-clone reproduction.

### 15.3 MoSCoW scope
| Priority | Items |
|---|---|
| **Must** | R0–R4, one learned latent (R5 or R6), `f_raw` + heads, leakage-safe L1 splits, Disagreement Table, predictor + **V0 CRC**, S1 + oracle, DES, Docker+netem e2e, B1–B5/B7/B8, E1–E4, E7 (DES vs Docker), reproducible paper artifacts |
| **Should** | R6 *and* R5, R7, V1 (LTT), S2 (MPC), Pi validation + measured energy, B9/B10, audit + ACI, transfer dataset, E5, E6, E8 |
| **Could** | R8 (JEPA), R9, S3/S4/B12, B11, C6 escalation (E9), privacy probe (E10), T2 task |
| **Won't (this project)** | images/video, multi-modal, custom radio/protocol design, privacy claims, new encoder architectures as a contribution |

If the Pi is unavailable: replace with Docker cgroup throttling + *modelled* energy; H7 becomes DES-vs-Docker only, labelled as such.

---

## 16. Day-0 setup, installs and pre-downloads (checklist)

### 16.1 Accounts and shared infrastructure
- [ ] GitHub organization/private repo `kai`; branch protection; CODEOWNERS
- [ ] DVC remote (shared drive / university server / NAS), with ≥ 100 GB for raw data on one machine; others pull only `processed/`
- [ ] Zotero group library; arXiv/Scholar alerts (§13.4)
- [ ] Overleaf (optional) or LaTeX-in-git

### 16.2 Software per machine (Linux / WSL2 Ubuntu 24.04 shown)
```bash
sudo apt update && sudo apt install -y git build-essential curl iproute2 unar p7zip-full
curl -LsSf https://astral.sh/uv/install.sh | sh          # then restart shell
uv python install 3.12
git clone <repo-url> kai && cd kai
uv sync --extra lab --all-groups --frozen                # exact lockfile versions
uv tool install pre-commit && pre-commit install
uv tool install dvc                                      # add the extra for your remote type, e.g. dvc[ssh]
uv run python scripts/verify_env.py                      # must print ALL GREEN
```
Docker (Engine or Desktop with WSL2 backend) then, for the Pi image:
`docker buildx create --use` and `docker run --privileged --rm tonistiiv/binfmt --install arm64`.

### 16.3 Raspberry Pi (Role D; one device is enough)
**Reference edge = Raspberry Pi 4B (≥ 4 GB, Cortex-A72, 5 V/3 A PSU).** Heatsink + fan is mandatory (the Pi 4 throttles around 80 °C and ruins timing; log `vcgencmd get_throttled` every run and discard throttled runs) · official PSU · decent SD card (or USB SSD) · Ethernet cable (not Wi-Fi, for stable timing) · **inline USB-C power meter rated ≥ 3 A**. Pi 4 caveats: no dot-product int8 instructions, so int8 ONNX encoders may not be faster than float32 — profile both and keep whichever is faster; keep the 64-bit OS image.
```bash
# on the Pi (64-bit OS)
sudo apt update && sudo apt install -y git curl iproute2
curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12
git clone <repo-url> kai && cd kai
uv sync --extra edge --no-dev --frozen                   # ~14 packages, no torch
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 16.4 Downloads (Week 1; Role A unless noted)
| Item | Notes |
|---|---|
| Paderborn bearing data | from the KAt-DataCenter download page; large `.rar`; extract with `unar`; write `processed/` (npz/Parquet) once, hash it, publish via DVC. *[VERIFY size/licence/terms]* |
| CWRU data | smoke tests only |
| Transfer dataset | decided in W2 using the checklist in §8.1 |
| Network traces (Role D/E) | public 3G/4G/5G throughput traces; record source + licence in `configs/scenario/traces/README.md` *[VERIFY]* |
| Pretrained models | **none required.** (JEPA R8 is optional and trained locally/Colab; no external weights assumed) |
| Container base images | pull and **pin by digest** in W1 |

### 16.5 `scripts/verify_env.py` must check
Python 3.12 · versions equal `uv.lock` · `import numpy, onnxruntime, pywt, zstandard, psutil, msgpack, pydantic` (edge set) · lab imports on lab machines · torch→ONNX→onnxruntime tiny parity · `torch` **not importable** inside the edge image · CPU info/governor · free disk · `tc qdisc add dev lo root netem delay 1ms` succeeds in the netem container (else prints how to grant `NET_ADMIN`) · deterministic seed test.

### 16.6 Open decisions to settle in the Week-1 kickoff
1. Real deadline and target venue (changes the §15 calendar). 2. Is a Raspberry Pi available (§15.3 fallback)? 3. Where do raw data and DVC live? 4. Transfer-dataset choice (W2). 5. Role assignments and backups. 6. Confirm the project name (Kai) and check that the GitHub org/repo name `kai` is available (the PyPI name `kai` is likely taken, which is fine because the package is never published to PyPI).

---

## 17. Risk register

| # | Risk | L | I | Early signal | Mitigation | Owner |
|---|---|---|---|---|---|---|
| 1 | Predictor too weak ⇒ safe set ≈ {raw} ⇒ no savings | M | H | G1/G3: savings < 10 % | richer probe features (budgeted), per-class predictors, V1 thresholds; the C3–C5 contributions still stand | C |
| 2 | Certificate fails on **new bearings** (exchangeability) | M | H | E2 folds exceed α | calibrate on held-out bearings, drift monitor + safe mode, report honestly (H2 pivot) | C |
| 3 | Learned latents never beat features | M | L | E1 | acceptable scientific finding (supports "simple is sufficient" claim); drop R8 | B |
| 4 | Dataset download/format trouble (PU) | M | M | W1 slips | start W1; fall back to alternative bearing dataset with many bearings | A |
| 5 | Timing noise (shared CPUs, throttling) | H | M | high MAD | pin governor, single-thread profiling mode, ≥ 30 reps, median+MAD, Pi with cooling | D |
| 6 | DES ≠ reality (sim-to-real gap) | M | M | E7 | calibrate cost model on Pi; report gap; DES only used for sweeps | D |
| 7 | Integration debt at the end | H | H | red Fridays | closed loop by W6; weekly integration demos | E |
| 8 | Key person unavailable | M | H | — | backups (§14), documented interfaces, ADRs | all |
| 9 | Competing paper appears (novelty) | M | H | weekly sweep | kill-criteria (§3.4) ⇒ pivot to C3–C5 | E |
| 10 | Dependency break after upgrade | L | M | CI | lockfile, upgrades only via PR, freeze after W12 | D |
| 11 | Over-fitting to one dataset | M | M | transfer drop | transfer dataset + L2 splits | A |
| 12 | JEPA time sink | M | M | W10–11 | 1-week time-box + hard kill-switch | B |
| 13 | Energy numbers questioned | M | M | review | measured on Pi, labelled *modelled* elsewhere, sensitivity analysis | D |
| 14 | Scope creep | H | H | PR sizes ↑ | MoSCoW, gates, Won't list | E |
| 15 | Reproducibility rot | M | H | nightly red | nightly mini-paper job, clean-clone test in W14 | E |

---

## 18. Paper plan

**Working title:** *Certify, Then Optimize: Sufficiency-Certified Representation Selection for Edge–Cloud Industrial Monitoring*

**Outline:** 1 Intro (ABR-for-machines framing; contributions C1–C5) · 2 Related work (use §3.3 table) · 3 Problem & definitions (§6.1) · 4 Kai: safe sets, certificate, Proposition 1, variants V0–V2, audit/ACI · 5 System (select-then-encode, contracts, degradation ladder) · 6 Regime theory (B\*(ρ)) · 7 Evaluation (leakage-safe protocol; E1–E8) · 8 Limitations (§6.4) · 9 Conclusion · Appendix: proofs, profiles, hyper-parameters, full ablations.

**Core figures:** F1 architecture · F2 Pareto fronts per representation · F3 **Phase Diagram (empirical vs theory)** · F4 certificate coverage across α/folds/variants · F5 selection over time on a bursty trace · F6 cost vs B1–B11 · F7 ablation bars · F8 sim-to-real scatter · F9 selector overhead on Pi · F10 break-even (learned-latent pays when…).

**Venue approach:** post an arXiv preprint at G4/G5; target an edge/IoT/networked-ML venue or workshop whose deadline fits (decide at Week 1, §16.6). Release code + bundles + tiny-fixture reproduction with an archival DOI. Submit with the pre-registration visible.

---

## 19. Literature and novelty protocol

- **Weekly sweep (Role E):** queries — "representation selection" edge offloading · "task-oriented" compression conformal · "conformal" edge inference payload · "heterogeneous representation" semantic offloading · "sufficiency" certificate sensor data · "adaptive bitrate" machine perception · plus the dataset name. Sources: arXiv (cs.NI, cs.LG, eess.SP, stat.ML), Semantic Scholar, IEEE Xplore, ACM DL, Google Scholar. Track *cited-by* of Kalør & Ohtsuki, Huang et al., Qu et al., Yang et al., Girgis et al.
- **Log format** (`docs/research/novelty-log.md`): date · query · new hits · overlap with C1–C6 (none/partial/high) · action.
- **Hard checkpoints:** W12 re-sweep #2, T−14 days and T−2 days before submission. A "high overlap" hit triggers an ADR and a claim-narrowing PR the same week.
- **Reading order** (from v1, extended): survey [1] → [2],[3] → [13],[4] → [6],[7],[8],[5],[20] → [11],[12] → **Kalør & Ohtsuki; Huang et al.; Angelopoulos et al. (CRC); LTT; ACI** → [15],[16],[17] (JEPA, last) → ABR/MPC classics (control inspiration).

---

## 20. References and sources

**Verified during this planning pass (10 Oct 2026):**
- A. E. Kalør, T. Ohtsuki, *Black-Box Edge AI Model Selection with Conformal Latency and Accuracy Guarantees*, arXiv:2506.11391 (2025). https://arxiv.org/abs/2506.11391
- J. Huang, S. Park, N. Paoletti, O. Simeone, *Reliable Inference in Edge-Cloud Model Cascades via Conformal Alignment*, arXiv:2510.17543 (2025). https://arxiv.org/abs/2510.17543
- *Towards a more realistic evaluation of machine learning models for bearing fault diagnosis*, arXiv:2509.22267; DOI 10.1016/j.ymssp.2026.114640. https://arxiv.org/abs/2509.22267
- A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, *Conformal Risk Control*, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html
- Paderborn bearing DataCenter (download page): https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
- Token-reduction tool survey: https://computingforgeeks.com/reduce-claude-code-token-usage-tools/ · native-first guidance: https://radar.firstaimovers.com/rtk-vs-native-claude-code-optimization-what-to-fix-before-adding-another-hook · Serena overview: https://mcp.directory/blog/serena-mcp-complete-guide-2026

**Carried over from v1:** references [1]–[20] (see the v1 review document, §16).

**Added from general knowledge — verify bibliographic details before citing:** Angelopoulos, Bates, Candès, Jordan, Lei, *Learn then Test* (2021, arXiv:2110.01052) · Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS 2021) · Lessmeier et al., Paderborn bearing data (PHME 2016) · Yin et al., MPC-based adaptive bitrate (SIGCOMM 2015) · Mao et al., Pensieve (SIGCOMM 2017) · Netravali et al., Mahimahi (USENIX ATC 2015) · Dempster et al., MiniRocket (KDD 2021).

---

## 21. Starter files (copy-paste; adjust after `uv lock`)

### 21.1 `pyproject.toml`
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "kai"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
# core = edge-safe only (no torch / sklearn / pandas / scipy)
dependencies = [
  "numpy>=2.5,<3", "pydantic>=2.14,<3", "msgpack>=1.2,<2", "zstandard>=0.25,<1",
  "pywavelets>=1.10,<2", "psutil>=7.2,<8", "typer>=0.27", "rich>=15",
]

[project.optional-dependencies]
edge  = ["onnxruntime>=1.31,<2"]
cloud = ["onnxruntime>=1.31,<2", "lightgbm>=4.7,<5", "scikit-learn>=1.9,<2"]
lab   = [
  "torch>=2.14,<2.15", "onnx>=1.23,<2", "onnxruntime>=1.31,<2", "scikit-learn>=1.9,<2",
  "lightgbm>=4.7,<5", "scipy>=1.18,<2", "pandas>=3.0,<4", "pyarrow>=26", "statsmodels>=0.15",
  "optuna>=5,<6", "hydra-core>=1.3,<2", "simpy>=4.1,<5", "matplotlib>=3.11,<4", "lz4>=4.4",
]

[dependency-groups]
dev = ["pytest>=9", "pytest-cov", "pytest-xdist", "hypothesis>=6.168", "ruff>=0.17", "mypy>=2.4",
       "import-linter", "deptry", "pip-audit", "nox", "pre-commit>=4.6", "jupytext"]

[tool.uv]
default-groups = ["dev"]

# CPU-only torch (avoids multi-GB CUDA wheels) — VERIFY with `uv lock` in Week 1
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]

[tool.hatch.build.targets.wheel]
packages = ["src/kai"]

[tool.ruff]
line-length = 100
target-version = "py312"
[tool.ruff.lint]
select = ["E","F","I","UP","B","SIM","C4","PL","RUF","NPY","PERF","PT"]
ignore = ["PLR0913"]

[tool.mypy]
python_version = "3.12"
warn_unused_ignores = true
[[tool.mypy.overrides]]
module = ["kai.contracts.*","kai.certify.*","kai.select.*"]
strict = true

[tool.pytest.ini_options]
addopts = "-ra --strict-markers --cov=kai --cov-report=term-missing -n auto"
markers = ["slow: nightly only", "pi: needs Raspberry Pi", "netem: needs NET_ADMIN"]

[tool.importlinter]
root_package = "kai"
include_external_packages = true
[[tool.importlinter.contracts]]
name = "edge/contracts must stay light"
type = "forbidden"
source_modules = ["kai.edge", "kai.contracts"]
forbidden_modules = ["torch","sklearn","pandas","scipy","lightgbm"]
```

### 21.2 `CLAUDE.md` (keep ≤ 150 lines)
```markdown
# Kai — agent guide
Goal: sufficiency-certified representation selection for edge-cloud sensor offloading. Read PRD.md §5–§6 only when needed.
## Commands
- setup: `uv sync --extra lab --all-groups --frozen` · lint: `nox -s lint` · types: `nox -s types` · tests: `nox -s tests`
- one test: `uv run pytest tests/unit/test_x.py -q -p no:cacheprovider`
## Rules
- Units: bytes, ms, J. Never mix. RNG: always pass `np.random.Generator`; no global seeds.
- `kai.edge` and `kai.contracts` must NOT import torch/sklearn/pandas/scipy/lightgbm.
- Splits: leave-bearings-out only. Never random-window splits in reported results.
- Certificate language: "decision agreement with raw model" ≠ "label accuracy".
- Changing `contracts/` or `certify/` needs a test and an ADR note.
## Don't read/edit
data/, artifacts/, uv.lock, *.ipynb, experiments/**/*.parquet
## Task format
Goal · Files in scope · Acceptance tests · Do-not-touch · Output format
```

### 21.3 `src/kai/contracts/representation.py` (stub)
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Protocol
import numpy as np


@dataclass(frozen=True)
class SizeModel:
    mean_bytes: float
    q95_bytes: float  # == mean for fixed-size codecs


class Representation(Protocol):
    rep_id: ClassVar[str]  # "r4_summary"
    version: ClassVar[str]  # bump on ANY change affecting payload semantics

    def contract_hash(self) -> str: ...  # hash(version, params, dtype, quantization, onnx sha256)
    def size_model(self) -> SizeModel: ...
    def encode(
        self, x: np.ndarray
    ) -> bytes: ...  # edge-safe, deterministic, numpy/onnxruntime only
    def decode_for_head(self, payload: bytes) -> np.ndarray: ...  # cloud side
```

### 21.4 `.claude/settings.json`
```json
{
  "permissions": {
    "deny": [
      "Read(./data/**)", "Read(./artifacts/**)", "Read(./uv.lock)",
      "Read(./**/*.ipynb)", "Read(./experiments/**/*.parquet)", "Read(./.env)"
    ]
  }
}
```

### 21.5 `.pre-commit-config.yaml` (pin `rev`s after `pre-commit autoupdate` in W1)
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: <pin>
    hooks: [{id: ruff, args: [--fix]}, {id: ruff-format}]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: <pin>
    hooks: [{id: check-added-large-files, args: [--maxkb=500]}, {id: check-yaml}, {id: end-of-file-fixer}]
  - repo: https://github.com/gitleaks/gitleaks
    rev: <pin>
    hooks: [{id: gitleaks}]
  - repo: local
    hooks:
      - {id: mypy, name: mypy (strict pkgs), entry: uv run mypy src/kai/contracts src/kai/certify src/kai/select, language: system, pass_filenames: false}
      - {id: no-l0-splits, name: ban random-window splits, entry: uv run python scripts/check_no_l0_splits.py, language: system, pass_filenames: false}
```

### 21.6 `.github/workflows/ci.yml` (skeleton — pin every action to a commit SHA in W1)
```yaml
name: ci
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@<sha>
      - uses: astral-sh/setup-uv@<sha>
      - run: uv sync --extra lab --all-groups --frozen
      - run: uv lock --check
      - run: uv run nox -s lint types tests
      - run: uv run lint-imports
      - run: uv run deptry . && uv run pip-audit
  edge-image:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@<sha>
      - uses: docker/setup-qemu-action@<sha>
      - uses: docker/setup-buildx-action@<sha>
      - run: docker buildx build --platform linux/amd64,linux/arm64 -f docker/edge.Dockerfile .
      - run: docker run --rm kai-edge python -c "import importlib.util as u; assert u.find_spec('torch') is None"
```

### 21.7 `docker/edge.Dockerfile` (pin base by digest in W1)
```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm@sha256:<PIN>
COPY --from=ghcr.io/astral-sh/uv:<PIN_VERSION> /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --extra edge --no-dev --frozen --no-install-project
COPY src ./src
RUN uv sync --extra edge --no-dev --frozen
ENV PATH="/app/.venv/bin:$PATH" PYTHONHASHSEED=0
CMD ["python", "-m", "kai.edge.runtime"]
```

### 21.8 `docs/research/hypotheses.md` template
```markdown
# Pre-registered hypotheses (frozen at <date>, commit <sha>)
For each of H1…H7 (copy from PRD §4): statement · metric · dataset/split · test · threshold · multiple-testing plan · pivot if false.
Amendments: only via ADR with date and reason; never after seeing test-fold results.
```

### 21.9 Per-week definition of "green Friday"
`main` builds · all quality gates pass · the week's artifact is demonstrably *used* by another role's code · registry contains at least one run produced by it · ADRs filed · risk register updated.

---
*End of PRD v2.0. This is a living document: change it by PR, record why in an ADR, and bump the version in the header.*
