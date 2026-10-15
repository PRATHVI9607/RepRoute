# Glossary

The whole team uses these words identically (PRD §6.1). Code names are in `monospace`.

| Term | Meaning |
|---|---|
| **Window** `x` | `N` consecutive int16 samples of one vibration channel (`N = 4096` at 64 kHz = 64 ms). |
| **Reference model** `f_raw` | The cloud model trained on raw windows. Its decision is the reference for the certificate. |
| **EnvNet** | The reference model used for the Paderborn data: a fixed envelope-spectrum front-end (`EnvFront`, 96 values) and a logistic regression (ADR 0006). SpecNet (ADR 0004) is the earlier spectral-shape model, still used by the synthetic fixture. |
| **Supervised bottleneck** R6 | For the Paderborn data: `f_raw`'s front-end followed by a learned 16-value projection, sent as 16 int8 bytes. The edge therefore runs the reference model's feature extraction, not its decision rule. |
| **Representation** `r` | A payload format: an edge encoder `e_r` plus a cloud head `h_r`. `r0_raw` is raw. |
| **Menu** | The ordered list of representations. Index 0 is always raw; the order is the column order everywhere. |
| **Decision** `ŷ_r(x)` | `h_r(e_r(x))`: the class the cloud outputs when it receives representation `r`. |
| **Decision-agreement loss** `d_r(x)` | `1[ŷ_r(x) ≠ f_raw(x)]`. What the certificate controls. Called *risk* when averaged. |
| **Label fidelity** | Macro-F1 of `ŷ_r` against ground-truth labels. Reported, never certified. |
| **Probe** `φ(x)` | 20 cheap features computed for every window (`kai.certify.probe`). |
| **Sufficiency predictor** `ĝ_r(φ)` | Estimate of `P(d_r = 1 | φ)`, one small network per representation, fitted on `ptrain`. |
| **Safe set** `S_λ(x)` | `{raw} ∪ {r : ĝ_r(φ(x)) ≤ λ}`. Raw is always a member. |
| **Safe-set loss** `L(x; λ)` | `max_{r ∈ S_λ(x)} d_r(x)`: 1 if *any* member of the safe set disagrees. |
| **λ̂** | The calibrated threshold. `−1` means "raw only"; `1` means "everything". |
| **V0 / V1 / V2** | Calibration variants: CRC (expected risk), Learn-then-Test (high probability), policy-specific CRC. |
| **Partitions** | `train` (models) · `ptrain` (sufficiency predictor) · `calib` (λ̂) · `test`. No bearing is in two. |
| **L1 / L2 split** | Leave-bearings-out / additionally leave-one-operating-condition-out. L0 (random windows) is forbidden. |
| **Disagreement Table** | Per window × representation: decision, `d_r`, correctness, payload bytes (`kai.tasks.pipeline.Table`). |
| **Bundle** | Immutable, hash-verified directory with everything the edge and the cloud need for one calibration. |
| **Contract hash** | 4-byte hash of a representation's code version, parameters and model bytes; carried in every frame. |
| **State** `z` | What the selector knows about the system: bandwidth estimate, RTT, backlog, load ρ, remaining deadline, budgets. |
| **Selector** | The policy that picks one representation inside the safe set (S0 static, S1 greedy-safe, S2 MPC-safe, oracle). |
| **Flag** | Outcome class of a decision: `certified`, `deferred`, `uncertified`, `dropped`. Only `certified` is covered by the guarantee. |
| **Select-then-encode** | Only the probe, the predictor and the selector run for every window; exactly one encoder runs. |
| **Certify-then-optimize** | The safe set comes from the certificate; cost minimisation is a separate, swappable step inside it. |
| **Break-even bandwidth** `B*_ij(ρ)` | Bandwidth at which a compact representation `i` and a larger, cheaper-to-encode `j` have equal latency. |
| **Phase diagram** | Partition of (bandwidth, deadline, load) by the best representation. |
| **SLO violation** | A window not delivered within its deadline (late, deferred or dropped). Never counted as risk. |
| **Audit** | With probability `p_audit` the edge also sends raw so the cloud can observe `d_r` of the chosen representation. |

Units are fixed and appear in names: bytes, milliseconds, joules, bits per second (`payload_bytes`,
`t_enc_ms`, `energy_j`, `bandwidth_bps`).
