# Edge-plane results ledger (`edge` branch)

This is the detailed ledger for the laptop-side edge work on branch `edge`. It records the
metrics from the current experiment summaries, the verification runs, and the caveats that the
paper-facing report compresses. The paper-facing report is still
[`docs/research/results.md`](research/results.md); this file is the audit trail.

The results are frozen at branch state `92a90ff` (2026-10-24). The working tree was clean when
this ledger was written. No hypothesis threshold was changed to produce any result below.

## Provenance

| Item | Value |
|---|---|
| Branch | `edge` |
| Results frozen at | `92a90ff` |
| Dataset | Paderborn, 5 L1 folds and 5 L2 folds |
| Timing regime | `power-profiles-daemon` profile `performance`; 16/16 CPUs on the `performance` governor and `performance` EPP |
| Laptop CPU | 16 logical CPUs, `amd-pstate-epp`, max 2.0 GHz, boost enabled |
| Profile | `artifacts/profiles/local.json` (ignored local artifact) |
| Bundle | `bb809ab9f5c8` |
| Bundle profile hash | `e31425b47743c64ef4c2bc306b5fe381ea5ebccbb75b8db17f5bcc132d256c33` |
| Bundle git SHA | `a4ad4088f4cb2e83c924660d7124adc5ca4d7dbe` |
| Selected representation | `r7_vq`: 16 B payload, 44 B frame |
| Small-frame rate policy | keep the configured `initial_bps` prior; ADR 0015 |
| Loopback backend | in-process token-bucket pacer, no Pi |
| Docker backend | containers, `tc netem` rate + 10 ms delay |

The profile and bundle are local artifacts and are not committed. The summaries and registry rows
that identify them are committed. Raw Docker telemetry (`experiments/docker/*.msgpack`) is ignored.

## Verification status

| Check | Result |
|---|---|
| `make check` | lint, import contracts, mypy, and **134 tests** pass |
| `make mini-paper` | **9 passed**, 5 expected sklearn `ConvergenceWarning`s, 132.64 s |
| `make figures` | all seven figures regenerated |
| `make report` | `docs/research/results.md` regenerated |
| Docker E7d run | 300 windows offered and 300 delivered at each of 256 kbit/s, 1 Mbit/s, 5 Mbit/s |

The test count includes the new edge tests:

- `tests/unit/test_edge_pacing.py`
- `tests/unit/test_experiment_metadata.py`
- `tests/unit/test_profile_selection.py`
- zero-bandwidth coverage in `tests/unit/test_analysis.py`

## Hypothesis summary

| Hypothesis | Result | Where |
|---|---|---|
| H1 diversity | **false** | E3 |
| H2a certificate, expected | **false** | E2 |
| H2b certificate, PAC | **false** | E2 |
| H3 adaptation value | **false** | E3 |
| H4 encode-cost sensitivity | **false** | E5 / E5pi |
| H5 ablation value | **false** | E5 / E5pi |
| H6 phase model | **true** | E4 |
| H7 sim-to-real | **false overall** | E7 fails; E7d passes on its Docker backend |

H7 is not claimed as supported overall. The loopback result is a failed result and is retained.
The Docker result is recorded separately because its fixed 10 ms netem delay changes which term
dominates total latency.

## E1 — edge profile

Source: [`experiments/e1/summary.json`](../experiments/e1/summary.json)

| Metric | Value |
|---|---:|
| Folds | 5 |
| Representations profiled | 9 |
| `f_raw` macro-F1 | 0.608124 |
| Majority macro-F1 | 0.183582 |
| `f_raw` | `env` |
| Bottleneck | `teacher_front` |
| `t_probe_ms` | 0.527546 |
| `t_pred_ms` | 0.279255 |
| `t_sel_ms` | 0.264742 |
| Probe + predictor + selector | 1.071543 ms |
| `t_io_ms` | 0.050000 (floor) |
| Mean load exponent | 0.000000 |
| `io_load_exponent` | 2.259259 |
| IO excess at load 0 | −0.371152 ms |
| IO excess at load 0.8 | 1.897249 ms |
| `t_enc_ms/r6_bottleneck`, 1 sensor | 0.899577 |
| `t_enc_ms/r6_bottleneck`, 8 sensors | 0.587988 |
| `t_enc_ms/r6_bottleneck`, 32 sensors | 0.413504 |
| Overhead, 1 sensor | 1.071543 ms |
| Overhead, 8 sensors | 0.754739 ms |
| Overhead, 32 sensors | 0.291763 ms |

The negative load-0 calibration means the model overpredicted in that cell. `t_io` is at its
0.05 ms floor, so it cannot be reduced by the current calibration method. The three sensor counts
match the duty-cycle profiles used by E3.

## E2 — certificate variants

Source: [`experiments/e2/summary.json`](../experiments/e2/summary.json)

`H2a_pass = false`, `H2b_pass = false`. Folds = 5.

| Variant | α | Risk mean | 95% CI | Folds within α | Bytes/raw | Non-raw available |
|---|---:|---:|---|---:|---:|---:|
| `v0` | 0.02 | 0.019727 | [0.015415, 0.026605] | 0.60 | 0.049479 | 1.00 |
| `v0` | 0.05 | 0.048147 | [0.034959, 0.077908] | 0.80 | 0.006494 | 1.00 |
| `v0` | 0.10 | 0.087787 | [0.061699, 0.145012] | 0.80 | 0.002253 | 1.00 |
| `v1` | 0.02 | 0.017917 | [0.014091, 0.023883] | 0.80 | 0.060330 | 1.00 |
| `v1` | 0.05 | 0.045090 | [0.033610, 0.072742] | 0.80 | 0.007078 | 1.00 |
| `v1` | 0.10 | 0.084605 | [0.059713, 0.139111] | 0.80 | 0.002338 | 1.00 |
| `v1_per_rep` | 0.02 | 0.031062 | [0.008606, 0.061789] | 0.40 | 0.429815 | 1.00 |
| `v1_per_rep` | 0.05 | 0.060430 | [0.029478, 0.101429] | 0.40 | 0.111956 | 1.00 |
| `v1_per_rep` | 0.10 | 0.089569 | [0.052222, 0.147517] | 0.40 | 0.012770 | 1.00 |
| `v2` | 0.02 | 0.006361 | [0.005199, 0.007851] | 1.00 | 0.001953 | 1.00 |
| `v2` | 0.05 | 0.006361 | [0.005167, 0.007843] | 1.00 | 0.001953 | 1.00 |
| `v2` | 0.10 | 0.006361 | [0.005177, 0.007848] | 1.00 | 0.001953 | 1.00 |
| `v0_static` | 0.02 | 0.009917 | [0.008076, 0.012332] | 1.00 | 0.001953 | 1.00 |
| `v0_static` | 0.05 | 0.028731 | [0.023469, 0.036249] | 1.00 | 0.001953 | 1.00 |
| `v0_static` | 0.10 | 0.028731 | [0.023429, 0.036428] | 1.00 | 0.001953 | 1.00 |
| `v0_cascade` | 0.02 | 0.011383 | [0.009697, 0.013939] | 1.00 | 0.034777 | 1.00 |
| `v0_cascade` | 0.05 | 0.025858 | [0.020045, 0.037241] | 1.00 | 0.001953 | 1.00 |
| `v0_cascade` | 0.10 | 0.051255 | [0.038763, 0.080962] | 0.80 | 0.001953 | 1.00 |

Same-bearing control:

| α | Risk, same bearings |
|---|---:|
| 0.02 | 0.019632 |
| 0.05 | 0.048724 |
| 0.10 | 0.099673 |

## E3 — scenario grid and adaptation value

Source: [`experiments/e3/summary.json`](../experiments/e3/summary.json)

Price preset `metered_cellular`; 5 folds; 1296 cells.

| Metric | Value |
|---|---:|
| `H1/reps_dominant_in_10pct_of_cells` | 1 |
| `H1/b4_extra_cost_vs_oracle` | −0.411722 |
| `H1_pass` | false |
| `H3/cost_reduction_vs_b4` | −1.002519 |
| `H3/cost_reduction_vs_b4_ci_lo` | −1.631250 |
| `H3/cost_reduction_vs_b4_ci_hi` | −0.548769 |
| `H3/cost_geo_ratio_vs_b4` | 5.716643 |
| `H3/slo_reduction_vs_b5_bursty` | 0.337579 |
| `H3/slo_reduction_vs_b5_ci_lo` | 0.251984 |
| `H3/slo_reduction_vs_b5_ci_hi` | 0.394258 |
| `H3_pass` | false |
| Oracle dominant share, `r6_bottleneck` | 0.9954 |
| `kai_s1_v0_static` cost reduction vs B4 | −0.271100 |
| `kai_s1_v0_static` probe rate | 1.000000 |
| `kai_s1_cascade` cost reduction vs B4 | −0.595642 |
| `kai_s1_cascade` probe rate | 0.117345 |

Per-fold values:

| Fold | H1: 1 − oracle/B4 | H3: 1 − Kai/B4 | H3: 1 − Kai/B5 |
|---|---:|---:|---:|
| 1 | −0.604516 | −0.493966 | 0.311923 |
| 2 | −0.299353 | −1.409052 | 0.405610 |
| 3 | −0.719788 | −0.534147 | 0.199717 |
| 4 | −0.898071 | −0.610883 | 0.348847 |
| 5 | −0.977659 | −1.964548 | 0.421800 |

`per_fold.H1_cost` is `1 − oracle/B4`, `per_fold.H3_cost` is `1 − Kai/B4`, and
`per_fold.H3_slo` is `1 − Kai/B5`. The H1 summary metric is derived as
`1 / (1 − mean(H1_cost)) − 1`, so it is not literally the same quantity as the per-fold column.
H1 requires at least three dominant representations and a B4-to-oracle cost gap of at least 25%;
this run has one dominant representation (`r6_bottleneck`) and a negative B4 extra-cost. H3
requires both a cost reduction of at least 20% and a bursty SLO reduction of at least 30%; the SLO
side clears the threshold but the cost side does not.

## E4 — phase boundaries

Source: [`experiments/e4/summary.json`](../experiments/e4/summary.json)

| Metric | Value |
|---|---:|
| `H6_pass` | true |
| Boundaries measured | 8 |
| Boundaries matched to theory | 1 |
| Theoretical boundaries | 4 |
| Maximum representative-boundary error | 0.047312 (4.73%) |
| Median latency-model error | 0.072481 (7.25%) |
| Latency bias at load 0 | −0.068246 (−6.82%) |
| Latency bias at load 0.8 | −0.100631 (−10.06%) |
| Family maximum error | 0.656071 (65.61%) |
| Family matched / measured / theory | 2 / 2 / 4 |

Representative boundaries:

| Load | Low | High | Measured bit/s | Theoretical bit/s | Relative error |
|---:|---|---|---:|---:|---:|
| 0.0 | `r6_bottleneck` | `r4_summary` | 161,465.67 | 169,484.34 | 0.047312 |
| 0.8 | `r7_vq` | `r4_summary` | 73,227.70 | — | — |
| 0.8 | `r4_summary` | `r7_vq` | 137,155.77 | — | — |
| 0.8 | `r7_vq` | `r4_summary` | 263,726.45 | — | — |
| 0.8 | `r4_summary` | `r5_ae` | 2,790,793.55 | — | — |
| 0.8 | `r5_ae` | `r4_summary` | 4,496,896.67 | — | — |
| 0.8 | `r4_summary` | `r5_ae` | 10,846,556.94 | — | — |
| 0.8 | `r5_ae` | `r7_vq` | 23,623,433.88 | — | — |

Family boundaries:

| Load | Low | High | Measured bit/s | Theoretical bit/s | Relative error |
|---:|---|---|---:|---:|---:|
| 0.8 | `r2b_wavelet` | `r1_lossless` | 11,200,698.01 | 32,566,934.55 | 0.656071 |
| 0.8 | `r1_lossless` | `r0_raw` | 30,340,846.17 | 51,662,988.96 | 0.412716 |

H6 passes on the representative boundaries. The family check is exploratory: the theoretical
51.7 Mbit/s boundary lies outside the 31.6 Mbit/s measurement grid, so the measured 30.3 Mbit/s
crossing is not resolvable. The other family pair still differs. The capacity model was not
changed to chase the family check.

## E5 — ablations at nominal edge load

Source: [`experiments/e5/summary.json`](../experiments/e5/summary.json)

`edge_slowdown = 1.0`; 5 folds.

| Metric | Value |
|---|---:|
| `H4/choice_changed_in_high_load_cells` | 0.000000 |
| `H4/slo_with_encode_cost` | 0.157348 |
| `H4/slo_ignoring_encode_cost` | 0.158491 |
| `H4_pass` | false |
| `H5/cost_reduction_vs_best_family` | 0.017524 |
| `H5/ci_lo` | 0.000160 |
| `H5/ci_hi` | 0.038130 |
| `H5_pass` | false |

Policy totals:

| Policy | Cost total | Certified risk | SLO |
|---|---:|---:|---:|
| `audit_aci` | 8,750,505.56 | 0.005802 | 0.106217 |
| `compression_only` | 71,950,847.86 | 0.018369 | 0.538149 |
| `kai_s1_v0` | 8,750,505.56 | 0.005802 | 0.106217 |
| `learned_only` | 8,955,075.24 | 0.005794 | 0.110895 |
| `no_certificate` | 9,840,826.85 | 0.005925 | 0.131171 |
| `no_encode_cost` | 8,738,966.14 | 0.005780 | 0.106266 |
| `no_learned` | 62,950,721.12 | 0.065874 | 0.485516 |
| `no_predictor` | 5,256,680.01 | 0.005887 | 0.098862 |
| `no_queue_awareness` | 8,501,384.96 | 0.005930 | 0.144088 |
| `oracle` | 8,794,786.89 | 0.000000 | 0.108262 |

Cost reduction of Kai versus each comparison:

| Comparison | Reduction |
|---|---:|
| `audit_aci` | +0.000000 |
| `compression_only` | +0.878882 |
| `learned_only` | +0.017524 |
| `no_certificate` | +0.110643 |
| `no_encode_cost` | −0.001114 |
| `no_learned` | +0.860207 |
| `no_predictor` | −0.664645 |
| `no_queue_awareness` | −0.195232 |
| `oracle` | −0.034890 |

Per-fold H5 values: `[0.000139, 0.035475, 0.000160, 0.000181, 0.051665]`.

## E5pi — modelled 8× edge slowdown

Source: [`experiments/e5_pi_class/summary.json`](../experiments/e5_pi_class/summary.json)

This is a **modeled 8× slowdown**, not Raspberry Pi hardware. 5 folds.

| Metric | Value |
|---|---:|
| `edge_slowdown` | 8.0 |
| `H4/choice_changed_in_high_load_cells` | 0.005556 |
| `H4/slo_with_encode_cost` | 0.653193 |
| `H4/slo_ignoring_encode_cost` | 0.657731 |
| `H4_pass` | false |
| `H5/cost_reduction_vs_best_family` | 0.002084 |
| `H5/ci_lo` | 0.000958 |
| `H5/ci_hi` | 0.003792 |
| `H5_pass` | false |

Policy totals:

| Policy | Cost total | Certified risk | SLO |
|---|---:|---:|---:|
| `audit_aci` | 70,626,975.58 | 0.005756 | 0.600192 |
| `compression_only` | 84,101,189.28 | 0.014515 | 0.697054 |
| `kai_s1_v0` | 70,626,975.58 | 0.005756 | 0.600192 |
| `learned_only` | 70,775,397.54 | 0.004283 | 0.600873 |
| `no_certificate` | 71,144,742.03 | 0.004880 | 0.611809 |
| `no_encode_cost` | 70,583,903.09 | 0.004118 | 0.616800 |
| `no_learned` | 79,725,241.79 | 0.072787 | 0.679457 |
| `no_predictor` | 34,919,732.96 | 0.005887 | 0.423343 |
| `no_queue_awareness` | 68,893,281.87 | 0.010714 | 0.618986 |
| `oracle` | 40,611,341.55 | 0.000000 | 0.385935 |

Cost reduction of Kai versus each comparison:

| Comparison | Reduction |
|---|---:|
| `audit_aci` | +0.000000 |
| `compression_only` | +0.160225 |
| `learned_only` | +0.002084 |
| `no_certificate` | +0.007276 |
| `no_encode_cost` | −0.000599 |
| `no_learned` | +0.113820 |
| `no_predictor` | −1.022552 |
| `no_queue_awareness` | −0.025228 |
| `oracle` | −0.740125 |

Per-fold H5 values: `[0.003079, 0.001718, 0.000453, 0.000450, 0.004722]`.

## E6 — drift and operating-condition shift

Source: [`experiments/e6/summary.json`](../experiments/e6/summary.json)

α = 0.05, audit probability 0.02, 1 Mbit/s, one sensor. L1 holds bearings out; L2 additionally
holds one operating condition out.

| Metric | L1 | L2 |
|---|---:|---:|
| Folds | 5 | 5 |
| Safe-set risk, static threshold | 0.048147 | 0.127509 |
| Policy risk, static threshold | 0.006419 | 0.009121 |
| Policy risk, ACI | 0.006361 | 0.008693 |
| Bytes, static | 223.570175 | 447.749428 |
| Bytes, ACI | 187.086419 | 357.954042 |
| Audit bytes | 142.373748 | 147.365976 |
| Cloud audit risk estimate | 0.000000 | 0.012080 |
| Drift fired | 0.971454 | 0.885135 |

The L2 static safe-set risk exceeds α, which is a descriptive failure of the static set under
condition shift. The realised greedy-policy risk stays below 1% on both levels, and ACI reduces
bytes on both. L2 was trained locally after the missing-fold gap was identified; the L2 fold
artifacts are ignored, not committed.

## E7 — loopback sim-to-real

Source: [`experiments/e7/summary.json`](../experiments/e7/summary.json)

Backend: loopback sockets, token-bucket pacing, same laptop as the DES, no Pi.
Bundle: `bb809ab9f5c8`; profile SHA: `e31425b4...`; bundle git SHA: `a4ad4088...`.

`H7_pass = false`. Current summary (the later of two runs):

| Rate (bit/s) | Real p50 (ms) | Sim p50 (ms) | p50 gap | Real p95 (ms) | Sim p95 (ms) | Selection agreement | Delivered |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256,000 | 2.981846 | 3.676545 | 0.232977 (23.30%) | 3.385537 | 3.676545 | 1.00 | 1.00 |
| 1,000,000 | 1.917487 | 2.653545 | 0.383866 (38.39%) | 2.375118 | 2.653545 | 1.00 | 1.00 |
| 5,000,000 | 1.667065 | 2.371945 | 0.422827 (42.28%) | 2.105758 | 2.371945 | 1.00 | 1.00 |

`max_p50_rel_gap = 0.422827`; `min_selection_agreement = 1.0`.

Run-to-run note: an earlier E7 run on the same bundle produced 6.3% / 16.0% / 20.5% p50 gaps
(max 20.5%). The summary file now holds the later run. Both runs fail H7; the difference is
recorded in ADR 0014 and the work log. The model was not changed after seeing either result.

## E7d — Docker sim-to-real

Source: [`experiments/e7d/summary.json`](../experiments/e7d/summary.json)

Backend: Docker containers, `tc netem` rate + 10 ms delay on the edge egress, same laptop.
Bundle: `bb809ab9f5c8`; profile SHA: `e31425b4...`; bundle git SHA: `a4ad4088...`.

`H7_pass = true`; `max_p50_rel_gap = 0.021872`; `min_selection_agreement = 1.0`.

| Rate (bit/s) | Real p50 (ms) | Sim p50 (ms) | p50 gap | Real p95 (ms) | Sim p95 (ms) | Selection agreement | Delivered | Real rate estimate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000,000 | 7.824685 | 7.653546 | 0.021872 (2.19%) | 8.774706 | 7.653546 | 1.00 | 1.00 | 1,000,000 |
| 256,000 | 8.803530 | 8.676546 | 0.014424 (1.44%) | 10.021665 | 8.676546 | 1.00 | 1.00 | 1,000,000 |
| 5,000,000 | 7.408154 | 7.371946 | 0.004888 (0.49%) | 8.207754 | 7.371946 | 1.00 | 1.00 | 1,000,000 |

The real rate estimate stays at the 1 Mbit/s prior for the 44 B frame; the estimator is not
updated by sub-1000-byte frames (ADR 0015). The selected payload still delivers at all rates.

## E8 — selector overhead

Source: [`experiments/e8/summary.json`](../experiments/e8/summary.json)

| Metric | Value |
|---|---:|
| Probe + predictor + selector | 1.071543 ms |
| Overhead / 64 ms window | 0.016743 |
| Overhead / cheapest encoder | 2.738014× |
| Greedy-safe | 13.728627 µs/decision; 72,840.50 decisions/s |
| MPC h3 | 43.032646 µs/decision; 23,238.17 decisions/s |
| MPC h4 | 75.295719 µs/decision; 13,280.97 decisions/s |
| MPC h5 | 294.150639 µs/decision; 3,399.62 decisions/s |

The full per-window overhead is still larger than the cheapest encoder, so the Role C
“time-statistics-only probe / score every k-th window” task remains open.

## E10 — condition and re-identification leakage

Source: [`experiments/e10/summary.json`](../experiments/e10/summary.json)

| Representation | Payload bytes | Condition accuracy | Condition chance | Re-ID accuracy | Re-ID chance |
|---|---:|---:|---:|---:|---:|
| `r0_raw` | 8192 | 0.523718 | 0.25 | 0.965620 | 0.178333 |
| `r3_spectral` | 128 | 0.594463 | 0.25 | 0.976862 | 0.178333 |
| `r4_summary` | 24 | 0.462700 | 0.25 | 0.851951 | 0.178333 |
| `r5_ae` | 32 | 0.421568 | 0.25 | 0.682214 | 0.178333 |
| `r6_bottleneck` | 16 | 0.328470 | 0.25 | 0.593109 | 0.178333 |
| `r7_vq` | 16 | 0.368978 | 0.25 | 0.604287 | 0.178333 |

This is descriptive. It does not change the choice of `r7_vq` for the edge experiments.

## Known limitations

1. **E7 loopback is unstable and fails H7.** Two runs under the same bundle measured 20.5% and
   42.3% maximum p50 gaps. The real edge time varies more than the simulated side; the model was
   not tuned after seeing this. Docker E7d passes because its fixed delay dominates.
2. **One performance profile.** The profile was measured once under the pinned governor.
   `t_io` is at its floor and the load-0 calibration is negative. A repeated profile is needed
   before treating small timing differences as stable.
3. **E4 family check is exploratory.** H6 passes on representative boundaries; the family
   comparison has a 65.6% maximum error and one theoretical boundary lies outside the grid.
4. **L2 static certificate misses α.** The L2 static safe-set risk is 12.75% at α = 5%; the
   realised policy risk is below 1%. This is descriptive and is not converted into a hypothesis
   pass.
5. **Small-frame rate estimation.** The 44 B payload cannot produce a reliable goodput sample;
   the configured prior remains in force (ADR 0015).
6. **No Pi, cloud, or other-lab measurements here.** E5pi is a modelled 8× slowdown, not hardware.
   Energy, arm64 builds, Pi E1/E7/E8, cloud promotion, and lab-machine runs are handoff work.
7. **E3 price presets were not rerun on this branch.** `e3_battery_node`,
   `e3_deadline_critical`, and `e3_wired_gateway` exist in the repository but are not part of
   this edge refresh. The same applies to `experiments/drift_study`.

## Reproduction

From the repository root, with the laptop on AC power:

```bash
powerprofilesctl set performance
make folds FOLDS=5
make experiments
uv run --no-sync kai replay artifacts/folds/820e37cd09ba/s0_r0f0 data/processed/replay.npz --n 300
uv run --no-sync kai bundle artifacts/folds/820e37cd09ba/s0_r0f0
BUNDLE=artifacts/bundles/<id> FOLD=artifacts/folds/820e37cd09ba/s0_r0f0 make docker-loop
make figures
make report
make check
make mini-paper
```

`make experiments` includes E1 and does not run E7d. Build the bundle after E1 and before the
Docker loop. The profile, bundle, folds, and raw Docker telemetry are ignored local artifacts;
the summaries, figures, registry, and this ledger are tracked.

## Artifact map

| Artifact | Location | Tracked |
|---|---|---|
| Experiment summaries | `experiments/*/summary.json` | yes |
| Per-cell/per-run tables | `experiments/*/results.parquet` | yes |
| Run registry | `experiments/registry.parquet` | yes |
| Figures | `experiments/figures/` | yes |
| Paper-facing report | `docs/research/results.md` | yes |
| Profile | `artifacts/profiles/local.json` | no |
| Bundle | `artifacts/bundles/<id>/` | no |
| Folds | `artifacts/folds/` | no |
| Raw Docker telemetry | `experiments/docker/*.msgpack` | no |
| Edge diagnostics | `experiments/edge/` | no |
