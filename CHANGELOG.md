# Changelog

## Unreleased — simulator, certificate and replication pass (2026-10-14, later)

### Changed
- **Timings follow the node's duty cycle.** E1 measures every encoder and the per-window overhead
  at one window every period / sensors; the scenario grid uses the matching timings (ADR 0008).
- **Cost model** gains a calibrated per-frame runtime overhead `t_io(ρ)`; with `t_io = 0` it is the
  PRD formula.
- **Audit loop:** raw decisions are audited for free, so the adaptive threshold can leave raw-only
  (ADR 0007).
- **Reporting:** a tested hypothesis is *supported* only if its threshold is met and Holm rejects;
  `risk_certified_joint` (the quantity Proposition 1 bounds) is reported next to the rate among
  certified decisions; Proposition 1 is corrected for tied scores.
- `kai_s1_v1` in E3 is now the single-threshold V1; the per-representation variant is
  `kai_s1_v1_per_rep`. The predictor cache is keyed on the fold's table.
- Docker base images pinned by digest; CI actions pinned to commit SHAs; pre-commit hooks frozen.
- Data licences recorded: Paderborn is CC BY-NC 4.0.

### Added
- `kai_s1_v0_static`: the V0 certificate with constant scores — one safe set for every window, no
  probe or predictor on the edge (exploratory, ADR 0008).
- Baseline B13, an AdaEdge-style codec bandit fed by audits (`docs/research/baselines-repro.md`).
- Measured 4G throughput traces in the scenario grid (`scripts/download_traces.py`, CC BY 4.0).
- Transfer dataset: HUST bearing data (`src/kai/data/hust.py`, `scripts/download_hust.py`,
  `scripts/prepare_hust.py`, `configs/dataset/hust.yaml`); results in
  `docs/research/results-hust.md`.
- The main comparison under every price preset (`kai exp e3 profile=… max_folds=5`).
- Training-seed variance in the report; `scripts/drift_study.py` (a negative result).
- Three L1 repeats (15 folds) instead of one.

## Reference-model pass (2026-10-14)

### Changed
- **Reference model `f_raw` for the Paderborn data is EnvNet** (envelope spectrum over its own
  median + logistic regression), chosen by a recorded search over about 150 candidates
  (ADR 0006, `docs/research/model-search.md`). SpecNet stays as `--f-raw spec` and for the
  synthetic fixture.
- **R6 (supervised bottleneck) is built on the reference model's front-end** for the Paderborn data
  (`--bottleneck teacher_front`); the generic CNN encoder stays as `--bottleneck generic`.
- Both choices are dataset settings (`configs/dataset/*.yaml`: `f_raw`, `bottleneck`).
- All experiments were re-run on the new folds; the earlier runs remain in the registry.

### Added
- `scripts/model_search.py`: the search harness (inner CV on training bearings only), its
  diagnostics and the generated report.
- E10, the privacy-leakage probe (`kai exp e10`), with a section in the results.
- `docs/research/baselines-repro.md`; first novelty sweep in `docs/research/novelty-log.md`
  (24 works added to the literature matrix).
- Subagent rules in the agent guide.

## First implementation pass (2026-10-10)

### Added
- Contracts: representation protocol, 28-byte wire frame, bundle manifest, split and run schemas.
- Data: Paderborn loader with int16 quantisation, synthetic fixture, bearing-wise L1 and L2 splitters.
- Representations R0–R7: raw, lossless, two wavelet codecs, spectral features, summary statistics,
  autoencoder latent, supervised bottleneck, VQ codes (learned ones run through ONNX Runtime).
- Reference model (low-capacity spectral-shape classifier; the waveform CNN is kept for comparison),
  equal-budget heads distilled from it, fold pipeline and the Disagreement Table.
- Certification: probe, sufficiency predictor, CRC (V0), Learn-then-Test (V1), policy-specific CRC
  (V2), audit-in-the-loop ACI, drift monitor.
- Selection: cost model, degradation ladder, greedy-safe (S1), MPC-safe (S2), oracle, baselines
  B1–B5 and B8–B11.
- Link: SimPy discrete-event simulator, traces, bandwidth estimator; asyncio edge runtime and cloud
  server; hash-verified calibration bundles.
- Experiments E1–E8 (plus the Docker comparison E7d and a slow-edge sensitivity run), run registry,
  generated results report, figures, Hydra configuration with four price presets.
- Tooling: uv lockfile with CPU and CUDA torch extras, nox, pre-commit, CI workflows, Dockerfiles.

### Results
See `docs/research/results.md`. Of the pre-registered hypotheses only H5 was supported.

### Not done
- Raspberry Pi validation and measured energy (no power meter; Pi not yet available).
- R8 (JEPA), R9 (MiniRocket), S3 (bandit), S4 / B12 (RL), E9 (escalation), E10 (privacy probe),
  transfer dataset, the paper text.
- Measured cellular traces (the shipped ones are synthetic).
