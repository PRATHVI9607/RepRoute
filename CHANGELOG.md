# Changelog

## Unreleased — first implementation pass (2026-10-10)

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
