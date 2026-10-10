# ADR 0003 — CPU torch by default, CUDA as an opt-in extra

Date: 2026-10-10 · Status: accepted · Amends: PRD §10.3 ("GPU not required"), §21.1

**Decision.** `torch` is no longer part of the `lab` extra. Exactly one of two mutually exclusive
extras supplies it:

```
uv sync --extra lab --extra cpu --all-groups --frozen    # default: CPU wheels
uv sync --extra lab --extra gpu --all-groups --frozen    # CUDA 13.0 wheels (make setup-gpu)
```

Training code takes `--device auto|cpu|cuda`; `auto` uses CUDA only if the installed build has it.

**Why.** The project owner asked for both paths so that laptops with and without a GPU can work.
On the development laptop a single leave-bearings-out fold did not finish in 45 minutes on CPU; the
reference-model stage alone takes about 6 minutes on an RTX 2050.

**Consequences.**
- Both builds are in `uv.lock`; `uv` refuses to install them together.
- GPU training is not bit-reproducible across machines. Everything downstream of training is
  unaffected: the Disagreement Table is computed from the **exported ONNX models on CPU**, so what is
  certified is exactly what the cloud runs.
- CI, Docker images and the edge stay CPU-only. Every fold records nothing about the device in its
  artifacts yet — add it to `metrics.json` before mixing CPU- and GPU-trained folds in one result.
