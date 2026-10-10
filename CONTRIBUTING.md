# Contributing

## Setup

```bash
scripts/bootstrap.sh              # uv, Python 3.12, locked environment (CPU torch), env check
KAI_TORCH=gpu scripts/bootstrap.sh   # the same with the CUDA 13.0 build of torch
```

Windows: use WSL2 Ubuntu 24.04. macOS: native for lab work, Docker for netem runs.

## Before you open a pull request

```bash
make check        # ruff + format check + import contracts + L0-split ban, mypy, pytest + coverage
```

| Gate | Threshold |
|---|---|
| `ruff check`, `ruff format --check` | zero findings |
| `mypy` | strict flags on `contracts/`, `certify/`, `select/`; standard elsewhere |
| `pytest` | all green; coverage ≥ 85 % on `certify/` + `select/`, ≥ 70 % overall |
| `lint-imports` | the edge and contracts stay free of torch / sklearn / pandas / scipy / lightgbm |

## Rules that are not negotiable

- **Splits are by bearing.** No random-window split in anything reported; `scripts/check_no_l0_splits.py` fails the build on the usual culprits.
- **Agreement is not accuracy.** `d_r` compares with the raw model's decision. Keep the two apart in names, columns and plots.
- **Units in names**: bytes, ms, J, bit/s.
- **Randomness is passed in** as an `np.random.Generator`. No global seeds.
- **A failed hypothesis stays failed.** `docs/research/hypotheses.md` is pre-registered; changing a metric or a threshold needs an ADR dated before the result was seen.
- **`contracts/` and `certify/`** changes need a test, an ADR note and a second reviewer.

## Workflow

Trunk-based: short-lived branches `role/topic`, Conventional Commits (`feat:`, `fix:`, `exp:`,
`docs:`, `chore:`), squash-merge, pull requests of at most ~400 changed lines. Experiments that go
into the paper run from a clean tree at a tagged commit; the run registry records the git SHA and
the lockfile hash of every run.

## Where things go

See `docs/architecture.md` for the package map and `docs/glossary.md` for vocabulary.
