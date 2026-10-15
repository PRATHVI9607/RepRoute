# One-command shortcuts. Everything runs in the locked uv environment.
.PHONY: docker-loop setup setup-gpu verify lint types tests check data folds profile experiments figures report paper-artifacts mini-paper

setup:            ## CPU-only lab environment (the default, PRD §10)
	uv sync --extra lab --extra cpu --all-groups --frozen
setup-gpu:        ## same, with the CUDA 13.0 build of torch
	uv sync --extra lab --extra gpu --all-groups --frozen
verify:
	uv run --no-sync python scripts/verify_env.py
lint:
	uv run --no-sync nox -s lint
types:
	uv run --no-sync nox -s types
tests:
	uv run --no-sync nox -s tests
check: lint types tests

data:             ## download the Paderborn archives and build data/processed + split manifests
	uv run --no-sync python scripts/download_data.py
	uv run --no-sync python scripts/download_traces.py
	uv run --no-sync python scripts/prepare_pu.py
folds:            ## train models and build the Disagreement Table (FOLDS / REPEATS overridable)
	uv run --no-sync kai folds --folds $(or $(FOLDS),5) --repeats $(or $(REPEATS),1)
profile:
	uv run --no-sync kai exp e1
experiments: profile
	for e in e2 e3 e5 e5pi e6 e8 e10 e4 e7; do uv run --no-sync kai exp $$e || exit 1; done
figures:
	uv run --no-sync python -m kai.analysis.plots
docker-loop:      ## closed loop in containers with kernel shaping, then the DES-vs-Docker comparison (E7d)
	uv run --no-sync kai replay $(FOLD) data/processed/replay.npz --n 300
	mkdir -p experiments/docker
	for r in 256kbit 1mbit 5mbit; do \
	  BUNDLE=$(BUNDLE) RATE=$$r N=300 docker compose -f docker/compose.yaml up --build --abort-on-container-exit || exit 1; \
	done
	uv run --no-sync kai exp e7d
report:
	uv run --no-sync python scripts/report.py
paper-artifacts: experiments figures report   ## regenerate every number, figure and results.md
mini-paper:       ## tiny synthetic end-to-end run used by the nightly CI job
	uv run --no-sync pytest tests/golden -m slow -q -n 0 --no-cov
