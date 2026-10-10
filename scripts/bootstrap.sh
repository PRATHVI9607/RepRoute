#!/bin/sh
# One-command developer setup (PRD §16.2). Linux / WSL2 / macOS.
set -eu
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv sync --extra lab --extra "${KAI_TORCH:-cpu}" --all-groups --frozen   # KAI_TORCH=gpu for CUDA
uv run --no-sync python scripts/verify_env.py
