#!/bin/sh
# Raspberry Pi 4B (64-bit OS) edge setup (PRD §16.3). Run on the Pi, from the repository root.
set -eu
sudo apt update && sudo apt install -y git curl iproute2
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv sync --extra edge --no-dev --frozen                 # edge packages only, no torch
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
uv run --no-sync python scripts/verify_env.py --edge-only
# a throttled Pi ruins timing: 0x0 means never throttled since boot
echo "throttle state: $(vcgencmd get_throttled 2>/dev/null || echo 'vcgencmd not available')"
echo "next: copy a bundle and a replay file to the Pi, then"
echo "  uv run --no-sync kai edge <bundle> <replay.npz> --host <cloud-ip> --port 7400"
