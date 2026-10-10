#!/bin/sh
# Shape the container's uplink with tc (needs NET_ADMIN). Usage: shape <rate, e.g. 1mbit> [delay_ms]
set -eu
DEV="${DEV:-eth0}"
RATE="${1:?rate, e.g. 1mbit}"
DELAY="${2:-10}"
tc qdisc del dev "$DEV" root 2>/dev/null || true
# netem's `rate` delays every packet by its serialisation time, like a real link. A token bucket
# (tbf) would not: its burst lets any frame smaller than the bucket through at line speed.
tc qdisc add dev "$DEV" root netem delay "${DELAY}ms" rate "$RATE" limit 10000
tc qdisc show dev "$DEV"
