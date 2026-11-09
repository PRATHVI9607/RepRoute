#!/usr/bin/env bash
# Progress of Kai's long runs without asking an agent: active kai-* units and their logs.
# Drivers log to $KAI_LOG_DIR (default ~/.cache/kai/logs).
dir=${KAI_LOG_DIR:-$HOME/.cache/kai/logs}
echo "== units"
systemctl --user list-units 'kai-*' --no-legend --plain 2>/dev/null | awk '{print $1, $3, $4}' || true
[ -n "$(systemctl --user list-units 'kai-*' --no-legend 2>/dev/null)" ] || echo "(none running)"
for f in $(ls -t "$dir"/*.log 2>/dev/null | head -n 3); do
  echo "== $f"
  grep -E "START|OK|FAIL|DONE" "$f" | tail -n 6
done
