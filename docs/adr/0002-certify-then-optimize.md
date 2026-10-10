# ADR 0002 — Certify-then-optimize, select-then-encode

Date: 2026-10-10 · Status: accepted · Source: PRD §5.2, §6.2

**Decision.**
1. The safe set `S_λ̂(x)` is produced by the certifier (`kai.certify`) alone. Selectors
   (`kai.select`) may only return a representation inside it for a `certified` action.
2. Only the probe, the sufficiency predictor and the selector run for every window. Exactly one
   encoder runs, and all three overheads are charged to the same latency and energy budget.

**Why.** The guarantee `E[d_π] ≤ α` then holds for any cost-aware policy, so the optimizer can be
replaced (greedy, MPC, bandit, learned) without re-proving anything; and the selector's own cost
cannot be hidden.

**How it is enforced.**
- Property test `test_certified_decisions_are_always_inside_the_safe_set` fuzzes every selector.
- The ladder (`kai.select.base.ladder`) is the only place a `certified` flag is issued for Kai.
- `import-linter` keeps `kai.edge` and `kai.contracts` free of torch / sklearn / pandas / scipy /
  lightgbm, so the edge can only run what it could run on a Pi.
