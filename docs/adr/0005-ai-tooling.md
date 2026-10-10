# ADR 0005 — AI-assist tooling

Date: 2026-10-10 · Status: provisional (the A/B protocol of PRD §13.2 has **not** been run)

**Installed on the lead development machine, outside the repository**
- *ponytail* (`DietrichGebert/ponytail`): "smallest complete change" rules.
- *codegraph* (`colbymchenry/codegraph`): local code index; usage telemetry turned off.

**Removed.** *claude-mem* was installed and removed the same day: its default provider spends the
user's model plan on memory compression. Cross-session context uses the assistant's built-in memory.

**Not adopted.** RTK, ccusage, Serena, context-mode, token-savior: PRD §13.2 requires a measured
≥ 15 % token reduction with no quality loss before adoption, and that measurement was not made.

**Repository policy.** `.claude/`, `CLAUDE.md`, `.mcp.json` and `.codegraph/` are git-ignored at the
project owner's request, so agent configuration is per-developer, not shared. This departs from
PRD §11, which lists `.claude/` and `CLAUDE.md` as committed files.
