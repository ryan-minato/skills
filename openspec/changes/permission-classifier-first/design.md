## Context

`.claude/settings.json` (from the `repo-harness-alignment` change) has 29 allow, 26 ask, and 32 deny entries; `agent-authority.md ## Permission entries` gives each ask family a reason. `.codex/config.toml` holds only the agentskills MCP server. Mechanisms, from the vendors' documentation (read 2026-10-21):

- Claude Code auto mode: rules resolve first (deny blocks in every mode; a content-scoped ask always prompts, even in auto mode; allow skips the classifier); read-only actions and in-directory edits are auto-approved; everything else goes to the classifier, which also reads `CLAUDE.md`. A project file cannot set `defaultMode: "auto"` or an `autoMode` block; both are user- or managed-level. Auto mode is the built-in starting mode in current versions.
- Codex: project `.codex/config.toml` and `.codex/rules/*.rules` load only for a trusted project; `approval_policy` (`on-request`, `never`, granular), `sandbox_mode` (`read-only`, `workspace-write`, `danger-full-access`), `[sandbox_workspace_write] network_access`, `approvals_reviewer` (`user`, `auto_review`); `prefix_rule(pattern, decision = "allow" | "prompt" | "forbidden", justification, match, not_match)`, the most restrictive match winning. Rules are documented as experimental.

See proposal.md for motivation.

## Placement

| Item | File and section | Check that proves it |
|---|---|---|
| Policy | `agent-authority.md ## Permission entries` | readback; `just validate` |
| Claude Code lists | `.claude/settings.json` | JSON parses; ask empty; each deny and allow entry traceable to the policy |
| Codex config | `.codex/config.toml` | TOML parses; keys as listed in Context |
| Codex rules | `.codex/rules/default.rules` | `codex execpolicy check` on one denied and one allowed command, when the CLI is available |
| Threat model, register, calibration | `agent-authority.md ## Threat model`; `harness-maintenance.md` register row and `## Layer calibration` | readback |
| Layout | `ARCHITECTURE.md` layout block | `just validate` (pointers) |

## External impact

- `skills/**`: unchanged.
- The maintainer's own settings: Claude Code runs in auto mode by default in current versions; Codex needs the project trusted and reads `approvals_reviewer` from the project file. Recorded in `agent-authority.md`, not committed as user settings.

## Decisions

- **Ask empty**: a prompt on a command the classifier could judge buys no safety once approval fatigue sets in, and it halts the run. The publish rule stays in `AGENTS.md` and `agent-authority.md ## Publishing`, which the classifier reads. Rejected: keeping `git push` as the one ask — the push is pre-authorized per change already, so the prompt only re-confirms.
- **Deny for clear risk only**: irreversible or authority-crossing acts (history rewrite on the remote, merge, approval, the approval label, repository settings and deletion, credential disclosure, privilege escalation, destroying local work). Kept as fool-proofing; the container stays the boundary.
- **Allow only the per-task checks**: they run repository code the agent edits anyway, run on every task, and are documented commands; the built-in read-only set covers `git status`, `git diff`, `gh pr view`, and the like. Rejected: the broad read allows — they skip the classifier for nothing.
- **Codex `workspace-write` with network**: every task needs `gh`, `git push`, and Deno's registry; a network-less sandbox would turn each into an escalation. The reviewer judges escalations; the container stays the boundary.

## Risks / Trade-offs

- [The classifier approves an outward write the user did not authorize] → `AGENTS.md` states the publishing rule and the classifier reads it; the deny list covers the irreversible acts; a push lands only on a branch, and `main` takes changes only through a reviewed merge.
- [A maintainer runs Claude Code in Manual mode] → unmatched commands prompt as before; nothing becomes less safe.
- [Codex rules syntax is experimental] → one file, checked with `codex execpolicy check` when available; a parse change shows at Codex start.

## Verification plan

Script and tool harnesses:
- `.claude/settings.json` parses; `permissions.ask` is empty; allow has only the listed recipes; each deny entry maps to a policy item.
- `.codex/config.toml` parses as TOML (`python3 -c 'import tomllib'`); `.codex/rules/default.rules` checked with `codex execpolicy check` when the CLI is installed, else skipped with the reason.
- `just check` passes.
- Clean-context readback of `agent-authority.md ## Permission entries`: which commands are refused, which run unreviewed, and who judges the rest.

Skipped:
- Live classifier decisions on outward writes: confirming one would publish.
