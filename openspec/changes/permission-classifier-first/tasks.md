## 1. Repository files

- [x] 1.1 `.claude/settings.json`: deny as the policy lists (adding the `gh` token print), allow only the per-task check recipes, no ask list; verify the JSON parses, ask is absent, and each entry maps to the policy
- [x] 1.2 `.codex/config.toml` (approvals, sandbox with network, `approvals_reviewer = "auto_review"`) and `.codex/rules/default.rules` (`forbidden` for the deny list, `allow` for the recipes); verify the TOML parses and, when the Codex CLI is installed, `codex execpolicy check` on one denied and one allowed command
- [ ] 1.3 `agent-authority.md ## Permission entries` and `## Threat model`, `harness-maintenance.md` register row and layer calibration, `ARCHITECTURE.md` layout; verify `just validate` and a clean-context readback of `## Permission entries`

## 2. Verification

- [ ] 2.1 Run the verification plan of `design.md` and `just check`
