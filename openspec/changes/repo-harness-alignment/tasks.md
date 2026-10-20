## 1. Skill and repository files

- [x] 1.1 `.claude/settings.json`: deny list and allow additions; verify the file parses and a clean-context reader states the boundary — proves the deny-list bullet
- [x] 1.2 `AGENTS.md ## Always`: blocked-work and mismatch rules; verify `just validate` and the line count — proves the Always bullet
- [x] 1.3 `.agents/knowledge/agent-authority.md`: `## Pre-authorized actions`, `## Threat model`, update trigger; verify `just validate` — proves the authority bullet
- [x] 1.4 `.agents/skills/change-workflow/SKILL.md`: checkpoint rules in §4 and the handoff-file gotcha; verify `just check-skill .agents/skills/change-workflow` — proves the checkpoint bullet
- [x] 1.5 `.devcontainer/devcontainer.json`: pagers off, `github-cli` feature; verify `just validate` — proves the container bullet
- [x] 1.6 `scripts/validate_harness.py` `check_workflows` and docstring; `.agents/knowledge/github-checks.md` allowed-workflows line; `.github/workflows/checks.yml` gate job permissions; verify the validator passes on the tree and fails on a broken copy, `just lint` — proves the workflow-check bullet
- [x] 1.7 `.agents/knowledge/harness-maintenance.md`: `## Layer calibration` and the thinning question; verify `just validate` — proves the calibration bullet
- [x] 1.8 `.agents/knowledge/spec-workflow.md` artifact roles; `openspec/schemas/skill-change/schema.yaml` and `templates/proposal.md` deliverable section; PR template row; verify `just spec-validate` and `openspec instructions proposal` on a scratch change — proves the artifact-roles bullet

## 2. External impact

- [x] 2.1 `ARCHITECTURE.md` names the workflow check and the calibration; verify `just validate`
- [x] 2.2 `.agents/knowledge/harness-maintenance.md` register rows for the allowed-workflows line and the proposal section; read-through

## 3. Tests

- [x] 3.1 Clean-context readbacks: `AGENTS.md` (blocked work, mismatch), `agent-authority.md` (pre-authorized push, no isolation tier), change-workflow (squash sequence, drafts never pushed), `harness-maintenance.md` (six layers) — proves the readback items of the verification plan
- [x] 3.2 Record skipped or manual proofs (container rebuild, framework refusal of a denied command) with reasons

## 4. Finish

- [x] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, and leave the change unarchived for the maintainer's second deliberation
