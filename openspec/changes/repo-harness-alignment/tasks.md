## 1. Skill and repository files

- [x] 1.1 Toolchain commit: fold in the maintainer's `.devcontainer/devcontainer.json` (`deno`, `uv` features, tools unpinned) with the Python feature kept; `.gitignore` adds `deno.lock`; the ruff hooks in `.pre-commit-config.yaml` call the installed ruff; `check_ruff` retired from the Python validator; `no-commit-to-branch` hook; OpenSpec installed through Deno at the justfile pin into `.tools/bin` and OpenTofu at its latest release; `checks.yml` on Deno and the latest tools; verify `just check`, `just spec-sync` leaving the generated skills unchanged, and that a scratch commit on `main` is refused — proves the dev-container, OpenSpec, ruff-hook, and `no-commit-to-branch` items
- [x] 1.2 `deno.json` (imports by major range, `"lock": false`, `fmt`/`lint` include only the management code), `scripts/lib/` shared modules, `just test` and the Deno part of `just lint`; verify `deno check`, `deno fmt --check` and `deno lint` list nothing outside the include globs — proves the Deno configuration item
- [x] 1.3 Port `check_skill` to `scripts/check_skill.ts` with tests; verify equivalence on the baseline tree and fixtures — proves the port item for this script
- [x] 1.4 Port `validate_skills` to `scripts/validate_skills.ts` with tests, the install-command check, and the corrected self-containment message; the legacy plugin-manifest guard ported unchanged; verify equivalence, its tests, and the fixture `plugin.json` case — proves the port item, D4, and S15
- [x] 1.5 Port `gen_marketplace` to `scripts/gen_marketplace.ts` without `--check`; verify equivalence (`just gen-marketplace` leaves `marketplace.json` unchanged) — proves the port item
- [x] 1.6 Port `validate_harness` to `scripts/validate_harness.ts` with the `tools`, `workflows`, and `budget` checks, YAML read by `@std/yaml`; add `.github/actions/setup-tools/action.yml`; `github-checks.md` privileged-workflow list; verify the tree passes and each fault fixture of the verification plan fails with file and line — proves the workflow-check, `tools`, budget, and setup-action items
- [x] 1.7 Port `check_commit_safety` to `scripts/check_commit_safety.ts` with `--range` (per-commit lines, same-repository author emails, attribution trailers), without the layer printout or `--allow-private-email`; verify equivalence on staged fixtures and the range fixture tests — proves the commit-range item
- [ ] 1.8 Port `check_pr_policy` to `scripts/check_pr_policy.ts` without the bare `Spec:` path; `pr-policy.yml` guarded transition (SHA-pinned `setup-deno`, the base's `.ts` or `.py`) and the range scan step; verify equivalence on recorded payloads, the bare-path test, and a scratch run of the Deno branch — proves the `pr / policy` transition item and S13
- [ ] 1.9 Port `sync_labels` to `scripts/sync_labels.ts`; `labels-sync.yml` and its path filter; verify equivalence on a recorded label list and a dry run against the live repository that plans `spec/approved` as a create — proves the port item
- [ ] 1.10 Port `sync_issue_metadata` to `scripts/sync_issue_metadata.ts`; `issue-triage.yml`; verify equivalence on recorded issue payloads — proves the port item
- [ ] 1.11 Port `run_log_digest` to `.agents/skills/change-workflow/scripts/run_log_digest.ts`, failing on an unfetchable log or a run without `jobs`; verify tests with a stubbed `gh` and one live read of a past failed run — proves the port item and D6
- [ ] 1.12 Add `scripts/review_threads.ts` and `just review-threads <n>`; verify the recipe refuses a non-numeric argument and lists the threads of #117 — proves the review-threads item
- [ ] 1.13 Add `scripts/ci_gate.ts`; `checks.yml` `gate` job; verify tests over success, failure, cancelled, and skipped dependencies — proves the gate item
- [ ] 1.14 Port `spec_changes` to `scripts/spec_changes.ts`; `spec-labels.yml`, `spec-command.yml`, `checks.yml` spec path filter, the `spec-check` and `spec-changes` recipes; verify equivalence over every archived change in the `check`, `labels`, `status`, `show`, and `archive --dry-run` modes, a read-only `snapshot` of #117, and `just spec-sync` reproducing the committed skills — proves the port item and the OpenSpec item
- [ ] 1.15 Add `scripts/spec_approval.ts` and `just spec-approval <n>`; `spec-labels.yml` label events, deciding-step timeout, and failure step; `spec/approved` in `.github/labels.json`; verify the stubbed-`gh` tests and the bounds test of the verification plan and a read-only run against #117 — proves the approval-label item
- [ ] 1.16 `.claude/settings.json`: deny entries for `spec/approved`, the `ask` families, the two wildcard reads replaced; verify JSON parses and no allow entry is a prefix of a write form — proves the settings item
- [ ] 1.17 `checks.yml` `devcontainer` job in the gate's `needs`; the Python feature removed from the dev container; every remaining Python caller gone; verify the job on this pull request and the `.py` grep of the verification plan — proves the container-build item
- [ ] 1.18 `.agents/knowledge/management-code.md` (new) and `skill-quality.md ## Script assets`; `ruff.toml` header comment; verify `just validate` and `git diff origin/main -- ruff.toml` shows comment lines only — proves the knowledge-split item
- [ ] 1.19 `spec-workflow.md`: the lifecycle, freeze, archive executor (with the `openspec-archive-change` note), and `## Approval gates` with the label as the only full statements; `openspec/config.yaml` archive pointer; verify `just spec-validate` and readback — proves the owner map for its rules and D3
- [ ] 1.20 `agent-authority.md`: authority rules as owner, the label rule, `ask` reasons, threat-model risks; verify readback — proves the owner map for its rules
- [ ] 1.21 `github-workflow.md`, `github-checks.md`, `harness-maintenance.md`, `references.md` rebuilt to their owned rules, rows under 200 columns, the `backup-*` note removed, the Dependabot exemption stated; verify the budget check — proves the owner map for their rules and S16
- [ ] 1.22 `github-settings.md` rewritten as the settings declaration (Actions allowlist, full-length SHA pinning, merge methods, ruleset, security features); verify the read-only readback lists which rows differ — proves the settings-declaration item
- [ ] 1.23 `change-workflow`, `skill-authoring`, `code-review` skills point to the owners instead of restating them (D1, D2, D7); verify `just check-skill` on each and readback — proves the contradiction items
- [ ] 1.24 `AGENTS.md` within its budget with one line per always-on rule; `ARCHITECTURE.md` mechanisms only; `CONTRIBUTING.md` lifecycle, toolchain, and banner; the PR template and schema where they restate a rule; verify the budget check and the pointer check — proves the entrypoint and D10 items

## 2. External impact

- [ ] 2.1 `.github/dependabot.yml` covers `.github/actions/*`; read-through
- [ ] 2.2 The `harness-maintenance.md` register row for `scripts/spec_changes.py` becomes a note that the repository's script is its own port; verify no row names a deleted file
- [ ] 2.3 Stale passages S1–S12, S14, S17 removed or dated; verify grep returns none

## 3. Tests

- [ ] 3.1 Equivalence results for every port, with the intentional differences, recorded for the Validation section — proves the port item
- [ ] 3.2 Permissions: every script run through its shebang in its real invocation, local and CI — proves the shebang decision
- [ ] 3.3 The audit's duplication grep re-run, each rule in full only in its owner — proves the owner map
- [ ] 3.4 Clean-context readbacks of the verification plan (`AGENTS.md`, `spec-workflow.md`, `management-code.md`, `skill-authoring`, `code-review`, the archive question, the settings) — proves the knowledge items
- [ ] 3.5 Settings readback before ready, with the rows to apply listed in the description — proves the settings-declaration item
- [ ] 3.6 Untouched product: the `skills/` and generated-file diff empty, ruff over the public skills unchanged from `main`
- [ ] 3.7 Record the skipped cases (privileged workflows and the live `pr / policy` branch after merge, live `ask` prompts) with reasons

## 4. Finish

- [ ] 4.1 Run `just check`, write the results to the pull request's Validation section linking the verification plan, record the harness review in `harness-maintenance.md`, and leave the change unarchived for the maintainer's second deliberation
