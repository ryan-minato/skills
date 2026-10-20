## Context

The harness as built before this change: `AGENTS.md` (121 lines; `## Always` then `## When To Read What`, `## Validation`, `## Keep In Sync`), eight knowledge files under `.agents/knowledge/`, three project skills (`change-workflow`, `skill-authoring`, `code-review`), `scripts/validate_harness.py` (nine checks before this change — labels, template, intake, pointers, suite, spec, checks, openspec, ruff — plus `workflows` after it) run by `just validate`, pre-commit, and `checks / quality`; pre-commit with the commit-safety gate, detect-secrets, gitleaks, ruff, and both validators; seven workflows, all with `permissions: {}` at the top and job-level permissions except the `gate` job, every action SHA-pinned, one using `pull_request_target` (`spec-labels.yml`; `issue-triage.yml` runs on `issues` events); `.claude/settings.json` with an allow list of twenty read-only or recipe commands and no deny list; a dev container with no secrets, `OPENSPEC_NO_UPDATE_CHECK` and `DO_NOT_TRACK` set, pagers unset; the authority policy at H1. Binding constraints: `AGENTS.md ## Always` (never weaken a check; the policy and the entrypoint may not be edited to widen agent authority — this change narrows it); the mirrored pairs in `harness-maintenance.md` and the pointer check in `validate_harness.py` (every new knowledge section stays inside an existing file, so no new pointer is needed); the entrypoint budget (121 lines now; the two new Always rules add about nine, landing at about 130); the project schema's templates are read by `spec_changes.py` only for `tasks.md` checkboxes, so a new proposal section is safe for the automation. See proposal.md for motivation.

## Placement

| What Changes bullet | File and section | Check that proves it |
|---|---|---|
| Deny list and allow additions | `.claude/settings.json` `permissions.deny`, `permissions.allow` | JSON parses; a clean-context read states the boundary sentence; manual: `git push --force` attempted in a session is refused by the framework |
| Blocked work continues; mismatch lands | `AGENTS.md ## Always`, two bullets | `just validate` (entrypoint pointers); line count at or under 130; clean-context readback states the three landing places |
| Pre-authorized actions; threat model | `.agents/knowledge/agent-authority.md` two new sections and the "Update this file when" list | `just validate`; readback: an agent given only the file answers which pushes need no new decision and why no isolation tier applies |
| Checkpoint rules; handoff hygiene | `.agents/skills/change-workflow/SKILL.md ## 4` and `## Gotchas` | `just check-skill .agents/skills/change-workflow`; readback: the squash sequence is stated non-interactively and the push names one ref |
| Pagers off; github-cli feature | `.devcontainer/devcontainer.json` `remoteEnv`, `features` | parses as JSONC (the file carries comments), which `just validate` reads; `just validate` (ruff pin check still agrees); in a rebuilt container `gh pr list` returns without a pager |
| Workflow-shape check | `scripts/validate_harness.py` `check_workflows`, docstring; `.agents/knowledge/github-checks.md` the `pull_request_target` row | `python3 scripts/validate_harness.py` passes on the tree; fails with the file and line on a fixture copy with an unpinned action, a job without permissions, and a head checkout under `pull_request_target`; `just lint` |
| Layer calibration and thinning question | `.agents/knowledge/harness-maintenance.md ## Layer calibration`, `## Periodic entropy review` step | `just validate`; readback: six rows each with a variable |
| Artifact roles; deliverable and pre-authorized actions | `.agents/knowledge/spec-workflow.md ## Artifacts` (`### Artifact roles` subsection and tests), `openspec/schemas/skill-change/schema.yaml` proposal instruction, `templates/proposal.md` (`skill-authoring/SKILL.md` lists no proposal sections and `ARCHITECTURE.md` does not enumerate them, so neither changes) | `just spec-validate`; `just spec-sync` unchanged (the schema is the project's, not the CLI's); a new change created with `openspec new change` shows the section |

## Description

Not applicable: no public skill changes.

## Dependencies and handoffs

- None added. The pre-authorized-actions section points to the proposal section of the change record and to the conversation's authorization as the change workflow already defines it.
- The two sibling changes are independent: this change adopts the spec-contract shapes (artifact roles, deliverable and pre-authorized actions) into the project's own contract and schema, which the `spec-contract-alignment` design names as a separate repository change; nothing here depends on either sibling merging first.

## External impact

- `ARCHITECTURE.md`: the validator's check list and the schema description — read-through; `just validate` (pointers).
- `.agents/knowledge/harness-maintenance.md` synchronization register: a row for the `pull_request_target` list (`github-checks.md` ↔ `check_workflows`) and a row for the proposal section (`templates/proposal.md` ↔ `spec-workflow.md` artifact table ↔ `skill-authoring`) — read-through.
- `.github/PULL_REQUEST_TEMPLATE.md`: unchanged unless the Specification table's record descriptions name the proposal's sections (they do: "Why, what changes, skills touched, installed behavior, impact, non-goals"), in which case the row gains "deliverable and pre-authorized actions" and change-workflow §6 is re-read — `python3 scripts/check_pr_policy.py` on a sample body.
- `.gitmessage`, labels, workflows: unchanged.

## Decisions

- **Deny list in the framework file, boundary sentence beside it** (serves the first bullet): the methodology this repository publishes says a deny list is fool-proofing and the sandbox is the boundary; this repository's container holds no secrets, so the list guards against mistakes only. Rejected: a pre-commit or git hook intercepting `push --force` — the framework list is cheaper and the ruleset already blocks force push on `main`.
- **Pending decisions as an Always rule, not a new knowledge file** (serves the second bullet): it applies to every task and is four lines; a file would need a pointer and would load for nothing. Rejected: a section in the authority policy — that file loads at gates, not mid-task.
- **Mismatch landing with `catalog/repository`, no new label** (serves the same): the label exists and the triage check derives it; a `harness/drift` label would need the label sync and a form option. Rejected: a new label.
- **Checkpoint rules in the change workflow, not AGENTS.md** (serves the fourth bullet): they apply when committing, which the workflow owns; the Always section keeps its commit-format rule only.
- **Workflow-shape rules as a validator check, not Python unit tests** (serves the sixth bullet): the repository's management code is one validator with named checks and fix-it messages; a test suite would be a second mechanism. The reference repository's unit tests are the same idea in its own language. Rejected: a separate `scripts/tests/` directory.
- **Calibration table in the maintenance file** (serves the seventh bullet): it is read when thickness is questioned, which is the periodic review's trigger. Rejected: in `AGENTS.md` — it is not an always-on rule.
- **Artifact roles and the proposal section adopted now** (serves the last bullet): this repository is the first consumer of the sibling change's shapes, and running them here is the cheapest verification that they fit a real schema. Rejected: waiting for the sibling to merge — the schema is the project's own and the methodology change does not alter it.

## Risks / Trade-offs

- [The deny list blocks a legitimate `git reset --hard` the checkpoint recipe needs] → the checkpoint sequence uses `git reset --soft` and `git checkout -- <path>` only; the deny entry names `--hard`.
- [The new validator check fails on a workflow edit CI needs] → the failure names the file and line and the allowed list in `github-checks.md`; the check is written against the seven workflows as they are, and `just validate` runs before the commit.
- [`AGENTS.md` passes 130 lines (the budget is 130 or fewer)] → the two rules are written in four lines each; if the file exceeds the budget the Keep In Sync table's last row moves to `harness-maintenance.md`.
- [The proposal section is read as authorizing remote writes by itself] → the section's text says it names what the conversation's authorization is asked to cover; the change workflow's step 1 stays the gate.
- [Pagers set in the container differ from CI] → CI runs non-interactively already; the variables affect the container only, and `github-checks.md` says so.

## Verification plan

Per What Changes bullet:

- Deny list: `python3 -c "import json; json.load(open('.claude/settings.json'))"`; a clean-context reader given the file states which commands are denied and why; a session attempting `git push --force origin HEAD` reports the framework's refusal (manual, recorded in the Validation section).
- Always rules: `just validate`; `wc -l AGENTS.md` under 130; a clean-context reader given `AGENTS.md` answers "what do you do when a plan needs a decision it did not cover" with the continue-and-queue rule and the four fields, and "what do you do with a command the entrypoint names that the justfile lacks" with the three landing places.
- Authority sections: `just validate`; a clean-context reader given the file states that pushing the branch the proposal names needs no new decision and that opening a comment does; and answers why no isolation tier applies here.
- Checkpoint rules: `just check-skill .agents/skills/change-workflow`; a clean-context reader given the skill states the squash sequence without an interactive rebase and that drafts are never pushed; a dry run on a scratch branch: two draft commits, `git reset --soft`, one commit, `git log` shows one.
- Dev container: parses as JSONC (the file carries comments); `just validate`; in a rebuilt container (or with the variables exported in the current shell) `gh pr list --limit 1` and `git log -1` return without a pager (manual, recorded).
- Workflow check: `python3 scripts/validate_harness.py` exits 0 on the tree; on a scratch copy with one action unpinned, one job's `permissions` removed, and `actions/checkout` with `ref: ${{ github.event.pull_request.head.sha }}` added under `pull_request_target`, it exits 1 with three findings naming file and line; `just lint`.
- Calibration table: `just validate`; a clean-context reader names the six layers and the variable behind each; the thinning question is in the review steps.
- Artifact roles and proposal section: `just spec-validate`; `openspec new change scratch-test` in a scratch worktree followed by `openspec instructions proposal --change scratch-test --json` shows the section in the template and the instruction; the worktree is deleted; `python3 scripts/check_pr_policy.py` on a sample body if the PR template changed.
- Finish: `just check`.

## Open Questions

- Whether `.claude/settings.json` should also carry an `ask` entry for `gh pr ready` (the H1 action) so the framework records the decision moment; the proposal keeps the `ask` list empty, and the maintainer may add it on the draft without changing the placement.
