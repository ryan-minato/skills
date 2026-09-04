## Why

The repository's own harness contradicts the lifecycle the skills now
teach: the draft pull request opens only after implementation and archiving,
approval is said to happen "on the draft", archiving is manual, and the
OpenSpec layout has no fixed convention for where a skill's spec lives or
how a change to the repository itself is planned. This companion change
aligns the harness with the `sdd-workflow-integration` skill change on the
same branch.

## What Changes

- Spec layout convention: `openspec/specs/<catalog>/<skill-name>/spec.md` for
  public skills, permanently; changes to the repository itself (environment,
  harness, tooling, checks, workflows, documents) are `skip_specs` changes
  with a proposal, design, and tasks and no spec domain.
- Project schema `openspec/schemas/skill-change/` becomes the default: skill-
  shaped proposal, requirements named by kind, a design carrying the
  verification plan, fixed task groups; `openspec/config.yaml` selects it and
  drops its `rules` block; `scripts/validate_harness.py` checks the schema
  and config agree.
- Change request shape **combined** and archive mode **automated**: the draft
  pull request opens as soon as the change record is clarified; the
  maintainer approves it with a comment naming the commit; plan and tasks
  follow approval; ready requires every task complete and scenarios verified;
  a `spec-archive` workflow archives completed changes after merge to `main`,
  one run at a time, via `scripts/archive_completed_changes.py`, a mirror of
  the `spec-driven-development` skill's bundled script.
- `change-workflow`, `spec-workflow.md`, `agent-authority.md`,
  `github-checks.md`, `github-settings.md`, `harness-maintenance.md`,
  `ARCHITECTURE.md`, `AGENTS.md`, the pull request template, and
  `openspec/config.yaml` state the new lifecycle; `skill-authoring` and its
  testing reference put the verification plan in `design.md`.
- A skill change that also needs harness work carries a companion repository
  change named `<slug>-harness` on the same branch; the pull request names
  both on `Spec:` lines.

## Skills touched

Repository change (`skip_specs: true`): no public skill domain.

## Installed behavior

Agents working in this repository open the draft after clarifying the
change, wait for the maintainer's approval comment before planning and
implementing, stop archiving by hand once the archive workflow can push, and
plan every repository change as a `skip_specs` OpenSpec change.

## Impact

- New: `.github/workflows/spec-archive.yml`, `scripts/archive_completed_changes.py`,
  `openspec/schemas/skill-change/` (already written on this branch), a
  `just spec-archive-completed` recipe.
- Edited: `.agents/knowledge/spec-workflow.md`, `agent-authority.md`,
  `github-workflow.md` (branch and change naming), `github-checks.md`,
  `github-settings.md`, `harness-maintenance.md`,
  `.agents/skills/change-workflow/SKILL.md`,
  `.agents/skills/skill-authoring/SKILL.md` and `references/testing.md`,
  `.github/PULL_REQUEST_TEMPLATE.md`, `openspec/config.yaml`,
  `scripts/validate_harness.py`, `ARCHITECTURE.md`, `AGENTS.md`, `justfile`.
- Maintainer action after merge: grant the archive workflow's identity a
  bypass on the `main` ruleset; until then the in-request archive rule stays
  in force.

## Non-goals

- Changing `scripts/check_pr_policy.py` or the required checks.
- Backfilling specs for skills nobody is changing.

## Tracked work

No issue: companion of `sdd-workflow-integration`, planned in conversation.
