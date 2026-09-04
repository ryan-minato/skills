## Context

See proposal.md. Constraints: `main` is protected by the `Default` ruleset
(pull request only, required checks), so the archive workflow's push needs
a bypass the maintainer grants; `GITHUB_TOKEN` pushes trigger no further
workflows; `scripts/check_pr_policy.py` reads the template headings from the
base branch and requires every checklist box ticked on a ready pull request,
with a `Spec:` line matching an in-flight or archived change path;
`scripts/validate_harness.py` lists the mirrored file pairs and the
workflow-job-to-`github-checks.md` mapping.

## Placement

| What Changes bullet | File(s) | Check that proves it |
|---|---|---|
| Spec layout convention, repository changes skip specs | `.agents/knowledge/spec-workflow.md` Domains and Artifacts; `openspec/config.yaml` context; `ARCHITECTURE.md`; `AGENTS.md`; `change-workflow` §3.4 | `just validate`; a clean-context readback names the layout and the skip-spec rule |
| Project schema default | `openspec/schemas/skill-change/`; `openspec/config.yaml` `schema:`; `scripts/validate_harness.py` `check_openspec` | `openspec schemas --json` lists it as project; `python3 scripts/validate_harness.py` |
| Combined shape, automated archive | `.agents/knowledge/spec-workflow.md` (Change request shape, Archive mode, Lifecycle); `agent-authority.md` ready conditions; `change-workflow` §3–§7 and gotchas; `.github/PULL_REQUEST_TEMPLATE.md` checklist; `openspec/config.yaml` archive guidance | `scripts/check_pr_policy.py` on a body built from the new template; readback |
| `spec-archive` workflow and script | `scripts/archive_completed_changes.py` (mirror of `skills/engineering/spec-driven-development/scripts/archive_completed_changes.py`, registered as a pair in `scripts/validate_harness.py`); `.github/workflows/spec-archive.yml` (push to `main`, `concurrency: spec-archive`, `cancel-in-progress: false`, `permissions: contents: write`, install pinned CLI, script, strict validate, commit, push without retry); `scripts/archive_completed_changes.py`; `justfile` recipe; `github-checks.md` job table; `github-settings.md` bypass action; `harness-maintenance.md` | script harness below; `python3 scripts/validate_harness.py` (job listed) |
| Verification plan in `design.md` | `skill-authoring/SKILL.md`, `references/testing.md`, pull request template Validation comment | readback |
| Companion change naming | `.agents/knowledge/spec-workflow.md`, `github-workflow.md` branch rule | readback |

## Decisions

- **Two change records on one branch** for a skill change that needs harness
  work: the skill change keeps the branch slug; the companion is
  `<slug>-harness`. Alternative rejected: two pull requests, which would
  merge the harness rule before the skills that explain it, or a single
  mixed change, which cannot both carry and skip specs.
- **This pull request archives in-request** under the rule in force on
  `main`; the automated mode applies from the next merge after the bypass is
  granted. Alternative rejected: relying on the new workflow for its own
  pull request, which cannot push until the bypass exists.
- **The script fails on a rejected push and never rebases**: the run that
  the competing merge triggers archives whatever remains.

## Risks / Trade-offs

- [Bypass not granted] → the in-request rule stays in force; the knowledge
  file says so and names the maintainer action.
- [Two `Spec:` lines on the pull request] → `check_pr_policy.py` searches
  for one matching line; both match; recorded in `spec-workflow.md`.
- [Archive commit skips the `checks` workflow] → the job runs strict
  validation itself and touches only `openspec/`.

## Verification plan

Per What Changes bullet, the command or observation that proves it:

- Script: `python3 scripts/archive_completed_changes.py --help` prints usage;
  `--dry-run` on the repository with no completed change exits 0 with no
  output; in a temporary worktree a fixture change with every task ticked is
  archived, `openspec/specs/` carries its delta, and `just spec-validate`
  passes; the identical command repeated changes nothing; `--bogus` exits 2
  and names the option.
- Workflow: read-through confirms the `concurrency` block with
  `cancel-in-progress: false`, `on: push` to `main`, the strict validation
  step before the push, and a push step with no retry; `python3
  scripts/validate_harness.py` passes with the job listed in
  `github-checks.md`.
- Pull request template: `python3 scripts/check_pr_policy.py` (its body
  check) accepts a body built from the new template with every box ticked
  and rejects one with a box unticked.
- Knowledge and skills: a clean-context subagent given only the changed
  files answers: where a skill's spec lives, how a repository change is
  planned, when the draft opens, where approval is recorded, what ready
  requires, and who archives.
- `just check` passes.

Skipped: none.
