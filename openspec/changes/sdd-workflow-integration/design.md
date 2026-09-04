## Context

See proposal.md for motivation. Binding constraints:

- Engineering skills depend only on `core`; a pairing with a `meta` builder
  is an optional handoff with a fallback (`skills/engineering/CONTEXT.md`),
  so the SDD skill carries the lifecycle knowledge itself and hands the
  harness build to the builder.
- `meta` builders may depend on one another by name (`skills/meta/CONTEXT.md`);
  a contract builder can load a platform builder's `semantic-mapping.md`.
- Platform review approvals are dismissed by later pushes on GitHub (dismiss
  stale approvals) and GitLab (default "remove all approvals when commits are
  added"); drafts do not auto-request code owners.
- OpenSpec documents archiving after merge (recommended) or inside the pull
  request; Spec-Kit has no archive operation; Kiro ticks tasks in-file.
- The OpenSpec archive script is tool logic, not platform logic: it ships
  with `spec-driven-development` (`scripts/`), the platform builders ship
  only the workflow or job wrapper that calls the project's copy of it, and
  this repository's `scripts/archive_completed_changes.py` mirrors the skill's
  copy the way `scripts/sync_labels.py` mirrors a builder script.
- The repository's own alignment (archive workflow, knowledge files,
  `change-workflow`, pull request template) is the companion repository
  change `sdd-workflow-integration-harness` on the same branch.
- This repository's `.agents/knowledge/github-workflow.md` is the live example
  of a platform-worded deposit.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| SDD Behavior: shape, approval, draft timing, review scope, archive mode, tracked work | `spec-driven-development/SKILL.md` `## Specs and tracked work` (rewritten), `## The loop` steps 2–4 and 7, new `## What specification review examines` | — |
| SDD Handoff: the harness builder for spec workflows | `SKILL.md` `## Harness alignment` (fallback) + new `references/tracked-work-lifecycle.md` | "Read `references/tracked-work-lifecycle.md` when designing or repairing how tracked work, change requests, templates, and the archive step carry the specification — including when the harness builder is declined or absent." |
| SDD Script: archive script | `scripts/archive_completed_changes.py` (the OpenSpec archive automation; tool-specific by nature, so it lives with the SDD skill, not the platform builders) | `references/tracked-work-lifecycle.md` archive-mode section names it |
| SDD Trigger | `SKILL.md` description | — |
| meta-spec-workflow Behavior: questions | `SKILL.md` step 2 (questions 5–7), step 6 removal list, gotcha | — |
| meta-spec-workflow Behavior: deposit, approval scope | `assets/spec-workflow.md` (approval gate, change request shape, archive mode, lifecycle, tracked work), `references/durable-output.md` | — |
| meta-spec-workflow Behavior: tool references | `references/openspec.md`, `spec-kit.md`, `kiro.md`, `committed-documents.md` | existing step-3 selection |
| meta-workflow-design Behavior: neutral design, platform deposit, required platform | `SKILL.md` boundaries, steps 1, 2, 4, 5, 6, gotchas; `references/durable-output.md`; `assets/platform-workflow.md` (renamed from `project-workflow.md`); one sentence in `references/management-model.md` | — |
| meta-workflow-design Behavior: specification-only draft | `references/management-model.md` Change Request and Acceptance; `SKILL.md` gotcha | — |
| meta-agent-authority Behavior: gate and H1 | `SKILL.md` step 1 and step 2; `references/authority-profiles.md` H1 | — |
| meta-agent-authority Behavior: platform verbs | `SKILL.md` deposit step; `references/durable-output.md`; `assets/agent-authority.md` header | — |
| meta-github/gitlab Behavior: deposits implemented | `SKILL.md` contract paragraphs; `references/semantic-mapping.md` header; `decision-tree.md`; `durable-harness.md`; `spec-expression.md` knowledge deposit; planning references; `assets/agents-md-*.md`, `assets/project-skill.md` pointers | existing routing rows |
| meta-github/gitlab Behavior: take-work, templates, spec-only request | `references/spec-expression.md` new section; `assets/project-skill.md` SPEC_RULE / SPEC_ISSUES; `references/issues-and-prs.md` / `work-items-and-mrs.md` state machine; `assets/pull-request-template.md` / `mr-template-default.md` | existing routing rows |
| meta-github/gitlab Behavior: archive job | `assets/workflow-spec-archive.yml` / `assets/gitlab-ci-spec-archive.yml` (platform wrappers calling the project's `scripts/archive_completed_changes.py`); `references/spec-expression.md` archive-mode guidance; `references/rules-and-protection.md` / GitLab governance note | `spec-expression.md` |

## Description

`spec-driven-development` (the only description that changes substantially)
must state, besides its current capability, that the skill settles how
specifications and tracked work interact (when the draft opens, whether a
specification needs its own change request, how archiving lands, that work
items and change requests link instead of restate); must load for questions
about how issues, pull or merge requests, and specifications fit together,
where a specification is reviewed, and when to archive, in direct and
indirect phrasings (a user who describes their `openspec/` directory and
asks where reviewers discuss it without saying "SDD"); must not load for a
request to build platform intake forms, CI, or a project harness, which
belongs to the builders; must keep the tool list (Spec-Kit, OpenSpec, Kiro,
plain documents) and the existing adoption triggers; and must stay under
1024 characters (a warning above 900 is accepted rather than dropping the
tool list). The wording is tuned during implementation against the Trigger
scenarios. `meta-workflow-design` and `meta-spec-workflow` change one clause
each (deposit in platform vocabulary) and keep their triggers.

## Dependencies and handoffs

- `spec-driven-development` → `meta-spec-workflow` (out of range, optional
  handoff through `ryan-minato-skills-installing`, whole `meta` catalog);
  fallback: the new reference guides the agent itself. `plan-clarification`
  pairing unchanged.
- Contract builders → sibling platform builder's `references/semantic-mapping.md`
  (in range: `meta` is installed whole); fallback: install the whole catalog
  through the installer, never a command.
- No new cross-catalog or cross-repository dependency.

## External impact

- Other skills: `meta-github-workflow` and `meta-gitlab-workflow` consume the
  deposits that `meta-workflow-design`, `meta-spec-workflow`, and
  `meta-agent-authority` now write in platform vocabulary, so their contract
  paragraphs, `semantic-mapping.md` headers, decision trees, durable-harness
  registers, and asset pointers change in the same commit as the deposit
  rule — readback of both platform builders after the contract commit.
- `meta-git-branching`: no behavior change; only catalog documents stop
  calling it platform-neutral.
- Catalog documents: `skills/meta/CONTEXT.md` (contract flow, scope, the
  "use inside this repository" exemption), `skills/engineering/CONTEXT.md`
  disambiguation, one sentence in `skills/scaffold/CONTEXT.md`; README pair
  rows for `spec-driven-development`, `meta-workflow-design`,
  `meta-spec-workflow` — `just validate`.
- Mirrored files: `skills/engineering/spec-driven-development/scripts/archive_completed_changes.py`
  ↔ this repository's `scripts/archive_completed_changes.py` — `diff`;
  registration of the pair in `scripts/validate_harness.py` belongs to the
  companion repository change.
- Renamed asset `project-workflow.md` → `platform-workflow.md`: no symlink or
  marketplace entry references assets — `git grep -n project-workflow.md skills`
  must be empty.
- This repository's harness: aligned by the companion change
  `sdd-workflow-integration-harness` (knowledge files, `change-workflow`,
  pull request template, archive workflow).

## Decisions

- **Shape is selected by change propagation, not by profile** (serves SDD
  and meta-spec-workflow "shape" requirements). Profile A lists libraries
  as typical, so a profile rule would contradict itself; the propagation
  mode already records whether consumers depend on a contract. Alternative
  rejected: asking cold with no selecting fact.
- **Approval is a comment naming the commit** (serves SDD "review on the
  draft"). The review-approval state is dismissed by implementation pushes
  and conflates the two reviews. Alternative rejected: a checklist tick by
  the author, indistinguishable from self-approval.
- **The draft opens when the specification is written, before plan and
  tasks; review examines outcomes** (serves SDD "draft published",
  "review scope"). OpenSpec's propose generates all artifacts at once; the
  gate reviews proposal and delta specs only.
- **Archive mode defaults to automated where automation can push** (serves
  SDD "archive mode"). The job is serialized (`concurrency`
  group / `resource_group`), idempotent (full rescan), and fails without
  retry on a rejected push because the run triggered by the competing merge
  archives the rest. Alternative rejected: rebase-and-retry loops, which
  hide conflicts and can double-archive.
- **Under split, the integration branch holds approved unimplemented
  records**; stale = approved record with no open work item owning it.
- **Contract builders deposit in platform vocabulary themselves** (serves
  meta-workflow-design "deposit"). They load the sibling platform builder's
  mapping; the platform builder implements objects and appends to the same
  file. The neutral-vocabulary check inverts into a model-noun check.
  Alternative rejected: the platform builder writing the deposit, which
  leaves a contract builder run alone with no durable output.
- **Deposit name `<platform>-workflow.md`**; spec, authority, and branching
  files keep their topic names.
- **The archive script lives with the SDD skill** (tool logic), the
  platform builders wrap it (platform logic), and the repository mirrors it
  like `sync_labels.py`. Alternative rejected: shipping the script in both
  platform builders, which duplicates tool logic across catalogs.

## Risks / Trade-offs

- [SDD description above the 900-character warning] → kept under 1024.
- [Renaming `project-workflow.md` leaves stale pointers] → `git grep` task.
- [The repository's script copy drifts from the skill's] → `diff` task and
  validator registration.
- [A platform builder's wrapper calls a script the target lacks] → the
  wrapper's guidance routes to the SDD skill through the installer skill by
  role, and the builder copies the script into the project's `scripts/`.

## Verification plan

Solver tier for skill cases: the least capable tier the skills claim
(Sonnet-class); observation by the harness's transcript where available,
else the `SKILLS_LOADED` fallback line; isolation: a detached candidate
worktree per solver, or the degradation recorded. Per the user's decision
the run is minimal: the SDD trigger set and one outcome case each for
`spec-driven-development` and `meta-spec-workflow`; every other scenario is
skipped with its reason and listed below. Scores are recorded in the pull
request's Validation section, never here.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| SDD Trigger: Lifecycle question | "how should issues and PRs work now that we use OpenSpec?" (+ indirect: "our specs live in openspec/, where do reviewers discuss them, on the issue or the PR?") | loads (critical) | both load | Sonnet-class | transcript / SKILLS_LOADED | candidate worktree |
| SDD Trigger: Harness build request | "set up GitHub issue forms and CI for us" (+ near-miss: "add a spec field to our issue form") | does not load (critical) | both do not load | Sonnet-class | transcript / SKILLS_LOADED | candidate worktree |
| SDD Behavior: Library with downstream consumers | Task: a semver library with OpenSpec asks how specs and PRs should relate | recommends split (critical); names the consumer contract as the fact; offers combined as the deviation; leaves the decision to the user; states the approval record as a comment naming the commit | 4 of 5, critical met | Sonnet-class | output graded by a clean-context grader | candidate worktree |
| meta-spec-workflow Behavior: Propagation recorded as Dependency | Task: a target with `project-workflow.md`/`github-workflow.md` recording dependency propagation; run the builder's questioning round | asks shape, archive mode, author (critical); recommends split citing the propagation line; recommends the archive mode from the CI evidence; one recommendation per question | 3 of 4, critical met | Sonnet-class | output graded by a clean-context grader | candidate worktree |

Script and tool harnesses:
- SDD Script: Help / Representative run / Nothing completed / Repeated run / Bad arguments —
  `python3 skills/engineering/spec-driven-development/scripts/archive_completed_changes.py --help`;
  `--dry-run` on a fixture with nothing completed (exit 0, no output); a
  fixture change with every task ticked archived in a temporary worktree
  with strict validation passing; the same run repeated (no change);
  `--bogus` exiting 2 and naming the option; `just lint` passes.
- meta-github/gitlab Behavior: OpenSpec with automation — read-through of
  the two wrapper assets: `concurrency` / `resource_group`, a call to
  `scripts/archive_completed_changes.py`, no retry on push failure.

Skipped (user decision: minimal run; static checks and independent readback
cover them):
- SDD Behavior: Feature-driven application, Where the spec is reviewed,
  Planning before publication, Tasks offered for review, Automation cannot
  push, Automation available, Acceptance pasted into the issue; SDD Handoff:
  Handoff offered, User declines.
- meta-spec-workflow Behavior: No automation can push, GitHub project
  deposit, Reading the approval gate, Spec-Kit with automated archiving;
  meta-spec-workflow and meta-workflow-design Trigger: description (their
  descriptions change by one clause; the existing triggers are unchanged).
- meta-workflow-design, meta-agent-authority, meta-github-workflow,
  meta-gitlab-workflow: all Behavior scenarios; the two "Spec-Kit with
  automation" scenarios.
- For each skipped domain a clean-context readback of the changed files
  answers the scenario's THEN from the text alone; recorded as readback,
  not as a solver run.
