## Why

The two spec-driven development skills mix three layers that change at
different speeds: the methodology, the project's rules, and the usage of a
particular framework. Framework specifics (OpenSpec, Spec-Kit, Kiro,
committed documents) live as reference files inside both, so adding a
framework means editing the methodology skill and the builder; the
approval gate reviews the specification alone and treats the design as an
implementation draft, although a bounded design — the approach, its
constraints, and its preferences, never a step list — is what makes an
implementation controllable; and the recommended archive mode (a job that
pushes to the default branch after merge) needs a protected-branch bypass
that user-owned repositories cannot grant, so it never ran here. Now,
before the next project is built from these skills.

## What Changes

- New catalog `sdd`: holds the methodology skill and one skill per
  framework. `spec-driven-development` moves into it from `engineering`
  (its spec domain moves with it, by directory move, and its title line is
  corrected by hand).
- `spec-driven-development` — **BREAKING**: becomes platform- and
  framework-agnostic. Every per-tool passage leaves; the loop, the
  specification-quality rules, and the approval-package rule sit in
  SKILL.md; `references/adoption-decision.md` (levels, when it pays,
  approach families, rejected alternatives) loads only when the user is
  deciding whether or how to adopt; `references/tracked-work.md` (shapes,
  approval modes, reconciliation, request body, archiving at run time)
  loads only when a loop step meets the platform. The approval gate now
  waits for the complete approval package — the specification plus the
  design when one is warranted — and reviews the outcome and the approach
  bounds, never tasks; tasks and code follow approval. Archiving happens
  inside the request only, by hand or by the automation the framework
  skill installs; the after-merge mode is a rejected alternative. The
  skill hands the project's framework to the matching `sdd` skill through
  the installing skill.
- `meta-spec-workflow` — **BREAKING**: becomes framework-agnostic. The
  four tool references, the archive script, and the archive workflow and
  job assets leave; tool adoption (step 3) and the automation (step 7)
  are handed to the framework skill through the installing skill. The
  questioning round settles the approval package (and when a design is
  warranted) and the archive executor (by hand, or the framework skill's
  automation) instead of an archive mode; the deposited contract and the
  slot texts carry the package and the executor in platform vocabulary
  with no framework name, command, or script. `references/tracked-work-lifecycle.md`
  becomes `references/contract-design.md`.
- `openspec-workflow` (new, `sdd`): running OpenSpec changes through pull
  or merge requests — the approval package (`proposal.md`, delta specs,
  `design.md` when warranted; `tasks.md` after approval), the spec-less
  marker, the validator moments, archiving inside the request by hand or
  by a label-triggered bot, the `/spec show` and `/spec status` comment
  commands and the `spec/*` labels — plus, per platform, the automation
  that installs them: a required check that fails on an invalid structure
  or an unarchived related change, the comment-command workflow, the
  label-triggered archive workflow (same-repository branch: archive,
  commit, push with the platform token, remove the label; fork: mention
  the author with the commands to run), the status-label workflow, and
  GitLab CI equivalents with their limitations stated. Ships
  `scripts/spec_changes.py`.
- `spec-kit-workflow` (new, `sdd`): the same shape for Spec-Kit — the
  approval package is `spec.md` plus `plan.md`, `tasks.md` follows
  approval, completion before ready is every task ticked, no archive
  operation exists — with the check, the comment commands, and the
  progress labels per platform. Ships `scripts/spec_kit_features.py`.

## Skills touched

- `sdd/spec-driven-development` (modified; domain moved from
  `engineering/`): the approval package and the design's role, the draft
  and gate timing, in-request archiving with an executor, the contract
  facts, the request body, the handoff to the framework skill.
- `meta/meta-spec-workflow` (modified): description, the questioning
  round, the deposited contract, the second phase, the handoff to the
  framework skill; the tool-reference, archive-job, and script
  requirements are removed.
- `sdd/openspec-workflow` (new): the whole domain.
- `sdd/spec-kit-workflow` (new): the whole domain.

## Installed behavior

- `spec-driven-development`: an agent following it waits for the complete
  approval package before tasks, writes a bounded design, archives inside
  the request, and routes tool usage to the framework skill → `fix` for
  the gate and the archive (they corrected wrong installed behavior),
  `refactor!` for the split and the moved catalog.
- `meta-spec-workflow`: settles the package and the executor, deposits a
  framework-free contract, and hands adoption and automation to the
  framework skill → `fix` for the gate, `refactor!` for the removed
  tool references and archive assets.
- `openspec-workflow`, `spec-kit-workflow`: new capabilities → `feat`.

## Impact

- New catalog scaffold (`skills/sdd/README.md`, `README.zh.md`,
  `CONTEXT.md`), the `catalog/sdd` label and the issue forms' Catalog
  dropdown, the `ARCHITECTURE.md` catalog list, the root README pair, the
  `sdd` plugin in `.claude-plugin/marketplace.json` and the `engineering`
  plugin's shrunk skill list — the companion repository change
  `sdd-framework-skills-harness`.
- `.agents/skills/spec-driven-development` retargeted; two new symlinks.
- `skills/engineering/{README.md,README.zh.md,CONTEXT.md}` drop the
  skill; `skills/meta/README.md` + `README.zh.md` row for
  `meta-spec-workflow`; `skills/machine-learning/CONTEXT.md` pointers.
- The mirror of `archive_completed_changes.py` in this repository becomes
  a mirror of `openspec-workflow`'s `spec_changes.py`; the
  `spec-archive` workflow is replaced; this repository's contract, project
  skill, templates, and knowledge follow — all in the companion change.

## Non-goals

- Skills for Kiro or committed specification documents: named as approach
  families in the methodology only.
- Changing `meta-github-workflow` or `meta-gitlab-workflow`: they stay
  paradigm-neutral; the `/spec` comment commands are owned by the
  framework skill's workflow.
- Editing the CLI-generated `openspec-*` project skills.
- A GitHub App or personal-token push identity for the archive bot: it
  pushes with the platform token, and a maintainer approves the resulting
  check runs.

## Tracked work

No issue: user-directed change planned in conversation.
