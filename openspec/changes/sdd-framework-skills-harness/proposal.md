## Why

The skill change `sdd-framework-skills` creates the `sdd` catalog, moves
`spec-driven-development` into it, adds two framework skills, and replaces
the after-merge archive job with archiving inside the pull request. This
repository generated its harness from those skills and pilots them: the
catalog needs its scaffold, the mirror of the archive script and the
`spec-archive` workflow no longer exist, the contract and the project
skill describe an approval gate that excludes the design and an archive
mode that never ran, and no check yet stops a pull request from merging
with an unarchived change or holds the merge open while the finished
implementation is being deliberated. Now, in the same pull request, so the skills
and the repository that ships them say the same thing.

## What Changes

- Catalog scaffold: `skills/sdd/README.md`, `README.zh.md`, `CONTEXT.md`;
  the `ARCHITECTURE.md` catalog list; the root README pair's catalog table
  and install example; the `catalog/sdd` label and the Catalog dropdown of
  the three issue forms; the `sdd` plugin in `.claude-plugin/marketplace.json`
  (landing with the first skill in the catalog) and the shrunk
  `engineering` plugin; the `.agents/skills/` symlinks (one retargeted, two
  new); `skills/engineering/{README.md,README.zh.md,CONTEXT.md}` without
  the moved skill; the entropy-review date in `harness-maintenance.md`.
- Request automation: `.github/workflows/spec-archive.yml` is deleted with
  the after-merge job it held, and no job archives or pushes in its place;
  new `spec-command.yml` (`spec / command`, collaborator-only) and
  `spec-labels.yml` (`spec / labels`), both filled from
  `openspec-workflow`'s assets and both reading the head through the API
  with no checkout of it;
  `checks.yml` gains the `ready_for_review` and `converted_to_draft`
  activity types and its `checks / spec` job runs the strict validator
  plus the unarchived-change rule for the pull request's related changes
  (a warning on a draft, a failure when ready) and `check --all` on a push
  to `main`; `scripts/archive_completed_changes.py` is replaced by
  `scripts/spec_changes.py`, a byte-identical mirror of the skill's;
  `justfile` recipes `spec-check` and `spec-changes` replace
  `spec-archive-completed`; five `spec/*` labels in `.github/labels.json`
  (`applied_by: workflow` for the five the workflows manage);
  `scripts/validate_harness.py` follows (mirror pair, the `workflow`
  applier restricted to `spec/`, a `spec/*` taxonomy check against the
  script, the hard-coded-version scan).
- Approval gate and archive executor in this repository:
  `.agents/knowledge/spec-workflow.md` (design as part of the approval
  package, the lifecycle, the archive section rewritten around the
  in-request executor and the automation, the framework skill named),
  `.agents/skills/change-workflow/SKILL.md` (the draft opens with the
  complete package, then stop; archive in the pull request by hand or by the
  the implementer's archive command after the deliberation closes; the
  check that holds the merge until then), `.github/PULL_REQUEST_TEMPLATE.md`
  (records table, approval line, archive checklist item),
  `openspec/config.yaml` archive guidance,
  `openspec/schemas/skill-change/schema.yaml` design instruction,
  `AGENTS.md`, `ARCHITECTURE.md`, `.agents/knowledge/agent-authority.md`,
  `.agents/knowledge/github-workflow.md` (label rows),
  `.agents/knowledge/github-checks.md` (job rows, the privileged-read
  note), `.agents/knowledge/github-settings.md` (bypass row obsolete),
  `.agents/knowledge/harness-maintenance.md` (mirror rows),
  `skills/machine-learning/CONTEXT.md` pointers.
- The spec domain of `spec-driven-development` is moved by directory move
  to `openspec/specs/sdd/` and its title line corrected by hand (the
  narrow hand-edit exception the contract records).

## Skills touched

repository change

## Installed behavior

Agents working in this repository publish the draft once the proposal,
the delta specs, and the design are written, write tasks after the
closing, mark the pull request ready for the deliberation on the finished
implementation, archive every related change inside the pull request once
that deliberation closes, and see the `checks / spec` job hold the merge
red until they do. No job of this repository archives or pushes; the
maintainer syncs the labels once.

## Impact

`scripts/validate_harness.py` register checks (labels ↔ forms ↔ catalogs,
mirror pair, check table ↔ workflow jobs, `just` recipe names in
`AGENTS.md`); `just gen-marketplace`; the label sync is a maintainer action
after merge; the `checks / gate` required check is unchanged, so no
ruleset edit.

## Non-goals

- A push identity other than `GITHUB_TOKEN` for the archive bot.
- Editing `meta-github-workflow` or `meta-gitlab-workflow` or the
  CLI-generated `openspec-*` skills.
- Running `openspec validate --archived` in the check (a later change).

## Tracked work

No issue: companion of `sdd-framework-skills`, planned in conversation.
