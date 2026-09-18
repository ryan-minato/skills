## Context

See proposal.md. Catalog scaffolding is enumerated in `ARCHITECTURE.md`
`## Catalogs` (checked two-way by `check_architecture_md_catalog_list()`
in `scripts/validate_skills.py`), the root README pair's catalog table and
plugin-install example, `.github/labels.json` (`catalog/*` compared with
the directories in `skills/` by `check_labels()` in
`scripts/validate_harness.py`, which also requires every issue form's
Catalog dropdown to list the same names plus `repository`), and
`.claude-plugin/marketplace.json` (one plugin per non-empty catalog;
`check_marketplace_manifest()` errors on a non-empty catalog without an
entry and on a drifted `skills[]`). Every catalog directory needs
`README.md`, `README.zh.md`, and `CONTEXT.md` (`check_catalogs()`).
Precedent: `2026-09-11-machine-learning-catalog-harness`, commits
`184ff7b` and `921cfe4`. The new catalog is durable, so
`CATALOG_NAME_PREFIXES` and `DISPOSABLE_CATALOGS` stay unchanged.

The archive mechanism today: `.github/workflows/spec-archive.yml` (push
to `main`, needs a ruleset bypass never granted — `github-settings.md`
row "Ruleset bypass actors"), `scripts/archive_completed_changes.py`
(mirror pair in `validate_harness.py` `check_copies()`), the
`spec-archive-completed` recipe, and the `spec-archive / archive` row in
`github-checks.md` (`check_checks_doc()` demands one row per workflow job
name). `checks / gate` is the only ruleset-required check from
`checks.yml`; `pr / policy` and `scan-secrets` are the other two.
`labels.json` rows carry `applied_by` ∈ {form, triage, human}
(`APPLIERS`); a label nothing applies is an error.

Platform facts (verified 2026-09-17): a fork's `pull_request` run holds a
read-only token and no secrets, and the setting that would grant it write
exists for private repositories only, so a label or a reply on an external
contribution needs a privileged trigger; `issue_comment` runs the
default-branch workflow file; comments the platform token posts create no
workflow run, so a reply cannot loop.

## Placement

| What Changes bullet | File | Check that proves it |
|---|---|---|
| Catalog scaffold | `skills/sdd/README.md`, `README.zh.md`, `CONTEXT.md`; `ARCHITECTURE.md` `## Catalogs`; root `README.md` + `README.zh.md`; `.github/labels.json` `catalog/sdd`; `.github/ISSUE_TEMPLATE/{bug-report,feature-request,task}.yml`; `.claude-plugin/marketplace.json`; `.agents/skills/{spec-driven-development,openspec-workflow,spec-kit-workflow}`; `skills/engineering/{README.md,README.zh.md,CONTEXT.md}`; `skills/machine-learning/CONTEXT.md`; `.agents/knowledge/harness-maintenance.md` last-run date | `just validate`; `just gen-marketplace` then `git diff --exit-code .claude-plugin/marketplace.json`; `ls -l .agents/skills | grep -E 'spec-driven|openspec-workflow|spec-kit'`; a read of both files of each README pair; `just spec-sync` leaves `.agents/skills/openspec-workflow` in place |
| Spec domain move | `openspec/specs/sdd/spec-driven-development/spec.md` (moved, title line corrected) | `just spec-validate`; `git log --follow` shows the history |
| Request automation — workflows | `.github/workflows/spec-archive.yml` deleted; `spec-command.yml`, `spec-labels.yml` added; `checks.yml` (types, `checks / spec` steps) | YAML parses; `grep -rn '{{[A-Z]' .github/` empty; `grep -rn 'git .*fetch\|actions/checkout' .github/workflows/spec-*.yml` finds only base checkouts; `just validate` (`check_checks_doc`, hard-coded-version scan) |
| Request automation — script and recipes | `scripts/spec_changes.py` (mirror), `scripts/archive_completed_changes.py` deleted, `justfile` `spec-check` and `spec-changes` | `diff scripts/spec_changes.py skills/sdd/openspec-workflow/scripts/spec_changes.py` empty; `just spec-check` on this branch (draft mode warns, exit 0); `just spec-changes --help`; `just lint` |
| Request automation — labels and validator | `.github/labels.json` five `spec/*` rows; `scripts/validate_harness.py` (mirror pair, `workflow` applier, `check_spec_labels()`, scan set) | `just validate`; `python3 scripts/sync_labels.py --file .github/labels.json --repo ryan-minato/skills` dry run lists `catalog/sdd` and the five `spec/*` labels to create |
| Approval gate and executor — contract and skill | `.agents/knowledge/spec-workflow.md`, `.agents/skills/change-workflow/SKILL.md` | read-through; a clean-context read of the project skill can state what the draft's first content is, the stop, the archive executor, and the finish step |
| Approval gate and executor — templates and schema | `.github/PULL_REQUEST_TEMPLATE.md`, `openspec/config.yaml`, `openspec/schemas/skill-change/schema.yaml` | `just validate` (template headings, `Spec:` line); `python3 scripts/check_pr_policy.py` over a body built from the template with every item ticked passes; `just spec-validate` |
| Approval gate and executor — knowledge and maps | `AGENTS.md`, `ARCHITECTURE.md`, `.agents/knowledge/{agent-authority,github-workflow,github-checks,github-settings,harness-maintenance}.md` | `just validate` (pointers, recipe names, check table, mirror register); read-through |

## External impact

- The skill change `sdd-framework-skills` on the same branch supplies the
  skill directories, the assets the workflows are filled from, and the
  script the mirror copies; the marketplace `sdd` entry lands in the same
  commit as the first skill in the catalog.
- `scripts/validate_skills.py` is not edited; proof
  `git diff --stat origin/main...HEAD -- scripts/validate_skills.py` empty.
- The remote labels are created after the merge by the maintainer with
  `scripts/sync_labels.py --apply`; recorded as a maintainer action in
  the pull request handover. The `checks / gate` requirement is unchanged,
  so no ruleset edit.
- The first live run of the comment and label workflows happens on a later
  pull request; what is observed is recorded in `github-checks.md` then.

## Decisions

- **`sdd` is durable, no name prefix, no disposable marker** (serves the
  scaffold): the skills are installed one at a time like `engineering`;
  prefixes exist to group disposable builders.
- **The moved domain moves by `git mv`, not by an `ADDED` rebuild** (serves
  the domain move): history survives, and the delta's `MODIFIED` blocks
  resolve against the moved main spec; the title line is the one hand
  edit, recorded here and in the proposal.
- **The unarchived-change rule joins `checks / spec` instead of a new
  required check** (serves the automation): the gate already aggregates
  it; `durable-harness.md` says a paradigm's check joins the existing
  checks run; no ruleset edit and no first-run-before-required dance.
- **Draft warning, ready failure** (serves the automation): `checks.yml`
  gains the `ready_for_review` and `converted_to_draft` activity types so
  the verdict follows the draft state, as `pr-policy.yml` already does.
- **`pull_request_target` for the label workflow, nothing of the head in
  the runner** (serves the automation): a fork's `pull_request` token is
  read-only and gets no secrets, so a label cannot be applied from one;
  scripts come from the base checkout and the head is read through the
  REST API, never fetched or checked out. Rejected: moving the labels into
  the unprivileged check job, which works on this repository's own
  branches and fails silently on the contributions where the label matters
  most.
- **Labels applied by workflows get `applied_by: workflow`** (serves the
  validator): the register must say who applies a label; the value is
  restricted to the `spec/` prefix so no other label can claim it.
- **This pull request archives its two changes after the deliberation**:
  the same order it installs, so the request exercises the rule it ships.

- **This repository's `spec / *` workflows follow the same rule as the
  skill's assets** (serves the request automation): they are the
  dogfooded copy, so the read path changes with the assets in the same
  request rather than drifting. The mirror check keeps
  `scripts/spec_changes.py` byte-identical with the skill's copy.

## Risks / Trade-offs

- [The validators read the whole tree at commit time, so a skill
  directory staged without its symlink, README rows, or marketplace entry
  fails the hook] → the scaffold commit lands first without a marketplace
  entry (an empty catalog carries none); each skill lands in one commit
  with its symlink, README rows, and the regenerated marketplace; the
  second new skill is moved out of the tree while the first is committed.
- [`check_checks_doc` and the workflow set must agree in every commit] →
  the workflows, `github-checks.md`, `justfile`, `validate_harness.py`,
  and the script mirror land in one commit.
- [Label sync needs the remote] → maintainer action after merge; the dry
  run proves the file.
- [`just spec-sync` might touch the `openspec-workflow` symlink] → run it
  after creating the symlink; the diff must be empty.
- [This request's own `checks / spec` is red for as long as its
  deliberation lasts] → that is the rule it installs, and the status label
  says why; the request is never wrongly green.

## Verification plan

- Catalog scaffold: `just validate` green; a read of both READMEs of
  `skills/sdd` and `skills/engineering` confirms identical content per
  pair; `git diff --exit-code .claude-plugin/marketplace.json` after
  `just gen-marketplace`; `ls -l .agents/skills` shows the retargeted and
  the two new relative links; `just spec-sync && git status --porcelain
  .agents/skills` empty.
- Spec domain move: `just spec-validate` green; `git log --follow --oneline
  openspec/specs/sdd/spec-driven-development/spec.md` lists the pre-move
  commits.
- Workflows: YAML parse of `checks.yml`, `spec-command.yml` and
  `spec-labels.yml`; `grep -rn '{{[A-Z]' .github/` empty; actions pinned by
  SHA (`grep -c '@[0-9a-f]\{40\}'` per file ≥ the `uses:` count); no
  `spec / *` workflow contains a `git push`, a `git fetch` of the head, or a
  checkout of it; `.github/workflows/spec-archive.yml` no longer exists.
- Script and recipes: mirror diff empty; `just spec-changes --help` exits
  0; `just spec-check` on this branch before archiving reports the two
  changes as warnings (draft) and exits 0, and as failures with a ready
  flag; after archiving exits 0; `just lint` green.
- Labels and validator: `just validate` green; the `sync_labels.py` dry
  run lists exactly `catalog/sdd` and the five `spec/*` labels to create;
  `--apply` is not run; a scratch copy of `labels.json` with a
  `spec/extra` label `applied_by: workflow` makes `just validate` fail.
- Contract and project skill: a clean-context subagent reads
  `.agents/knowledge/spec-workflow.md` and `change-workflow/SKILL.md`
  and answers: what the draft's first content is, what the maintainer
  reviews, when the request is marked ready, who archives and when, and
  what the archive freezes.
- Templates and schema: `python3 scripts/check_pr_policy.py` over a
  ready-shaped body built from the new template passes and over a draft
  body with reserved lines passes; `just spec-validate` accepts the schema
  edit.
- Knowledge and maps: `just validate` green (pointers and recipe names);
  `grep -rn 'spec-archive / archive\|archive_completed_changes\|spec-archive-completed' . --exclude-dir=.git --exclude-dir=archive` empty.
- `just check` at the end.

### The privileged read path

- `.github/workflows/spec-labels.yml` and `spec-command.yml` parse; their
  job names (`spec / labels`, `spec / command`) and permissions are
  unchanged, so the checks table and `check_spec_labels()` still agree.
- `grep -n 'git .*fetch'` finds nothing in this repository's own
  `spec / *` workflows.
- `diff scripts/spec_changes.py skills/sdd/openspec-workflow/scripts/spec_changes.py`
  is empty and `just validate` is green.
- `just spec-check origin/main HEAD` and `just spec-changes check --all`
  behave as before on this branch (draft warns, ready fails, `--all`
  fails while the records are open).

## Open Questions

None.
