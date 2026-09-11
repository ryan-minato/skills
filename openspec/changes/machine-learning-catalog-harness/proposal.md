## Why

The skill change `machine-learning-catalog` on this branch adds a sixth
public catalog. This repository enumerates its catalogs in files no spec
covers — the architecture map, the root README pair, the label set and the
issue forms' Catalog options, the plugin marketplace — and its validators
fail when any of them disagrees with `skills/`. They must follow in the
same pull request, and the register that lists what a catalog change
touches names the entropy review that runs afterwards.

## What Changes

- Catalog scaffold: `skills/machine-learning/README.md`, `README.zh.md`
  (content-identical), and `CONTEXT.md` (durable per-project skills, `core`
  only as dependencies, siblings named by role, default naming shape, the
  catalog's scope and disambiguation).
- `ARCHITECTURE.md` `## Catalogs`: a `machine-learning` bullet, so
  `check_architecture_md_catalog_list()` agrees with `skills/`.
- Root `README.md` and `README.zh.md`: a catalog table row and the plugin
  install example naming the new catalog.
- `.github/labels.json`: `catalog/machine-learning` (`applied_by: triage`,
  the catalog color); the Catalog dropdown of `bug-report.yml`,
  `feature-request.yml`, and `task.yml` gains `machine-learning`, so
  `check_labels()` passes. Applying the label on GitHub with
  `scripts/sync_labels.py --apply` after the merge is a maintainer action.
- `.claude-plugin/marketplace.json`: a hand-added `machine-learning`
  plugin entry with a description, then `just gen-marketplace`; the entry
  lands with the first skill because an empty catalog may carry none.
- `.agents/skills/`: one relative symlink per new skill.
- `.agents/knowledge/harness-maintenance.md`: the entropy review's last-run
  date after the review that a catalog change triggers.

## Skills touched

Repository change (`skip_specs: true`): no public skill domain.

## Installed behavior

Agents working in this repository find the sixth catalog wherever the
harness lists catalogs — the architecture map, the READMEs, the labels
and issue forms, the marketplace — and `just check` passes with it.

## Impact

- Added: `skills/machine-learning/README.md`, `README.zh.md`,
  `CONTEXT.md`; five symlinks under `.agents/skills/`.
- Edited: `ARCHITECTURE.md`, `README.md`, `README.zh.md`,
  `.github/labels.json`, `.github/ISSUE_TEMPLATE/bug-report.yml`,
  `.github/ISSUE_TEMPLATE/feature-request.yml`,
  `.github/ISSUE_TEMPLATE/task.yml`, `.claude-plugin/marketplace.json`,
  `.agents/knowledge/harness-maintenance.md`.
- Unchanged: `scripts/validate_skills.py` (`CATALOG_NAME_PREFIXES` and
  `DISPOSABLE_CATALOGS` stay as they are — the catalog reserves no prefix
  and is not disposable), the validators, the workflows, the rulesets.

## Non-goals

- A name prefix for the catalog or a disposable marker on its skills.
- The remote label sync itself (maintainer action after merge).
- Any change to `scaffold/CONTEXT.md` or the `meta` catalog: the follow-up
  change `ml-standard-alignment` and its own harness companion.

## Tracked work

No issue: companion of `machine-learning-catalog`, planned in
conversation.
