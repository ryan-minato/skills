## Context

See proposal.md. Catalogs are enumerated in `ARCHITECTURE.md` (`## Catalogs`,
checked two-way by `check_architecture_md_catalog_list()` in
`scripts/validate_skills.py`), the root `README.md` and `README.zh.md`
catalog table and plugin-install example, `.github/labels.json` (`catalog/*`
compared with the directories in `skills/` by `check_labels()` in
`scripts/validate_harness.py`, which also requires every issue form's
Catalog dropdown to list the same names plus `repository`), and
`.claude-plugin/marketplace.json` (one plugin per non-empty catalog;
`check_marketplace_manifest()` errors both on a non-empty catalog without
an entry and on an entry whose `skills[]` drifts). Every catalog directory
needs `README.md`, `README.zh.md`, and `CONTEXT.md` (`check_catalogs()`).
`.agents/knowledge/harness-maintenance.md` lists these in its "a catalog
added or removed" row and requires the entropy review after any catalog
change. Precedent: commits `184ff7b` (scaffold catalog) and `921cfe4`
(its labels and forms). The new catalog is durable, so
`CATALOG_NAME_PREFIXES` and `DISPOSABLE_CATALOGS` stay unchanged.

## Placement

| What Changes bullet | File | Check that proves it |
|---|---|---|
| Catalog scaffold | `skills/machine-learning/README.md`, `README.zh.md`, `CONTEXT.md` | `just validate` (`check_catalogs`); a read of both READMEs for identical content |
| Architecture map | `ARCHITECTURE.md` `## Catalogs` bullet | `just validate` (`check_architecture_md_catalog_list`) |
| Root README pair | `README.md`, `README.zh.md` catalog table rows and the `/plugin install` example comment | read-through of both files |
| Labels and issue forms | `.github/labels.json`; `.github/ISSUE_TEMPLATE/bug-report.yml`, `feature-request.yml`, `task.yml` Catalog options | `just validate` (`check_labels`); `python3 scripts/sync_labels.py --file .github/labels.json --repo ryan-minato/skills` dry run listing the one label to create |
| Marketplace entry | `.claude-plugin/marketplace.json` `machine-learning` plugin | `just gen-marketplace` reports no drift; `just validate` (`check_marketplace_manifest`) |
| Symlinks | `.agents/skills/<name>` for the five skills | `just validate` (`check_symlinks`) |
| Entropy review | `.agents/knowledge/harness-maintenance.md` last-run date | read-through; `just validate` (`validate_harness.py` register checks) |

## External impact

- The skill change `machine-learning-catalog` on the same branch supplies
  the five skill directories the marketplace entry and the symlinks refer
  to; the marketplace entry lands in the same commit as the first skill.
- `scripts/validate_skills.py` and `scripts/validate_harness.py` are not
  edited; proof `git diff --stat origin/main...HEAD -- scripts` is empty.
- The remote label is created after the merge by the maintainer with
  `scripts/sync_labels.py --apply`; recorded as a maintainer action in the
  pull request handover.

## Decisions

- **No name prefix and no disposable marker** (serves the catalog scaffold):
  the skills are durable and installed one at a time, like `engineering`
  and `writing`; the prefix mechanism exists to group disposable builders.
- **`CONTEXT.md` grants `core` only and no sibling grant**: siblings are
  optional handoffs by role with fallbacks, so any one skill installs and
  works alone; the alternative (install the catalog whole, like `meta`) was
  rejected because the five skills serve different moments of a project.
- **Marketplace entry in the first skill's commit, not the catalog
  commit**: an empty catalog must carry no entry and a non-empty one must;
  the precedent commits kept the catalog empty and entry-less.
- **Root README example comment lists the new catalog** so the
  `/plugin install` line stays an accurate enumeration.

## Risks / Trade-offs

- [The validators read the whole tree at commit time, so a skill directory
  staged without its symlink, README rows, or marketplace entry fails the
  hook] → each skill lands in one commit with its symlink, its README pair
  rows, and the regenerated marketplace; unfinished skill directories are
  moved out of the tree before committing another.
- [Label sync needs the remote] → recorded as a maintainer action after
  merge; the dry run proves the file.

## Verification plan

- Catalog scaffold: `just validate` green; `diff <(sed -n '/^|/p'
  skills/machine-learning/README.md | cut -d'|' -f2)` against the zh file
  shows the same skill names; a read of both confirms identical content.
- Architecture map: `just validate` green with the bullet; removing the
  bullet in a scratch copy is not tested.
- Root README pair: read-through of the two catalog tables and the install
  example.
- Labels and forms: `just validate` green; `python3 scripts/sync_labels.py
  --file .github/labels.json --repo ryan-minato/skills` (dry run) lists
  exactly `catalog/machine-learning` to create; `--apply` is not run.
- Marketplace: `just gen-marketplace` then `git diff --exit-code
  .claude-plugin/marketplace.json` empty; `just validate` green.
- Symlinks: `ls -l .agents/skills | grep machine-learning` shows five
  relative links; `just validate` green.
- Entropy review: the register's last-run line carries today's date and
  the review's findings (or "none") are in the pull request.
- `just check` at the end.

## Open Questions

None.
