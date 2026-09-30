## Why

This repository's `spec / labels` and `spec / command` run its own `scripts/spec_changes.py snapshot`. Its snapshot code is the `openspec-workflow` asset's, line for line, so a pull request here that touches more than 3000 files also gets no `spec/*` labels and no `/spec` reply. The `snapshot-tree-compare` change fixes the asset. `.agents/knowledge/harness-maintenance.md` registers the asset's command line against this script, so the script follows the same change.

## What Changes

- **`scripts/spec_changes.py` `snapshot`.**
  - It finds the related changes by comparing the tree SHA of each directory under `openspec/changes/` and `openspec/changes/archive/` between the head and the merge base. The merge base is the `merge_base_commit` of the platform's comparison of the pull request's base and head. It no longer reads the pull request's file list.
  - A directory changed only on `main` after the branch left it is not related.
  - The base tip is still listed for the `at_base` state.
  - `--max-files` and `MAX_FILES` go.
  - The document becomes `spec-snapshot/2`: the touched directories replace `changed_paths`, and `head.sha` keeps its key and meaning.
  - The other subcommands, `archive` included, keep their behavior. `archive` accepts only the git source, which is unchanged.
- **`.agents/knowledge/github-checks.md`.** The privileged-jobs paragraph (item 5, "pulls the file list and the documents from the REST API") says that the snapshot reads which change directories differ from the merge base, plus the documents.
- **Unchanged, by reading:** `.github/workflows/spec-labels.yml` and `spec-command.yml` pass no cap flag and already grant `contents: read`. `scripts/validate_harness.py` reads only `labels --taxonomy`, whose output does not change.

## Skills touched

Repository change.

## Installed behavior

No installed skill changes. Agents working in this repository see `spec/*` labels and `/spec` replies on pull requests of any size, where before a pull request of more than 3000 files failed both jobs. The knowledge file describes the snapshot as it works.

## Impact

- `scripts/spec_changes.py`, `.agents/knowledge/github-checks.md`.
- The `harness-maintenance.md` row for the sdd workflow assets stays as written. The asset and this script keep the same subcommands, flags, and output.
- `spec / command` and `spec / labels` run from `main`, so their live behavior changes only after the merge.

## Non-goals

- The permission comment in `.github/workflows/spec-command.yml`, "reading the request and its files is a pull-request read". The change for issues #96 and #100 rewrites that step and owns its wording.
- `checks / spec`, which reads the head with git and has no file cap.

## Tracked work

Issue #101.
