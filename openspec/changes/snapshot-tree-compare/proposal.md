## Why

A pull request that touches more than 3000 files, such as a vendoring commit or a mass rename, gets no `spec/*` labels and no answer to `/spec show` or `/spec status`. The deposited management scripts build their snapshot from GitHub's "list pull request files" endpoint, which returns at most 3000 files however it is paged, and since pull request #95 the snapshot refuses a listing shorter than the request. That refusal is correct, but the snapshot needs only which change or feature directories the request touches, not every path. Issue #101 asks for that answer on requests of any size.

## What Changes

- `openspec-workflow`, `assets/spec_changes.py` (the script deposited as the project's `scripts/spec_changes.py`):
  - `snapshot` no longer reads the request's file list. It reads the merge base of the request's base and head from the platform's comparison of the two commits. A change is related when its directory under the changes directory or its archive has a different tree SHA at the head than at that merge base, or is a directory on one side only. This is the set the git source's `git diff base...head` gives, and it no longer depends on how many files the request touches.
  - A directory changed only on the base branch after the request branched is not related, because the comparison runs against the merge base and not against the base tip.
  - The base tip is still read for the `at_base` state that `status --json` reports, so the output stays identical to the git source's.
  - **BREAKING** for a caller of the deposited script: the `--max-files` option and its cap go, and the snapshot document becomes `spec-snapshot/2`. The touched directories replace `changed_paths`, `head.sha` keeps its key and meaning, and a `spec-snapshot/1` document is refused with a message to rebuild it. No delivered workflow passes `--max-files`, and every workflow builds and reads its snapshot in the same job.
  - A comparison that answers without a merge base fails naming the endpoint, like every other unexpected response.
- `spec-kit-workflow`, `assets/spec_kit_features.py` (deposited as `scripts/spec_kit_features.py`): the same for numbered feature directories under the specs directory, with the snapshot document becoming `spec-kit-snapshot/2`. The snapshot now also lists the specs directory at the merge base, since a feature deleted by the request exists only there.
- Both skills' `SKILL.md` fork-safety paragraph and `references/github.md` `## Fork safety` stop saying the snapshot pulls the file list, and the cap bullet no longer lists "files touched" among the caps.

## Skills touched

- `sdd/openspec-workflow` (modified): the `Script: assets/spec_changes.py` requirement — related changes by tree comparison against the merge base, no file-count cap, the schema bump.
- `sdd/spec-kit-workflow` (modified): the `Script: assets/spec_kit_features.py` requirement — the same for touched features.

## Installed behavior

A project that installs the automation after this change gets labels and `/spec` replies on requests of any size, where before the label job and the comment command failed on a request of more than 3000 files. An agent explaining the automation describes the snapshot as reading which directories differ from the merge base, not as pulling the file list. This adds a capability → `feat`, as issue #101 is filed.

## Impact

- Companion repository change `snapshot-tree-compare-harness`: this repository's own `scripts/spec_changes.py`, whose snapshot code is the asset's line for line and whose CLI `.agents/knowledge/harness-maintenance.md` registers against the asset's, and the prose in `.agents/knowledge/github-checks.md` that says the snapshot pulls the file list.
- No workflow asset changes: none passes `--max-files`, and the `contents: read` the comparison, trees, and blobs need is already granted. The GitLab assets read the head with git and are untouched.
- No description, README pair row, symlink, `marketplace.json` entry, or catalog `CONTEXT.md` changes.

## Non-goals

- The skills' bundled `scripts/spec_changes.py` and `scripts/spec_kit_features.py`. Their older `snapshot` stops at 3000 files without an error, and no delivered workflow calls it any more. That defect goes to a separate bug, following the `management-code` non-goal for public skills' own `scripts/`.
- The merge gate (`checks / spec`) and the GitLab jobs, which read the head with git and have no file cap.
- The workflow permission comment "reading the request and its files is a pull-request read" in both `assets/github/workflow-spec-command.yml`. It goes stale with the file list, and the change for issues #96 and #100, which rewrites that step, owns its wording.

## Tracked work

Issue #101.
