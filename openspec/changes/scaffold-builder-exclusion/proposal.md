## Why

`scaffold-colab` and `scaffold-data-science` state the disposable-builder exclusion outside their workflows: in `scaffold-colab` it is a loose bullet after the completion criterion, in `scaffold-data-science` the last bullet of `## Invariants`. It is a procedure with an order ("before the first commit"), so an agent following the numbered steps can reach the first commit before it reads it. `scaffold-ml` moved the same sentence into its workflow in `ml-standard-alignment`; the two sibling scaffolds should read the same way.

## What Changes

- `scaffold-colab` (no observable behavior change):
  - the exclusion procedure moves into workflow step 1, where the builder confirms the repository state and before step 2 writes the first file; in a directory that is not yet a repository, the step says to run it right after the repository exists and before the first commit. The loose bullet goes.
  - Gotchas keep the two facts no earlier step states: the local image is not real Colab, and `# @param` forms beat ipywidgets. The pip-cascade bullet and the regional-registry bullet go, because step 3's dependency reference and step 4 already state them.
  - the broken line wrap in step 8 ("record the gap in / the handoff") is rejoined.
- `scaffold-data-science` (no observable behavior change):
  - the exclusion procedure moves into workflow step 2, bound to the build's first commit on both of its branches: in a new package after `uv init --package` and before `uv.lock` is committed, in an existing project before any commit. The `## Invariants` bullet and the blank line above it go.
  - Gotchas keep the branch-or-`latest` identity fact, trim the secret-scanner bullet to its fact (the review procedure stays in the `AGENTS.md` that step 3 deposits for the target project's agents), and drop the template bullet, which step 3, the completion criterion, and the bundled validator already carry.
  - the mis-indented line of the run-record invariant and the broken wrap in step 10 are fixed.

## Skills touched

- `scaffold/scaffold-colab` (new): Behavior — disposable builders stay out of every commit, the exclusion written once the target is a git repository and before the build's first commit, including a directory that is not yet a repository.
- `scaffold/scaffold-data-science` (new): the same, for an existing project and for a target directory that becomes a repository only when the package is initialized.

## Installed behavior

Neither skill behaves differently: the exclusion procedure already existed with the same content and the same bound ("before the first commit"); only its place in each skill moves, worded so that the command still runs only once a repository exists, and the Gotchas edits remove restatements. The new domains pin the one behavior whose placement moves, so that a later change cannot drop it from the workflow unnoticed. Commit type: `refactor`.

## Impact

- No catalog `README.md` or `README.zh.md` row, symlink, `marketplace.json` entry, catalog `CONTEXT.md`, description, reference, asset, or script changes.
- No mirrored file: `.agents/knowledge/harness-maintenance.md` registers only `scaffold-ml` mirrors, and `scripts/validate_harness.py` pairs neither skill.
- At archive, the two domains appear under `openspec/specs/scaffold/` beside `scaffold-ml`.

## Non-goals

- `scaffold-ml`, whose step 1 already carries the procedure. That it runs the command before a repository may exist is a proposed follow-up.
- The identical rule bullets of the `meta` builders.
- Any Gotchas content beyond the review above, and any wording outside the moved procedure, the Gotchas, and the two whitespace defects.
- `Trigger:`, `Handoff:`, or `Script:` requirements for the new domains: the descriptions, handoffs, and scripts do not change, and specs are never backfilled.

## Tracked work

Issue #87.
