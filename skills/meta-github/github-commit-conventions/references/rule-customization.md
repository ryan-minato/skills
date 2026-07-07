# Customizing the rules

Load condition: tailoring the convention beyond the shipped defaults, or
the repository squash-merges.

Every enforced rule lives in the `CONFIG` block at the top of the
committed `scripts/check_commits.py`; the convention doc describes the
same rules for humans. Change both in the same commit.

## Scope rules

`CONFIG["allowed_scopes"]` has three shapes:

| Value | Meaning | Convention-doc wording |
|---|---|---|
| `None` | Scope optional, free-form | "Scope is optional; use the touched module's name." |
| `[]` | Scope forbidden | "This project does not use scopes." |
| `["api", "cli", ...]` | Scope required, from this set | List the set and what each covers |

Monorepos usually want the required-set shape with one scope per
package; keep the set short enough to memorize, or derive it from the
top-level directory names and say so in the doc.

## Custom types

Add project-specific types to `CONFIG["types"]` sparingly — every extra
type dilutes the signal. A type earns its place when the history
analyzer shows a recurring change kind the standard eleven cannot
express. Update the convention doc's table in the same commit.

## Required trailers

To require a trailer (for example an issue reference on every commit),
add a check to `check_message` in the validator — a compiled regex over
the message body — and document it. Keep the exemption patterns in
mind: merge and revert commits will not carry trailers.

## Squash-merge repositories

When the repository squash-merges, the PR **title** becomes the squash
commit's subject on the default branch, and individual branch commits
disappear. Two adjustments:

1. In the workflow, validate the PR title instead of (or in addition
   to) the range — replace the run step with:

   ```yaml
   - name: Validate the PR title
     env:
       PR_TITLE: ${{ github.event.pull_request.title }}
     run: python3 scripts/check_commits.py --message "$PR_TITLE"
   ```

   (Passing the title through `env:` avoids shell-injection via crafted
   titles; never interpolate it directly into the script line.)
2. Relax the branch-commit rules in the convention doc: branch commits
   may be informal when they are squashed away, and say which applies.

## Commit template

Ship a `.gitmessage` template so humans get the format as a scaffold:
put the title format and type list in a comment block, commit the file
at the repo root, and document
`git config commit.template .gitmessage` as a one-time setup step (or
run it via the project's bootstrap script).
