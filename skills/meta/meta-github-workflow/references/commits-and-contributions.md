# Commits and Contribution Flow

Read when the harness defines commit format, branch naming, merge method,
squash behavior, merge queue participation, or commit enforcement.

## Ground the convention in evidence

Run `scripts/analyze_history.py` first and design
from what the history already does. Adopt a structured convention
(Conventional Commits or similar) only when something consumes the
structure — release automation, changelog generation, or commit-driven
labeling; otherwise a subject-length and mood rule is a complete
convention. Preserve an existing working convention.

## Merge method decides what is validated

Read the allowed merge methods from the capability probe before writing any
rule:

- **Squash merge** makes the pull-request title the commit subject on the
  default branch — individual branch commits never land there. Validate the
  PR title, not the branch history, and say so in CONTRIBUTING: drive-by
  contributors are not punished for messy branch commits.
- **Merge commits / rebase** land the branch commits: validate the PR's
  commit range (`BASE..HEAD`, which needs `fetch-depth: 0` in CI — a
  shallow checkout silently validates nothing). Exempt merge and revert
  subjects.
- Never validate history behind the default branch; the past is not
  repairable.

## Branches and claiming

Branch naming rides on the native mechanism: `gh issue develop` creates an
issue-linked branch named `<number>-<slug>`; adopt that shape as the
default rather than inventing a scheme, and record any project-specific
prefix on top of it. The full claim procedure lives in
[issues-and-prs.md](issues-and-prs.md).

## Enforcement

Ship `assets/check_commits.py` into the target's
`scripts/` with its `CONFIG` block edited to the agreed convention, wired
as: a local hook or documented command, and the commit-check workflow
running the same script — title mode under squash, range mode otherwise.
The convention document, the `CONFIG` block, and CONTRIBUTING must state
the same rule; register the pair in the synchronization table.

## Merge queue

Decide merge queue before writing workflows: every required-check workflow
must add the `merge_group` event or queued merges hang, wildcard branch
patterns are unsupported, and draft PRs never enter the queue. Its
documented ruleset support is ambiguous — verify against the live
repository before designing on it. Auto-merge (`gh pr merge --auto`) is the
lighter default for solo and small-team repositories: it needs a blocking
protection to exist, and it disarms itself when the base changes or an
outsider pushes.

Done when: the convention document, the validator CONFIG, CI, the merge
setting, and CONTRIBUTING state one identical rule, and the PR-title-versus-
range choice matches the repository's actual merge method.
