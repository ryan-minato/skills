## Context

See proposal.md for motivation. `openspec/changes/split-shape-default-branch-check/design.md` covers the skill this change mirrors.

**Current state** (`origin/main` at 10e29c0):
- **`checks.yml`.** The `checks / spec` check step runs `just spec-check "$BASE" "$HEAD" $flag` on a pull request and `just spec-changes check --all` on a push (line 101). Its comment (lines 85–89) says a push fails on any change outside `archive/`. `just spec-check` pins `--shape combined` (`justfile` line 52); `spec-changes` passes its arguments through (lines 55–56). The push step therefore relies on the script's default, `combined`.
- **`scripts/spec_changes.py`.** Already warns under `--all --shape split` and fails under the default (lines 811–825). Its `--all` help (line 1014) says only "fail on any change outside archive/".
- **Knowledge.**
  - `.agents/knowledge/github-checks.md` line 10 shows the push command without a shape, and says "the recipe pins `--shape combined`".
  - `.agents/knowledge/spec-workflow.md` lines 158–162 say the `--shape combined` flag in the `spec-check` recipe states the combined shape explicitly; lines 224–227 show the push command without a shape.
  - `.agents/knowledge/harness-maintenance.md` line 30 names only "the `--shape combined` flag of the `spec-check` recipe". Line 31 registers `job-spec-check.yml` against the `checks / spec` job, and the asset's command line against `scripts/spec_changes.py` ("the same subcommands, flags, and output").

**Binding rules:**
- **Commits.** Every commit passes `just check` on its own, because branches are rebase-merged.
- **Push-only step.** A pull request never runs the push branch of `checks / spec`, so it is exercised locally and read back after the merge.
- **Validator.** `scripts/validate_harness.py` compares workflow job names with `github-checks.md`; no job name changes.
- **Management code.** This repository's `scripts/spec_changes.py` is its own. The register asks for the same flags and output as the asset's command line, not for identical files.

## Placement

| What Changes bullet | File and section | Proof |
|---|---|---|
| Push step pinned | `.github/workflows/checks.yml`, job `spec`, step "Validate strictly and refuse an unarchived related change": the `else` branch and the step comment | local step run and YAML parse below; `just validate` |
| Help text | `scripts/spec_changes.py`, the `check` parser's `--all` argument | `check --help` below |
| Knowledge | `.agents/knowledge/github-checks.md` `checks / spec` row (Command and Healthy run columns); `.agents/knowledge/spec-workflow.md` `## Archive executor and the freeze` (the combined-shape paragraph) and `## Verification in checks`; `.agents/knowledge/harness-maintenance.md` the `scripts/spec_changes.py` row | readback against `checks.yml` and the `justfile`; `just validate` |

## Decisions

- **The companion is required, not optional** (every bullet; maintainer decision). The register row for the job asset binds this repository's `checks / spec` job and script to the asset's steps and command line. Leaving them would make the register false the day the skill change lands.
  - Rejected: relying on the script's default. The verdict would be the same, but the push step would differ from the asset it mirrors, and `spec-workflow.md` says the shape is stated explicitly.
- **Pin inline through `spec-changes`** (push step bullet).
  - One flag on the existing passthrough recipe, the same form the asset uses.
  - Rejected: a new recipe for the push check. It adds a recipe, a table row, and a register entry for one line, and splits the check across three recipes.
- **Help text follows the asset** (help text bullet). The register asks for the same flags and output. The wording is written for this script and may match the asset's; nothing requires the files to be identical.
- **`## Request automation` stays** (non-goal). Its sentence describes this repository, which is combined-shape, and stays true.

## Risks / Trade-offs

- **[The push branch cannot run in this pull request]** → The step body runs locally, as the workflow runs it, on both sides of the archive commit (below). After the merge, the next push run on `main` is read back.
- **[A typo in the pinned flag turns every push red]** → The local run uses the exact line from `checks.yml`, and argparse rejects an unknown shape with exit 2.
- **[The skill change's MODIFIED blocks are re-copied at rebase]** → This change has no spec. It moves with the skill change's branch and archive, and its verification reruns after any rebase.

## Verification plan

Per What Changes bullet. Every command runs in this branch's worktree, where the `justfile` and `scripts/spec_changes.py` resolve; no scratch repository is needed.

- **Push step pinned.**
  - `checks.yml` parses with PyYAML; `actionlint` runs on it when available, and its absence is recorded.
  - The step's `else` line, copied from the file, runs with `EVENT=push` in the worktree:
    - before the archive commit, it exits 1 naming both `split-shape-default-branch-check` and `split-shape-default-branch-check-harness`, which proves the pin still fails under combined;
    - at the archive commit, it exits 0.
  - Before the archive commit, the same line with `--shape split` in place of `combined` runs in the worktree and exits 0 with warnings naming both changes. Beside the combined run, that shows the flag reaches the script through the recipe.
  - `just validate` passes: the job names still match `github-checks.md`.
- **Help text.** `python3 scripts/spec_changes.py check --help` exits 0, and its `--all` line names both shapes. A readback compares it with the asset's `check --help`: same flag, same meaning. `python3 scripts/spec_changes.py --bogus` still exits 2.
- **Knowledge.** A readback of each edited passage against `checks.yml` line by line and the `justfile` `spec-check` and `spec-changes` recipes: the commands quoted are the commands run, and each names `--shape combined`. `grep -rn 'check --all' .agents .github justfile ARCHITECTURE.md` shows no unpinned push command.
- **Everything.** `just check`, and `git diff --stat origin/main...HEAD`.
