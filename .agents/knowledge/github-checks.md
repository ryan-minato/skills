# GitHub Checks

Read this before adding, renaming, removing, or debugging a GitHub Actions
job, and before changing the ruleset's required checks. Every job runs a
command that exists and passes locally; CI never chooses its own linters.

| Check | Workflow | Command | Runs on | Tier | Healthy run |
|---|---|---|---|---|---|
| `checks / quality` | `checks.yml` | `just install-tools && just check` (Python 3.12, Node 24, pinned pre-commit, rust-just, ruff) | every pull request; push to `main` | feeds the gate | `just check` exits 0: validators, lint, spec validation, pre-commit hooks (commit-safety skipped by design). |
| `checks / spec` | `checks.yml` | `just spec-validate`, then `just spec-sync` + `git diff --exit-code -- .agents/skills` | pull requests that touch `openspec/`, `.agents/skills/openspec-*`, `.agents/skills/.openspec-target`, or `justfile`; every push to `main` | feeds the gate | Validation exits 0 and the regenerated OpenSpec skills are identical to the committed ones. On an untouched pull request the job logs "nothing to validate" and passes. |
| `checks / gate` | `checks.yml` | aggregates the two jobs above (`if: always()`) | every pull request; push to `main` | **required** | Green only when neither dependency failed or was cancelled; a skipped `checks / spec` is fine. |
| `pr / policy` | `pr-policy.yml` | `python3 scripts/check_pr_policy.py --event "$GITHUB_EVENT_PATH"` (template and script from the base commit) | pull request opened, edited, synchronized, reopened, ready, converted to draft | **required** | Body carries every template heading, the `Closes`/`N/A` line, the security checkbox line; ready pull requests also pass the Validation, checklist, `Spec:`, and commit-subject checks. |
| `scan-secrets` | `secret.yml` | TruffleHog `--only-verified` over the pull request range or the pushed range | every pull request; push to `main` | **required** | No verified secret. The job id is the check name; it predates the other checks. |
| `issues / triage` | `issue-triage.yml` | `python3 scripts/sync_issue_metadata.py --event "$GITHUB_EVENT_PATH" --apply` | issue opened or edited | automation | Prints the label plan as JSON and applies it; exit 1 means a form answer did not map to a label in `.github/labels.json`. |

Required-check names are a stable interface: `scripts/validate_harness.py`
fails when a workflow job name and this table disagree, and the ruleset
(`.agents/knowledge/github-settings.md`) names `checks / gate`,
`pr / policy`, and `scan-secrets`. Renaming one is three coordinated edits:
the workflow, this table, and the ruleset — in that order, with the new
name live on `main` before the ruleset requires it.

## Diagnosing a red check

1. Reproduce locally with the command in the table; `just check` is
   byte-for-byte what `checks / quality` runs.
2. For a failure you cannot reproduce, digest the run instead of reading
   full logs:
   `python3 .agents/skills/change-workflow/scripts/run_log_digest.py --repo ryan-minato/skills --run-id <id>`.
3. Never weaken, skip, or delete a check to make it pass; changing a
   check's strictness is a maintainer decision recorded here first.

## Tool pins

`checks / quality` installs the same versions the dev container and
pre-commit use: `ruff==0.16.4` (pre-commit rev `v0.16.4`), `rust-just`,
`pre-commit`, `pyyaml` and `@fission-ai/openspec` from the `justfile`
variables. `scripts/validate_harness.py` checks that the ruff pins agree in
all three places. Bump them together; see
`.agents/knowledge/harness-maintenance.md`.

## Update this file when

- A job is added, renamed, or removed, or its command changes.
- A required check changes in the ruleset.
- A tool pin changes.
