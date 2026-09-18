# GitHub Checks

Read this before adding, renaming, removing, or debugging a GitHub Actions
job, and before changing the ruleset's required checks. Every job runs a
command that exists and passes locally; CI never chooses its own linters.

| Check | Workflow | Command | Runs on | Tier | Healthy run |
|---|---|---|---|---|---|
| `checks / quality` | `checks.yml` | `just install-tools && just check` (Python 3.12, Node 24, pinned pre-commit, rust-just, ruff) | every pull request; push to `main` | feeds the gate | `just check` exits 0: validators, lint, spec validation, pre-commit hooks (commit-safety skipped by design). |
| `checks / spec` | `checks.yml` | `just spec-check <base> <head> [--draft]` on a pull request, `just spec-changes check --all` on a push; then `just spec-sync` + `git diff --exit-code -- .agents/skills` | pull requests (opened, synchronized, reopened, ready, converted to draft) that touch `openspec/`, `.agents/skills/openspec-*`, `.agents/skills/.openspec-target`, or `justfile`; every push to `main` | feeds the gate | Strict validation exits 0; a related change still unarchived is a warning on a draft and a failure on a ready pull request; on `main` no change sits outside `openspec/changes/archive/`; the regenerated OpenSpec skills are identical to the committed ones. On an untouched pull request the job logs "nothing to validate" and passes. |
| `checks / gate` | `checks.yml` | aggregates the two jobs above (`if: always()`) | every pull request; push to `main` | **required** | Green only when neither dependency failed or was cancelled; a skipped `checks / spec` is fine. |
| `pr / policy` | `pr-policy.yml` | `python3 scripts/check_pr_policy.py --event "$GITHUB_EVENT_PATH"` (template and script from the base commit) | pull request opened, edited, synchronized, reopened, ready, converted to draft | **required** | Body carries every template heading, the `Closes`/`N/A` line, and the security checkbox line, and every commit subject (the title, for a fork) follows Conventional Commits; ready pull requests also have Changes and Validation filled (no reserved line, not empty), `Phase: implementation`, every checklist box ticked, and a `Spec:` line (a link to the change directory; a bare URL or path is still accepted). |
| `scan-secrets` | `secret.yml` | TruffleHog `--only-verified` over the pull request range or the pushed range | every pull request; push to `main` | **required** | No verified secret. The job id is the check name; it predates the other checks. |
| `spec / command` | `spec-command.yml` | `python3 scripts/spec_changes.py snapshot …` then `show|status --snapshot …` from the default branch, replying with `gh pr comment` | a pull request comment starting with `/spec` by a collaborator — owner, member, or collaborator association (`issue_comment`) | automation | One reply comment per command (`/spec show [<change>] [proposal|design|tasks|specs|all]`, `/spec status [<change>]`, `/spec help`); the run is skipped for other comments and for bots. |
| `spec / labels` | `spec-labels.yml` | `python3 scripts/spec_changes.py snapshot …` then `labels --snapshot …`, then `gh api` label writes checked against the literal taxonomy | pull request opened, synchronized, reopened, or ready (`pull_request_target`) | automation | The plan JSON is printed and the `spec/archived`/`spec/unarchived` and `spec/not-started`/`spec/in-progress`/`spec/done` labels match it; a pull request with no related change carries none. |
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
4. A red `checks / spec` on a ready pull request whose change is not
   archived yet is the expected state, not a failure: the record is
   frozen only after the maintainer closes the deliberation on the
   finished implementation, and that red is what blocks the merge until
   then (`.agents/knowledge/spec-workflow.md`).
5. The two `spec / *` jobs run with a privileged token on content the
   request's author controls. The rule that keeps them safe is structural:
   no object authored by the request reaches the runner. They check out
   the base and read the head through `spec_changes.py snapshot`, which
   pulls the file list and the documents from the REST API. Neither needs
   the head's working tree, and neither may grant `contents: write`.
   Never add a checkout or a `git fetch` of the head to either, and if
   one ever genuinely needs it, guard it with a literal
   `github.event.pull_request.head.repo.full_name == github.repository`
   in the step's own `if:`, never behind an `env` variable.

## Tool pins

`checks / quality` and `checks / spec` install the same versions
the dev container and pre-commit use: `ruff==0.16.4` (pre-commit rev
`v0.16.4`), `rust-just` and `pre-commit` pinned in the workflows, and
`@fission-ai/openspec` at the `openspec_version` of the `justfile`
(`just install-tools`). `scripts/validate_harness.py` checks that the ruff
pins agree in all three places. Bump them together; see
`.agents/knowledge/harness-maintenance.md`.

## Update this file when

- A job is added, renamed, or removed, or its command changes.
- A required check changes in the ruleset.
- A tool pin changes.
