## Why

The `management-code` change makes the harness-producing skills deliver management code that is readable, fails fast, and belongs to the project. This repository does the opposite for its own management code. `scripts/validate_harness.py` requires `scripts/spec_changes.py` and `scripts/sync_labels.py` to stay byte-identical to the skills' bundled scripts, so both carry a product script's defensive style. In the spec job, a failing `git diff` inside an `if` reads as "no spec paths changed". Under `pipefail`, the gate's `printf | grep -q` could pass a failed job. And no knowledge file or review rule states what management code must look like.

## What Changes

- **Copies check removed.** `scripts/validate_harness.py` drops the `copies` check, its docstring entry, and the constants only it uses. This is a policy change the user decided in issue #94: a management script need not match a product script, and no check keeps the two identical; they may still match where the role needs the same code. It is not a check weakened to make a change pass.
- **Synchronization register.** In `.agents/knowledge/harness-maintenance.md`:
  - the `spec_changes.py` row names this repository's own script as the source that the archive executor and the freeze in `spec-workflow.md` follow;
  - the workflow-assets row states that this repository's workflows call their own script at a different path, and registers the pair a reader can no longer assume: the asset's CI-facing command line and this repository's script.
- **`scripts/spec_changes.py` rewritten for readability and fail-fast.**
  - It keeps every subcommand this repository calls: `snapshot`, `show`, `status`, `labels` (with `--taxonomy`), `check`, and `archive`. It drops `related`, which nothing calls.
  - Each external interface — the GitHub REST API, git plumbing, the snapshot file, the OpenSpec CLI — is checked once, and a failure there is reported at once with the interface named.
  - Fallbacks that read a failure as empty go, including a 404 on a tree the listing named.
  - A `/spec show` or `/spec status` naming a change the request does not touch exits 2, with the related changes listed and every name rendered as code.
  - It still reads a pull request's head only through the REST API, never with a fetch.
  - The prose that called this script the skill's mirror follows: `ARCHITECTURE.md`, `.agents/knowledge/spec-workflow.md`, the workflow header comments, and the `justfile` recipe comment.
- **`scripts/sync_labels.py` rewritten the same way.**
  - The `gh` interface stays checked: presence, exit status, parseable JSON, and a list shape.
  - `labels.json` type errors name the file and exit 1, instead of `str()` quietly accepting a wrong type.
  - The dry-run default, `--apply`, and `--prune` stay, and the plan it computes for this repository is unchanged.
- **Workflow shell.**
  - Every workflow with a `run:` step sets `defaults.run.shell: bash`.
  - The spec job decides the event first, then runs `git diff` on its own, so a failed diff fails the job.
  - The gate greps a here-string instead of a pipe.
  - `/spec` replies are posted even when the script fails. The run fails except on a bad argument.
  - The label loops read `jq` output that was produced before the loop, not through a process substitution.
- **Management-code rules.**
  - `.agents/knowledge/skill-quality.md` keeps its Scripts section for skills' `scripts/`, and gains a `## Management code` section, the rules' one source in this repository. It holds R1–R6, readability first with interface checks as fail-fast, and where the rules apply: scripts a skill deposits (`assets/`), this repository's `scripts/`, workflow shell, and project-skill scripts. It also records this repository's choice: stdlib-only scripts run with `python3` and no PEP 723 header, because CI installs no uv, until the first third-party dependency.
  - `.agents/skills/code-review/SKILL.md` links that section in place of "Scripts copied into target projects for CI follow the same calibration". It keeps the privileged-job threat model for request-authored content, states that exit 2 is guaranteed only for bad arguments and an unexpected crash exits 1, and drops the `sync_labels.py` copy from what `just check` enforces.
- **`skills/meta/CONTEXT.md`.** "Assets are raw starting shapes. Rework every line" gains its exception: script assets are working management code whose marked settings alone are configured, and the target owns the delivered copy.
- **Lint.**
  - `ruff.toml` adds `BLE001` and `S110` and exempts skills' own `scripts/` from them, which stay out of scope.
  - `just lint` and the pre-commit ruff hooks also cover the management assets under `skills/meta/*/assets` and `skills/sdd/*/assets`.
  - `AGENTS.md`, `ARCHITECTURE.md`, and the `ruff.toml` header describe the new scope.
- **Archive command.** `openspec/config.yaml` spells the archive command with its required `--base origin/main --head HEAD`. Today it exits 2.

## Skills touched

Repository change.

## Installed behavior

No installed skill changes. Agents working in this repository:
- run and review this repository's management scripts as its own code, no longer bound to a skill's bundled script;
- get an immediate, named failure when an interface of those scripts misbehaves, where before they got an empty answer;
- see a red spec job, gate, or `/spec` run when a command fails, where before they saw a pass;
- review management code, and scripts skills deposit, against one written rule set, with lint catching blind and silent handlers.

## Impact

- `scripts/validate_harness.py`, `scripts/spec_changes.py`, `scripts/sync_labels.py`.
- `.github/workflows/checks.yml`, `spec-command.yml`, `spec-labels.yml`, `labels-sync.yml`, `pr-policy.yml`, `issue-triage.yml` (`secret.yml` has no `run:` step).
- `.agents/knowledge/harness-maintenance.md`, `skill-quality.md`, `spec-workflow.md`; `.agents/skills/code-review/SKILL.md`.
- `skills/meta/CONTEXT.md`.
- `ruff.toml`, `justfile`, `.pre-commit-config.yaml`.
- `AGENTS.md`, `ARCHITECTURE.md`, `openspec/config.yaml`.
- The label set on GitHub does not change. The rewritten `sync_labels.py` computes the same plan, and `labels / sync` applies it after the merge.

## Non-goals

- The other repository scripts that fall short of the rules: `check_pr_policy.py`, `sync_issue_metadata.py`, the remaining fallbacks in `validate_harness.py`, and `.agents/skills/change-workflow/scripts/run_log_digest.py`. They are a follow-up.
- A PEP 723 header or uv in CI for this repository's scripts, until a script needs a third-party dependency.
- Flipping `archive` to `--apply`. It edits only the local working tree, refuses uncommitted paths, and is undone by git.

## Tracked work

Issue #94.
