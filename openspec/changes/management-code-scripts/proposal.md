## Why

The `management-code` changes rewrote `scripts/spec_changes.py` and `scripts/sync_labels.py` under `## Management code` in `.agents/knowledge/skill-quality.md` and left four scripts as a follow-up. Those four still read guaranteed interface data through defaults: `pr.get("head") or {}`, `issue.get("labels") or []` with an `isinstance` filter, `str(item.get("name", ""))` over `labels.json`, `run.get("jobs") or []`. They gate pull requests (`pr / policy`), apply issue metadata (`issues / triage`), validate the harness, and diagnose CI, so a default in them can let a check pass on data it never read.

## What Changes

- **`scripts/check_pr_policy.py` rewritten.**
  - The event payload is checked once, where it is loaded: an object whose `pull_request` carries the keys the script reads (`number`, `title`, `body`, `draft`, `head.sha`, `head.repo`, `base.sha`, `user.login`), each of its documented type. A mismatch fails naming the payload file, the key path, and the value received. The SHA check moves from the commit-range step to the load.
  - The `gh pr view` output of the `--pr` dry run is checked the same way: gh's exit status, parseable JSON, and the requested fields.
  - A failing `git log` names the command and carries git's own message.
  - `pull_request.body` (nullable) and `pull_request.head.repo` (null for a deleted fork) stay as domain logic, written explicitly and citing the schema.
- **`scripts/sync_issue_metadata.py` rewritten.**
  - The issues event payload is checked where it is loaded: `issue.number`, `issue.body`, `issue.labels`, and `repository.full_name`.
  - The `gh api` issue output of `--issue` is checked the same way.
  - `labels.json` is read strictly: a list of objects, each with a non-empty string `name`, otherwise the run fails naming the file and the index.
  - The body (nullable), labels absent from a webhook issue, and the REST form of a label (a string or an object) stay as domain logic.
  - The dry-run default stays. A DELETE answered with `HTTP 404` still counts as already removed, now matched on gh's message rather than on any `404` in stderr.
- **`scripts/validate_harness.py`: the remaining fallbacks go.**
  - `labels.json` is parsed once and checked (a list of objects with a string `name`). The second, differently shaped parse in the spec check is dropped.
  - The `spec_changes.py labels --taxonomy` output is checked (JSON, an object, `managed` a list of strings) and a mismatch names the command.
  - A missing `git` is named.
  - The `# unreachable` return that stands in for data goes, with `fail` typed as never returning.
  - The collected-finding checks stay as they are.
- **`.agents/skills/change-workflow/scripts/run_log_digest.py` rewritten for its management role.**
  - The `gh run view --json` output is checked at entry, the run and every job the digest reads. A mismatch fails naming the command and the value.
  - A job whose failed-step log gh cannot return keeps an empty `log_tail` and carries gh's message in a per-job `log_error` field (`null` when the log was fetched). This is a designed deferral and the only change to the output.
  - The docstring names the script's real path, and the file is formatted to the repository's ruff settings.
- **Exit codes.** In all four scripts an interface failure exits 1, naming the interface and the value received; exit 2 stays only for bad arguments. The docstrings and the `issues / triage` row of `.agents/knowledge/github-checks.md` say so. That row then tells an unmapped form answer (the rest of the plan applied) apart from an interface failure (nothing applied).
- **Lint.** `just lint` and the pre-commit ruff hooks also cover `.agents/skills/change-workflow/scripts`, named explicitly and never through a pattern that reaches the public-skill symlinks under `.agents/skills/`. The `ruff.toml` header, the `AGENTS.md` Validation row, the `ARCHITECTURE.md` Quality Gates bullet, and the last paragraph of `## Management code` describe the new scope.

## Skills touched

Repository change.

## Installed behavior

No installed skill changes. Agents working in this repository:
- get a failure naming the interface and the value received when a payload, `gh`, git, `labels.json`, or the spec taxonomy misbehaves, where before a script went on with a default;
- read an interface failure as exit 1 and reserve exit 2 for a mistyped command line in all four scripts;
- see which failed job's log could not be fetched, and why, inside the run digest instead of only on stderr;
- have the change-workflow project skill's script linted like the rest of the management code.

## Impact

- `scripts/check_pr_policy.py`, `scripts/sync_issue_metadata.py`, `scripts/validate_harness.py`.
- `.agents/skills/change-workflow/scripts/run_log_digest.py`; the skill's `SKILL.md` keeps its invocation.
- `.agents/knowledge/github-checks.md` (the `issues / triage` row), `.agents/knowledge/skill-quality.md` (`## Management code`, last paragraph).
- `justfile` (`lint`), `.pre-commit-config.yaml` (ruff `files`), the `ruff.toml` header.
- `AGENTS.md` (Validation row), `ARCHITECTURE.md` (`## Quality Gates`).
- The workflow command lines of `pr-policy.yml` and `issue-triage.yml` do not change. Both run from the default branch, so their live behavior changes only after the merge.

## Non-goals

- Public skills' own `scripts/`, which keep the `## Scripts` rules. This includes `meta-github-workflow`'s `scripts/run_log_digest.py` and its asset, which serves only as a starting shape.
- The `--pr` dry run reporting dependabot as `app/dependabot` while the event path reports `dependabot[bot]`. It is a separate bug, and the output stays unchanged here.
- A shared helper module across the scripts, and a check that every non-symlink `.agents/skills/*/scripts` directory is linted.
- The `/spec` comment parsing bug, tracked in its own issue.

## Tracked work

Issue #97.
