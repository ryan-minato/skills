# GitHub Checks

Read this before adding, renaming, removing, or debugging a GitHub Actions
job, and before changing the ruleset's required checks. Every job runs a
recipe or script that exists and passes locally; CI never chooses its own
linters. `just check` is exactly what `checks / quality` runs, so a green
`just check` predicts it.

| Check | Workflow | Runs on | Runs | Tier |
|---|---|---|---|---|
| `checks / quality` | `checks.yml` | every pull request; push to `main` | `just install-tools`, `just check` | feeds the gate |
| `checks / spec` | `checks.yml` | every pull request; push to `main` | `just spec-check` or `just spec-changes check --all`; `just spec-sync` | feeds the gate |
| `checks / devcontainer` | `checks.yml` | every pull request; push to `main` | builds the dev container, runs `just check` inside | feeds the gate |
| `checks / gate` | `checks.yml` | every pull request; push to `main` | `scripts/ci_gate.ts` over the jobs' `needs` | **required** |
| `pr / policy` | `pr-policy.yml` | pull request events | the base's `check_pr_policy.ts`, then `check_commit_safety.ts --range` | **required** |
| `scan-secrets` | `secret.yml` | every pull request; push to `main` | TruffleHog `--only-verified` over the range | **required** |
| `spec / command` | `spec-command.yml` | a `/spec` comment by a collaborator | `scripts/spec_changes.ts` `snapshot`, `show`/`status`; one reply | automation |
| `spec / labels` | `spec-labels.yml` | pull request events, `spec/approved` label events | `scripts/spec_changes.ts` `labels`; `scripts/spec_approval.ts` | automation |
| `issues / triage` | `issue-triage.yml` | issue opened or edited | `scripts/sync_issue_metadata.ts --apply` | automation |
| `labels / sync` | `labels-sync.yml` | push to `main` touching the labels or the script; weekly; dispatch | `scripts/sync_labels.ts --apply` | automation |

A healthy run:

- `checks / quality`: `just check` exits 0.
- `checks / spec`: works only when a pull request touches `openspec/`,
  `.agents/skills/openspec-*`, `.agents/skills/.openspec-target`,
  `scripts/spec_changes.ts`, `scripts/lib/`, `deno.json`, or `justfile`,
  and otherwise logs "nothing to validate" and passes. Strict validation
  exits 0; an unarchived related change warns on a draft and fails a ready
  pull request (`.agents/knowledge/spec-workflow.md`
  `## Archive executor and the freeze`); on `main` no change sits outside
  `openspec/changes/archive/`; the regenerated `openspec-*` skills equal
  the committed ones.
- `checks / devcontainer`: works only when `.devcontainer/`,
  `.github/actions/setup-tools/`, `justfile`, or `deno.json` change; the
  container builds and `just check` exits 0 inside it.
- `checks / gate`: green only when no dependency failed, was cancelled,
  or was skipped unexpectedly; `ci_gate.ts` names the one that was.
- `pr / policy`: the body carries every heading of the base's template,
  the `Closes #N` or `N/A — <reason>` line, and the security checklist
  line; every commit subject (the title, for a fork) follows Conventional
  Commits. A ready pull request also has Changes and Validation filled,
  `Phase: implementation`, every box ticked, and a `Spec:` line that links
  the change directory or reads `Spec: none — <reason>`; a bare path is
  refused. The commit-range scan reads each commit: no added line carries
  a credential pattern, no commit carries an agent-attribution trailer,
  and on a same-repository pull request every author email is a noreply
  address. Pull requests by `dependabot[bot]` skip the body checks; their
  commit subjects are still checked. The job runs the base's
  `check_pr_policy.ts` when the base has it, else the base's
  `check_pr_policy.py`, until a follow-up change removes that fallback.
- `scan-secrets`: no verified secret. The check name is the job id; the
  job sets no `name:`.
- `spec / command`: one reply comment per command (`/spec show [<change>]
  [proposal|design|tasks|specs|all]`, `/spec status [<change>]`,
  `/spec help`); skipped for other comments and for bots.
- `spec / labels`: the printed plan and the pull request's `spec/*` axis
  labels agree, and a pull request with no related change carries none;
  the `spec/approved` record follows `.agents/knowledge/spec-workflow.md`
  `## Approval gates`.
- `issues / triage`: prints the label plan as JSON and applies it; exit 1
  means a form answer maps to no label in `.github/labels.json`.
- `labels / sync`: the repository's labels match `.github/labels.json` in
  name, color, and description; a label missing from the file is a
  warning, never a deletion.

Required checks: `checks / gate`, `pr / policy`, and `scan-secrets`; the
ruleset requires exactly these names, and its live state is declared in
`.agents/knowledge/github-settings.md`. The names are a stable interface:
the `checks` check of `scripts/validate_harness.ts` fails when a workflow
job name and this table disagree. Renaming one is three edits in order —
the workflow, this table, then the ruleset once the new name is live on
`main`.

## Privileged workflows

Privileged workflows: `spec-labels.yml` (`pull_request_target`), `spec-command.yml` (`issue_comment`), `issue-triage.yml` (`issues`).

The `workflows` check of `scripts/validate_harness.ts` reads the line
above; the rules these workflows follow live in
`.agents/knowledge/management-code.md` `## Workflow security`.

## Diagnosing a red check

1. Reproduce locally with the recipe in the table.
2. For a failure you cannot reproduce, digest the run with
   `.agents/skills/change-workflow/scripts/run_log_digest.ts` as the
   `change-workflow` project skill directs, instead of reading full logs.
3. A red `checks / spec` on a ready pull request whose change is not yet
   archived is the expected state until the freeze
   (`.agents/knowledge/spec-workflow.md` `## Archive executor and the freeze`).
4. A check turned red by a new tool release names the tool; fix it in its
   own change (`.agents/knowledge/management-code.md` `## Tool versions`).
5. Changing a check's strictness is a maintainer decision, recorded here
   first (`.agents/knowledge/agent-authority.md` `## No self-escalation`).

## Update this file when

- A job is added, renamed, or removed, or its command or trigger changes.
- A required check changes in the ruleset.
- A workflow adopts or drops a privileged trigger.
