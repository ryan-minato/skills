# Actions and Checks Baseline

Read on every build. Every workflow the harness writes — quality gate or
automation — is bound by everything here. On GitHub, CI/CD, code-quality
control, and community automation all run on Actions by default; this file
is the contract that keeps that default safe.

## Availability is probed, never assumed

Actions can be disabled at the organization or repository level, and an
organization can force the default `GITHUB_TOKEN` to read-only or restrict
which actions may run. `check_tooling.py --repo` evidences all three. If
Actions is off and cannot be enabled, the harness degrades to hooks,
documented commands, and human review; every "enforced" claim becomes
"convention" in those words, and no workflow file is written into the
repository — dead files mislead future agents.

## The honesty gate

- A **skipped job reports "Success"**. Wherever path filters meet required
  checks, ship an aggregator gate job (`if: always()`) that inspects its
  dependencies' results; the gate job's name — not the filtered jobs' — is
  what a ruleset requires.
- Required checks match by **job name** across all workflows. Keep a job-name
  registry with domain prefixes (`checks / lint`, `pr / checklist`); a
  renamed job silently orphans its required check, which then blocks every
  PR forever waiting for a producer that no longer exists.
- What a check can *block* depends on the enforcement tier evidenced at
  stage 1. On a private Free repository nothing blocks; design for
  visibility and say "advisory".

## Workflow contract

- Mirror local checks: every CI job runs a command that exists and passes
  locally, and the job-name-to-command map is deposited in project
  knowledge. CI does not choose the project's linters or tests.
- Declare `permissions:` explicitly in every workflow — the default is
  organization-configurable and can change with no commit. Once any
  permission is specified the rest become `none`; prefer a minimal
  workflow-level block plus per-job grants. Resolve the current scope list
  from first-party docs when writing it.
- `concurrency` has three shapes: PR checks use
  `${{ github.workflow }}-${{ github.ref }}` with `cancel-in-progress:
  true`; per-object automation uses a per-issue group with
  `cancel-in-progress: false` (true drops the second event); deploys use
  `queue:` — and `queue: max` with `cancel-in-progress: true` is a
  validation error that yields **no runs at all**, which a required check
  then waits on forever.
- Fork pull requests get read-only tokens and no secrets. Policy on fork
  PRs is therefore expressed as failing checks, never as mutating
  workflows. `pull_request_target` runs with write permissions even from
  forks: never check out PR code under it, never run repository scripts,
  never install dependencies; any use is a reviewed, owner-listed
  exception.
- First-party actions only (`actions/*`, `github/*`). A third-party action
  is a supply-chain decision: explicit user opt-in, full commit-SHA pin,
  and a recorded review date and update owner — a pin also freezes
  security fixes.
- One `on:` design per workflow; `push` listens to the default branch only
  (a `pull_request` + `push` pair double-runs the same commit). Weekly, not
  daily, schedules, each with a named owner — a scheduled workflow is
  auto-disabled after 60 days of repository inactivity, and a stopped
  drift check is indistinguishable from a passing one.
- Merge queue, if selected, is decided **before** workflows are written: it
  requires the `merge_group` event on every required-check workflow and
  cannot be used with wildcard branch patterns.

## Runners and cost

github.com always has eligible hosted runners; the eligibility question
returns only for GHES or self-hosted runners — where a self-hosted runner
on a public repository is also a fork-PR code-execution hazard. On private
repositories minutes are billed: prefer lean matrices, cancel-in-progress,
and schedule discipline; note that advisory-only tiers still pay for every
run.

## Debugging discipline

Validate workflow YAML locally before pushing. Never fetch a full run log —
use `scripts/run_log_digest.py` (failed jobs only,
tail-limited). Cap the push-wait-read loop and report instead of iterating
indefinitely. Never weaken, skip, or delete a check to make it pass:
changing a check's strictness is a separate, human-approved decision, not
part of fixing a failure.
