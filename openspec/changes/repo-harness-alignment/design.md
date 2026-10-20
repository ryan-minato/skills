## Context

The branch already carries an earlier, narrower scope of this change (commits `671e482`..`05edc45`): the deny list, the blocked-work and mismatch rules in `AGENTS.md`, `## Pre-authorized actions` and `## Threat model` in `agent-authority.md`, the change workflow's checkpoint rules, pagers off, a regular-expression `check_workflows`, the layer calibration, and the artifact-roles schema section. That scope is the baseline this design audits; its ticked task list is removed with this package and superseded by the task list written after the closing, and `check_workflows` is replaced by the parsed check below.

The harness as audited on 2026-10-20 at `05edc45`:

- Entrypoint `AGENTS.md` (130 lines against a stated budget of about 100), `CLAUDE.md` (`@AGENTS.md`), `ARCHITECTURE.md` (238 lines), eight knowledge files (1,059 lines), three project skills (`change-workflow` 264 lines, `skill-authoring` 154, `code-review` 212) and two references.
- Management code: nine Python scripts under `scripts/` (3,267 lines; `spec_changes.py` 1,064, `validate_harness.py` 556) and `run_log_digest.py` in the change workflow, all standard library, no tests, YAML read by regular expressions; invoked by `justfile` recipes, three pre-commit hooks, nine `run:` lines in six workflows, and two path filters.
- Gates: pre-commit (14 hooks), `checks / quality`, `checks / spec`, `checks / gate`, `pr / policy`, `scan-secrets`; the ruleset requires the last three names. Labels: two `spec/*` axes kept by `spec / labels` (`pull_request_target`). Privileged workflows: `spec-labels.yml` (`pull_request_target`), `spec-command.yml` (`issue_comment`), `issue-triage.yml` (`issues`).
- Authorization: the H1 policy, a prefix deny list, no `ask` entry, two wildcard `gh api` read allows; the package closing is conversational and recorded on the `Approval:` line only.
- Working tree: the maintainer's uncommitted switch of the dev container to the `deno` and `uv` features with unpinned tools, which makes the ruff pin check exit 1 and blocks every commit through the always-run hook.

Binding constraints:
- Nothing under `skills/` changes (the product); ruff's selection over its Python stays byte-identical.
- The required check names stay.
- How each workflow finds its code: `pr / policy` takes its workflow file from the pull request's merge ref but checks out the base and runs the base's scripts, and a local `./.github/actions/…` reference resolves in that base checkout; `spec / labels` (`pull_request_target`), `spec / command` (`issue_comment`), and `issues / triage` (`issues`) run `main`'s workflow file and scripts; `labels / sync` runs on `main` only. A port of any of their scripts therefore takes effect after merge, and `pr / policy`'s script and its YAML cannot switch together in one pull request.
- The `openspec-*` skills change only through `just spec-sync`.
- `agent-authority.md` may narrow but not widen agent authority.

Overlapping open drafts: #105 rewrites `check_pr_policy.py`, `sync_issue_metadata.py`, `validate_harness.py`, and `run_log_digest.py`; the repository-side parts of #102, #103, and #104 change `spec_changes.py`, `checks.yml`, `spec-command.yml`, and register rows. See proposal.md for motivation.

## Placement

| What Changes item | File and section | Check that proves it |
|---|---|---|
| Container pins and lockfile | `devcontainer.json` `features`; `devcontainer-lock.json`; `.gitignore` | pin-agreement check; the `devcontainer` job on this pull request |
| CI tool setup | `.github/actions/setup-tools/action.yml` (inputs off by default); each workflow's setup steps | pin-agreement check; workflow check (privileged jobs ask for no tool) |
| Pin agreement | `scripts/validate_harness.ts` `tools` check, which also reads the Deno version of `pr-policy.yml`'s setup step | one disagreeing fixture per tool, one feature without a lock entry |
| Container build | `checks.yml` `devcontainer` job, in `gate`'s `needs` | the run on this pull request |
| Script ports | `scripts/*.ts`, `scripts/lib/*.ts`, `.agents/skills/change-workflow/scripts/run_log_digest.ts` | per-script tests; equivalence run |
| Deno configuration | `deno.json` (`imports`, `fmt.include` and `lint.include` = `scripts/**/*.ts`, `.agents/skills/change-workflow/scripts/**/*.ts`), `deno.lock` | `deno check --frozen`; `deno fmt --check` and `deno lint` with no path arguments list nothing outside those globs |
| Recipes and hooks | `justfile` (`test`, `lint`, `check` without the doubled hooks, `review-threads`, `spec-approval`, every script recipe); `.pre-commit-config.yaml` (ported hooks, `no-commit-to-branch`) | `just check`; a scratch commit on `main` is refused |
| Path filters | `checks.yml` `spec` job's changed-path pattern and `labels-sync.yml` `on.push.paths`, each naming the port, `scripts/lib/`, `deno.json`, `deno.lock` | grep: no `.py` path of a ported script in `.github/`, `justfile`, hooks, or knowledge |
| Workflow check | `scripts/validate_harness.ts` `workflows` check over workflows and `.github/actions/*/action.yml`; `github-checks.md` privileged-workflow list | one fault fixture per rule |
| Gate script | `scripts/ci_gate.ts`; `checks.yml` `gate` job | tests over the `needs` JSON shapes |
| Commit-range scan | `scripts/check_commit_safety.ts --range`; `pr-policy.yml` (base copy) | fixture repository tests |
| `pr / policy` transition | `pr-policy.yml`: pinned `setup-deno`, then the base's `.ts` or its `.py` | the run on this pull request takes the Python branch; a scratch run with a base holding the `.ts` takes the Deno branch |
| Approval label | `scripts/spec_approval.ts`; `spec-labels.yml` (types `labeled`/`unlabeled`, a deciding-step timeout below the job's, the failure step); `labels.json`; `spec-workflow.md ## Approval gates`; `agent-authority.md` | stubbed-`gh` tests; a bounds test over the workflow file; read-only run against this pull request |
| Settings | `.claude/settings.json`; `agent-authority.md ## Pre-authorized actions`, `## Threat model`; `scripts/review_threads.ts` | JSON parses; no allow entry is a prefix of a write form; readback |
| Rule owners | the owner map below | the audit's duplication grep re-run |
| Knowledge split | `management-code.md` (new); `skill-quality.md ## Script assets`; `ruff.toml` header comment | pointer check; `git diff origin/main -- ruff.toml` comment lines only; readback |
| Wide tables, budgets | knowledge files, `ARCHITECTURE.md`, the template; `validate_harness.ts` `budget` check | budget check and its tests |
| Contradictions D1–D4, D6, D7, D9, D10 and CONTRIBUTING setup | `skill-authoring`, `code-review`, `spec-workflow.md ## Archive executor and the freeze`, `openspec/config.yaml` archive guidance, `validate_skills.ts` (message and install-command check), `github-checks.md`, `.github/CONTRIBUTING.md` | readbacks; install-command test; grep |
| Stale passages S1–S12, S14, S17 | the files the audit names | grep for each stale string; S8 and S9 through the review record |
| S13, S15, S16 | unchanged; pending decisions in the description | decision recorded before ready |
| Register, calibration, review record | `harness-maintenance.md` | the settings readback; readback |

### Rule owners

| Rules (audit numbers) | Owner |
|---|---|
| Lifecycle, draft timing, freeze, archive executor, task ticking, deviation, no backfill, `skip_specs`, `Spec: none` (R1, R2, R5, R11, R12, R24–R26, R31) | `spec-workflow.md` |
| Approval on the pull request: reconciling threads, the label, phase lines (R3, R13) | `spec-workflow.md ## Approval gates`; the template carries the fields |
| Publish gate and authorization of remote writes, pending-decision shape, no self-escalation, never approve, merge, or set the approval label, push one ref (R4, R6, R19, R23, R32) | `agent-authority.md` for the rule; the change workflow for the mechanics |
| Merge method, branch naming, GitHub as the only platform (R7, R22, R27) | `github-workflow.md` |
| Public-skill bar, self-containment, frontmatter limits, script contract and script assets (R8, R17, R34, the asset part of R18) | `skill-quality.md` |
| Repository code, workflow security, tool pins (R16, the repository part of R18, R21) | `management-code.md`, with `github-checks.md` keeping the privileged-workflow list the check reads |
| Required check names, `just check` equals CI (R15, R36) | `github-checks.md` |
| Commit type semantics, README pairs (R9, R30) | `AGENTS.md` |
| Adding a public skill, catalog obligations, prefixes (R20, R33, R35) | `skill-authoring` (procedure); `ARCHITECTURE.md` (mechanism) |
| Generated OpenSpec skills, including that `openspec-archive-change` is not this repository's executor (R10) | `spec-workflow.md` |
| Design shape and when it is required (R28, R29) | the project schema's design instruction |
| Pull request Changes/Validation reserved lines (R14) | the template |

## Dependencies and handoffs

- Deno 2.9.7 (the version the maintainer's container runs and the reference harness pins), `jsr:@std/yaml`, `@std/assert`, `@std/path`, `@std/cli`, pinned in `deno.json` and locked.
- Container features `ghcr.io/hoshiori-dev/devcontainer-features/deno:1`, `uv:1`, `openspec:1` (public), each with an exact `version` option, resolved to digests by the committed lockfile; `denoland/setup-deno` and `devcontainers/ci` pinned by SHA.
- Overlapping drafts: the equivalence baseline is `main` at `f013357` (recorded now). A draft merged before the first port starts is ported as merged, with the baseline moved to that merge; a draft still open afterwards rebases its repository-side part onto the TypeScript after this change merges, and its skill-side part is unaffected. The alternative — carrying #105's script changes into the ports and closing it as superseded — is queued as a pending decision.

## External impact

- `skills/**`, `.agents/skills/openspec-*`, `openspec/changes/archive/`, `README.md`/`README.zh.md`, `marketplace.json`: unchanged; `just validate` proves the marketplace still matches.
- `skills/sdd/openspec-workflow/assets/spec_changes.py`: the register row naming `scripts/spec_changes.py` its mirror becomes a note that the repository's script is its own port and binds nothing; the asset is not touched.
- GitHub after merge: `labels / sync` creates `spec/approved`; `spec / labels`, `spec / command`, and `issues / triage` start running the Deno scripts from `main`; `pr / policy` runs the Deno branch from the next pull request on.

## Decisions

- **Deno for all management code, TypeScript files executed directly**: one runtime for every harness script, permissions stated in the shebang so a reader sees what a script can touch, the standard library's YAML parser, tests in the same toolchain. Rejected: keeping Python with `uv` and PEP 723 — workable, but permissions stay implicit and the reference harness's tested workflow checks cannot be reused in shape.
- **Shebang `#!/usr/bin/env -S deno run --no-prompt <permissions>`**: a missing permission is an error, never a prompt an unattended agent waits on. Read access is scoped to repository paths, relative to the repository root every caller runs from (recipes, hooks, workflow steps); a script that reads a caller-named file (event payloads) gets unscoped read; `--allow-run` names each command and is treated as unconfined for it; network access names the host.
- **Equivalence before deletion**: each port runs beside its Python original on the baseline tree, on fixture repositories, and on recorded event payloads; stdout and exit status must match except for the intentional differences — diagnostics on stderr, the dropped modes, the dropped `--allow-private-email`, and `run_log_digest` failing on an unfetchable log or a run without `jobs`. The comparison harness lives in the scratch directory; its results go to the pull request.
- **Order**: the toolchain commit first — the working-tree dev container pinned exactly, Python kept, so every check passes from the first commit; then the ports from the leaves inward — `check_skill`, `validate_skills`, `gen_marketplace`, `validate_harness` (with the `tools`, `workflows`, and `budget` checks; the setup action lands in the same commit, because the `tools` check reads it), `check_commit_safety`, `check_pr_policy` (with the guarded transition), `sync_labels`, `sync_issue_metadata`, `run_log_digest`, `review_threads` (new), `ci_gate` (new), `spec_changes`, `spec_approval` (new); then the knowledge rebuild; the `devcontainer` job and Python's removal from the container land in the commit after the last Python caller is ported. Each port is one task and one commit.
- **`pr / policy` guarded transition**: after the base checkout the job installs Deno with the SHA-pinned `denoland/setup-deno` (not the local action, which the base lacks) and runs `./scripts/check_pr_policy.ts` when that file exists, else `python3 scripts/check_pr_policy.py` with the runner's Python. On this pull request the base has only Python, so the Python branch runs; from the next one on, the Deno branch. Rejected: two pull requests in sequence — slower for the same result; a branch-copy fallback as the reference uses — it lets the pull request check itself.
- **Workflow security as a validator check, the approval workflow's bounds as a test**: rules every workflow must follow belong to the check every commit runs, with a fix-it message; the exact bounds of the one label-writing workflow (job permissions, job timeout, a deciding-step timeout below it, the failure step, checkout without a ref and with `persist-credentials: false`, setup with Deno only, the token only in the API steps' environment, no cache) are asserted by a test over that file, as the reference does. Privileged means a write token near untrusted input: `pull_request_target`, `issue_comment`, `issues`. `workflow_run` is forbidden outright.
- **Approval label recomputed from rules**, following `devcontainer-features`:
  - The run decides from the pull request's current labels and timeline, not from the event that started it, so a cancelled pending run loses nothing.
  - A record is written when the latest person-added `spec/approved` event names the current head and no person removed the label after it.
  - A record counts only when its author is `github-actions[bot]` (type Bot) and it answers that latest event.
  - The package is every path of each unarchived change except its `tasks.md`, compared by mode and blob, plus the main spec of each delta. A symlinked or submodule change entry refuses approval.
  - An archive commit is not a package change.
  - The label does not authenticate the person, because agents act through the maintainer's `gh` login. It proves a write-access act bound to one package. Adding or removing it, and editing or deleting the record comment, are the maintainer's alone, stated in `agent-authority.md` and denied in the settings.
  - After asking for approval an agent pushes nothing to the package until the record appears, and the maintainer checks that the recorded commit is the one reviewed (owned by `spec-workflow.md ## Approval gates`): a push just before the label would otherwise be recorded as approved.

  Rejected:
  - A digest-bound artifact with an expiry (`devcontainer-templates`): more code, and re-approval on slow pull requests.
  - The platform's review approval: the maintainer approves at merge, not at the package.
  - A label rule inside `pr / policy`: the withdrawal uses `GITHUB_TOKEN`, which starts no run, so a check reading the label races it.

  The freeze stays conversational.
- **One label-writing privileged workflow**: the approval logic joins `spec-labels.yml`; `spec-command.yml` keeps its read-and-comment role; both and `issue-triage.yml` are on the privileged-workflow list, each with its trigger.
- **Commit-range scan in `pr / policy` from the base's copy**: the scan must not run the pull request's own code, and must read each commit, because rebase merges land every commit on `main`. Author emails are checked only on same-repository pull requests; fork commits are squashed.
- **`ask` entries for outward-write families, recipes for the two reads**: a GraphQL or wildcard `gh api` allow entry also matches writes, so the reads move to a fixed-query recipe and an already-allowed `gh pr view` form. The `ask` and deny entries are fool-proofing; the container is the boundary.
- **Owner per rule, pointers elsewhere**: the owner holds the full statement; another file may name the rule in a pointer clause but not restate its conditions. `AGENTS.md ## Always` keeps a one-line form of each always-on rule because it loads first.
- **`management-code.md` for the repository's code; `skill-quality.md ## Script assets` for the product's Python**: the two have different readers and, after the port, different languages.
- **The budget check covers lines and table width only**: duplication cannot be checked mechanically without false positives; the audit's grep list is re-run at the next periodic review.
- **Container lockfile committed**: feature install scripts run as root in the container that holds the forwarded `gh` login, the same reasoning as SHA-pinned actions.

## Risks / Trade-offs

- [A port changes behavior a privileged workflow depends on, visible only after merge] → equivalence runs on recorded payloads; read-only runs of `spec_changes.ts snapshot` and `spec_approval.ts` against this pull request through the live API; `sync_labels.ts` in its default dry run against the live repository, which must plan `spec/approved` as a create; the first events after merge are watched and a regression is fixed forward or reverted.
- [`spec_changes.ts` is the largest port and drives the freeze] → ported late, with the equivalence run over every archived change and the `check`, `labels`, `status`, `show`, and `archive --dry-run` modes.
- [Node in the container is the OpenSpec feature's unpinned `lts`] → OpenSpec needs Node 20.19 or later, which any `lts` meets; Node stays out of the pin-agreement check on purpose, since a second `node` feature entry would install it twice; the `devcontainer` job runs `just check` inside the built container.
- [A workflow on a same-repository branch can post a record-shaped comment as `github-actions[bot]`, and any write-access account can add the label] → accepted, as the reference accepts it: write access is the maintainer's and agents acting through the maintainer's login; the record must also answer the latest person-added label event and its head; recorded in `agent-authority.md ## Threat model`.
- [The deny entry on `spec/approved` also stops a read-only command that names the label] → agents read it with the search tools or `just spec-approval`; the cost is small next to an agent setting its own approval.
- [The `ask` list prompts on every push] → that is the gate's decision point; pre-authorization names which ref the prompt may confirm, and the threat model has no unattended agent.
- [A large diff for the second deliberation] → one commit per task, the Changes section linked per area, and the owner map as the reading guide.
- [Bootstrap: `spec/approved` does not exist until `labels / sync` runs on `main`, and `pr / policy` runs Python on this pull request] → this pull request's package closing stays conversational; both are recorded in its description, not in a durable file.

## Verification plan

Before the pull request is marked ready:

- Toolchain: the `devcontainer` job green on this pull request (container built, `just check` inside, `tf-validate` not skipped); `checks / quality` green with the setup action; the pin-agreement check passes on the tree and fails on fixtures with one disagreeing pin per tool, the justfile's OpenSpec pin included, and with a feature missing from the lockfile.
- Ports: each script's `deno test` green; the equivalence run matches on the baseline (results per script, intentional differences listed); `just check` green; no `.py` file under `scripts/` or `.agents/skills/change-workflow/scripts/`; `git grep -nE 'scripts/[a-z_]+\.py|run_log_digest\.py'` over `.github/`, `justfile`, hooks, knowledge, project skills, `AGENTS.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md` finds only the `pr / policy` fallback line.
- Permissions: each script run through its shebang (not `deno run -A`) in its real invocation, locally and in CI.
- Workflow check: passes on the tree; on a scratch copy, one finding naming file and line for each fault — missing permissions, unpinned external action (in a workflow and in an action file), event expression in `run:` and in `script:`, missing `persist-credentials: false`, missing timeout, an unlisted `pull_request_target` and an unlisted `issue_comment` workflow, a head checkout under each, a `workflow_run` trigger, a write permission in a `pull_request` workflow, a tool input in a privileged job; a local `uses: ./.github/actions/setup-tools` passes.
- Gate script: tests over success, failure, cancelled, and skipped dependencies; the `checks / gate` run on this pull request.
- Commit-range scan: fixture tests — same-repository private email fails, fork private email passes, a key line fails, a key added in one commit and removed in a later one fails, an attribution trailer fails, a clean noreply range passes.
- `pr / policy` transition: its run on this pull request (Python branch); a scratch run of the job's script step against a base that holds the `.ts` (Deno branch).
- Approval label: stubbed-`gh` tests covering:
  - the maintainer's account adds the label at the current head, and a record is written;
  - a package change after the record, an unrelated change after it, and an archive commit;
  - `.openspec.yaml` changed, a main spec under a delta changed, a second change added, and a non-design file added;
  - the head moved between the event and the run, and during the run;
  - a person re-added or removed the label during the run;
  - two records, a record comment by a non-bot, and a record answering an earlier event;
  - the label present with no record (the run was cancelled);
  - a symlinked change entry, a failed run, a deciding step that timed out, and a closed pull request.

  Also: the bounds test over `spec-labels.yml`; `just spec-approval 117` read-only against the live pull request.
- Settings: JSON parses; no allow entry is a prefix of `gh api graphql`, of a `gh api` write form, or of `gh pr edit`/`gh issue edit`; `just review-threads 117` lists threads with `isResolved`; a clean-context reader given `agent-authority.md` and the settings states which commands stop for a decision, why, and that agents never set `spec/approved`.
- Knowledge: the audit's duplication grep re-run, each rule found in full only in its owner; the budget check passes; clean-context readbacks of `AGENTS.md` (where the freeze rule lives, what to do when blocked), `spec-workflow.md` (the archive executor, the approval label), `management-code.md` (how to write a new script), `skill-authoring` and `code-review` (no archive before ready, no block on an unarchived ready pull request), and — given `AGENTS.md`, the change workflow, and the installed `openspec-archive-change` skill — "the maintainer says 'archive it'; what do you run?" answered with `just spec-changes archive`.
- Stale passages: grep for each of S1–S12, S14, S17 returns nothing.
- Settings readback (periodic review step 4): read-only `gh api` reads of the ruleset, repository settings, branch protection, labels, and security features, compared row by row with `github-settings.md`, whose `## Last verification` becomes one dated result.
- Untouched product: `git diff --stat origin/main...HEAD -- skills/ .agents/skills/openspec-* openspec/changes/archive README.md README.zh.md .claude-plugin/` empty; `ruff check` over the public skills' Python gives the same result as on `main`.

Skipped:
- Branch copies of the privileged workflows and `labels / sync`: they run `main`'s copy. Proved before merge by the equivalence runs, the stubbed tests, the bounds test, and the read-only live runs above; after merge by the `labels / sync` run on the merge commit (`spec/approved` exists) and the first pull request, comment, and issue event — recorded as pending in the description.
- The commit-range scan and the Deno branch of `pr / policy` on a live pull request: `pr / policy` runs the base's copy, so both take effect from the first pull request after merge; proved now by the fixture tests and the scratch run.
- Triggering every `ask` entry in a live session: an `ask` prompt on a real write would publish if confirmed; the entries are proved by the parse and the prefix check, and the deny entries by one refused `gh pr edit 117 --add-label spec/approved` in a session.
