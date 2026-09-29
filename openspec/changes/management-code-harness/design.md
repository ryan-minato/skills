## Context

See proposal.md for motivation. `openspec/changes/management-code/design.md` covers the skills this change serves.

**Current state:**
- **`scripts/spec_changes.py`** (1029 lines) is `cmp`-identical to `skills/sdd/openspec-workflow/scripts/spec_changes.py`.
  - Callers:
    - `checks / spec` runs `check`, through `just spec-check` and `just spec-changes check --all`;
    - `spec / command` runs `snapshot`, `show`, and `status`;
    - `spec / labels` runs `snapshot` and `labels`;
    - `validate_harness.py` runs `labels --taxonomy`;
    - agents run `archive`, through `openspec/config.yaml` and the change-workflow gotcha.
  - Nothing calls `related`.
  - The unknown-change error of `select()` puts request-authored directory names unescaped into the reply comment.
  - The snapshot reader turns a 404 on a subtree into "empty".
- **`scripts/sync_labels.py`** (239 lines) is `cmp`-identical to `meta-github-workflow`'s. `labels / sync` runs it with `--apply` on every push to `main` that touches it.
- **`validate_harness.py` `check_copies`** enforces both identities: `sync_labels.py` with a `# DIVERGENCE:` escape, `spec_changes.py` without one, although the docstring promises one for both.
- **Workflow steps that hide an unexpected failure:**
  - `checks.yml` decides whether spec paths changed with `git diff … | grep -Eq` inside an `if`, so a failed diff reads as "nothing to validate".
  - `spec-command.yml` ends its reply block with `|| true`. Posting the script's output as the reply is a designed deferral, but the `|| true` also turns a crash into a green run.
  - `spec-labels.yml` loops over `< <(jq …)`, so a `jq` failure reads as an empty plan.
- **Script guidance.**
  - `.agents/knowledge/skill-quality.md` `## Scripts` governs skills' `scripts/`.
  - `.agents/skills/code-review/SKILL.md` calibrates scripts as developer tooling and says scripts copied into targets "follow the same calibration: their input is repository content, not adversarial traffic". That is false for a script running under `pull_request_target` or `issue_comment`.
- **Lint.** `ruff.toml` selects `E,F,W,I,UP,B,SIM,RUF`, so E722 is already on and BLE001 and S110 are off. `just lint` and the pre-commit hooks cover `scripts/` and `skills/*/*/scripts/`. Asset scripts are unlinted; `ruff format --check` reports three of them unformatted today.

**Binding rules:**
- **Commits.** Every commit passes `just check` on its own, because branches are rebase-merged. So the copies check leaves before either script diverges, and the lint scope grows only after the skill change's assets are formatted and exist.
- **Parsers.**
  - `labels_case_line` reads the literal `spec/…|…)` case line of `spec-labels.yml`, which must survive the loop rewrite.
  - `check_ruff` warns when the pre-commit ruff `files` pattern stops matching `skills/core/x/scripts/y.py`.
- **Commit gate.** `check_commit_safety.py` flags added lines shaped like `token = …`. The rewritten scripts keep the existing `# pragma: allowlist secret` markers where a header is built.
- **Privileged runs.** `spec / command` and `spec / labels` run from `main` and never from this pull request, so their live behavior changes only after the merge.

## Placement

| What Changes bullet | File and section | Proof |
|---|---|---|
| Copies check removed | `scripts/validate_harness.py`: `check_copies` and its entry in `main()`; the `SYNC_LABELS`, `SYNC_LABELS_ORIGIN`, and `SPEC_CHANGES_ORIGIN` constants; the docstring's `copies` entry. `SPEC_CHANGES` stays, because `check_spec_labels` runs it. | negative test below; `just validate` |
| Synchronization register | `.agents/knowledge/harness-maintenance.md`: the `spec_changes.py` row (source becomes `scripts/spec_changes.py`; mirror becomes the archive executor and the freeze in `spec-workflow.md`); the workflow-assets row (same steps and script path; the CI-facing command line of `openspec-workflow`'s `assets/spec_changes.py` pairs with this repository's script) | readback |
| `spec_changes.py` rewritten | the script, and its callers' prose: `justfile` recipe comment; `ARCHITECTURE.md` spec paragraph; `.agents/knowledge/spec-workflow.md` "Framework skill" sentence and `## Request automation`; header comments of `spec-command.yml` and `spec-labels.yml` | script harness below |
| `sync_labels.py` rewritten | the script | script harness below; plan-identity run |
| Workflow steps | `.github/workflows/checks.yml` (the `changed` step); `spec-command.yml` (reply status handling); `spec-labels.yml` (the two loops, keeping the literal case line). Every other `run:` step is read and left as it is. | local step runs below; `just validate` |
| Management-code rules | `.agents/knowledge/skill-quality.md`, new `## Management code` after `## Scripts`, and one line in `## Scripts` pointing at it for `assets/`; `.agents/skills/code-review/SKILL.md` `## Scripts: the threat model` and `## Do not report what machines catch` | readback |
| `meta/CONTEXT.md` carve-out | `skills/meta/CONTEXT.md` `## Contract`, the assets bullet | readback |
| Lint | `ruff.toml` (`select` gains `BLE001` and `S110`; `[lint.per-file-ignores]` exempts `skills/*/*/scripts/**/*.py` from both; the header comment); `justfile` `lint` recipe (directories `skills/meta/*/assets` and `skills/sdd/*/assets`); `.pre-commit-config.yaml` ruff `files` patterns; `AGENTS.md` Validation row; `ARCHITECTURE.md` `## Quality Gates` lint bullet | `just lint`; negative test below; `just validate` (no ruff warning) |
| Archive command | `openspec/config.yaml` archive guidance | the command runs (below) |

## Decisions

- **Remove the check; do not loosen it with a divergence marker** (copies bullet). A marker keeps the pair bound and only moves the burden onto a comment. The pair should not be enforced at all; after the rewrite the files differ, but a future match would not be a defect either.
- **Keep every subcommand a caller uses, and drop `related`** (spec_changes bullet).
  - `grep` finds no caller of `related` outside the script, the recipe comment, and the epilog.
  - `archive` stays, with its local-write default and `--dry-run`, for the reasons in Non-goals.
- **Interfaces are checked once, where the data enters** (both scripts).
  - The places are: API responses in the HTTP helper, `gh` output where it is parsed, git plumbing at each call, and the snapshot document on load.
  - Inside, data is indexed directly.
  - Base-side snapshot entries carry no `paths` by the script's own schema, so that absence stays domain logic.
  - A path missing from a tree listing means "not present at this revision", a legitimate state for a new change. A 404 on a tree the listing named means the interface broke, so the script fails.
- **Unknown change name is a bad argument, exit 2** (spec_changes bullet, workflow steps bullet).
  - A commenter's typo is answered with the list of related changes, and the run stays green.
  - Every other nonzero exit fails the run after the reply is posted.
  - Rejected: `|| true`, which hides crashes. Rejected: always failing the run, which makes a typo look like an outage.
- **Stdlib, `python3`, no PEP 723 header** (management-code rules bullet; user decision). CI installs no uv, so a header would be read by nothing. The rule is revisited when the first third-party dependency appears.
- **Lint the management assets, exempt the product scripts** (lint bullet; user decision to enforce in this repository only).
  - `skills/meta/*/assets` and `skills/sdd/*/assets` hold only management `.py` files.
  - Scaffold assets are product templates with placeholders, and ruff cannot parse them.
  - Skills' `scripts/` keep their rules, which the brief places out of scope. The global target (py310) already matches the assets' 3.10 floor, so no per-file target entry is needed.
  - Directories, not globs, go into the recipe: `sh` has no brace expansion, and an unmatched glob would reach ruff as a literal path.
  - A justified exception carries `# noqa: BLE001` or `# noqa: S110` with its reason. The rules make a hidden failure visible in review; they do not forbid a reasoned one.
- **Fix only the steps that hide an unexpected failure** (workflow steps bullet; user clarification that fail-fast targets fallbacks that postpone unexpected errors, not deferral by design).
  - Deferrals designed on purpose stay: the `/spec` reply posted before the run fails, the label plan printed before it is applied, the gate deciding on the needed jobs' results.
  - Reading every `run:` step found three that hide an unexpected failure, and only those are rewritten; each is fixed in the way that reads most plainly for that step.

## Risks / Trade-offs

- **[`labels / sync` applies the rewritten script to the live repository on merge]** → Before the pull request is marked ready, the old and the new script each run a dry run against `ryan-minato/skills`, and their plans must be byte-identical.
- **[A step that hides an unexpected failure is missed]** → Every `run:` step of every workflow is read, and the reading is recorded in the Validation section.
  - `spec-command.yml` keeps `grep … || true`, which handles grep's expected exit 1 on no match.
  - The jq pipelines in `labels-sync.yml` run after a successful plan file exists.
- **[The privileged workflows cannot be exercised by this pull request]** → Local runs of their step bodies against fixture snapshots and plans, and a readback after the merge on the next pull request carrying a `/spec` command.
- **[A crash in `/spec show` now posts a traceback into a pull request comment]** → The traceback holds only paths inside the base checkout and no credential, because the token lives in the environment and is never printed. It is visible where the maintainer will act on it.
- **[Removing a documented escape (`# DIVERGENCE:`) surprises a contributor]** → The docstring entry and the register row go in the same commit.

## Verification plan

Per What Changes bullet. The scratch repositories and stubs live under the session scratch directory, stay untracked, and are removed afterwards.

- **Copies check removed.**
  - In a disposable worktree, append a comment line to `scripts/sync_labels.py`: `just validate` passes, and before the change it failed naming the file.
  - `grep -n 'DIVERGENCE\|check_copies\|ORIGIN' scripts/validate_harness.py` is empty.
- **Synchronization register.** A readback of both rows against the files they name.
- **`spec_changes.py`.**
  - `--help` and each subcommand's `--help` exit 0; `related` is gone from the usage.
  - In a scratch OpenSpec repository with an archived change, an unarchived change with an open task, and one with no ticked task:
    - `check` exits 1 naming the unarchived change;
    - `--draft` exits 0 with a warning;
    - `--shape split` admits only the no-task change;
    - `archive --base main --head HEAD` refuses while a task is open, archives once every task is ticked, and changes nothing on an identical second run;
    - `show --change nope` exits 2, listing the related changes as code;
    - `--bogus` exits 2 naming it;
    - `labels --taxonomy` prints the five managed names.
  - A local stub HTTP server, reached through `--api-url`, serves the scratch repository's pulls, files, trees, and blobs:
    - `snapshot`, then `status`, `show`, and `labels` on its output, match the git source byte for byte;
    - a cap reached, a truncated tree, and a non-UTF-8 document each exit 1 naming the path and write nothing;
    - a list where an object is expected, and a 404 on a listed tree, each fail naming the endpoint and write nothing;
    - a snapshot file whose `changes_dir` differs, or which lacks a required key, fails naming the file.
  - A read-only `snapshot` against this pull request, followed by `status`, matches `status --base origin/main --head HEAD`.
  - `openspec` hidden from `PATH` makes `check` fail with the install hint.
- **`sync_labels.py`.**
  - `--help` exits 0; `--bogus` exits 2; `--prune` without `--apply` exits 2.
  - The dry-run plans of the old and the new script against `ryan-minato/skills` are byte-identical.
  - A stub `gh` makes each failure mode fail, naming `gh label list`, with no plan printed:
    - non-JSON output;
    - an object where a list is expected;
    - `gh` missing from `PATH`.
  - A `labels.json` with a numeric color, a duplicate name, and a non-list top level each exit 1 naming the file.
  - A second dry run gives identical output.
- **Workflow steps.**
  - The bodies of the `changed` step and the gate step run locally the way the workflow runs them:
    - a pull request event with an unknown base SHA fails;
    - a push event with no base or head passes and runs the job;
    - a pull request touching `openspec/` sets `run=true`;
    - a needs result with a failure exits 1;
    - one with only successes exits 0.
  - The `spec-command.yml` reply block runs with a stub script exiting 0, 2, 1, and with a Python traceback. A stub `gh` records the posted body each time, and the step's exit is 0, 0, 1, and 1.
  - The `spec-labels.yml` loop runs on a fixture plan with one add and one remove.
  - `just validate` passes: the job names parse and the case line is read.
- **Management-code rules; `meta/CONTEXT.md`.** A readback: the section states R1–R6 as the fail-fast philosophy aimed at fallbacks that postpone or hide unexpected errors, with deferral by design allowed, names where it applies, and records the stdlib and `python3` choice. The code-review text links it and keeps the privileged-job threat model. The CONTEXT bullet names the script-asset exception.
- **Lint.**
  - `just lint` passes.
  - In a disposable worktree, adding `try: pass\nexcept Exception: pass` to a repository script fails `just lint` with BLE001 and S110. The same lines in a skill's `scripts/` file do not.
  - `just validate` reports no ruff warning.
- **Archive command.** The command in `openspec/config.yaml` runs in the scratch repository and archives a complete change.
- **Everything.** `just check` and `git diff --stat origin/main...HEAD`.
