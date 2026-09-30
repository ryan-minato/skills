## Context

See proposal.md for motivation. The rules are `## Management code` in `.agents/knowledge/skill-quality.md` (R1–R6). The reference rewrites are `scripts/spec_changes.py` and `scripts/sync_labels.py` from the archived `management-code-harness` change.

**Current state:**
- **`scripts/check_pr_policy.py`** (270 lines). `pr / policy`, a required check, runs `--event "$GITHUB_EVENT_PATH"` from a checkout of the base commit. Agents run `--pr N` as a local dry run.
  - `load_event` reads every key through `.get` with a default (`head`, `base`, `user`, `number`, `title`, `draft`).
  - A payload that is not an object raises `AttributeError`.
  - Missing SHAs are noticed only after the body checks.
  - `load_pr_via_gh` parses `gh pr view` output without a guard and drops gh's stderr, and `commit_subjects` drops git's stderr.
  - The docstring documents exit 2 for "bad input, unreadable template, or git failure".
- **`scripts/sync_issue_metadata.py`** (182 lines). `issues / triage` runs `--event "$GITHUB_EVENT_PATH" --apply` from the default branch.
  - `load_label_names` skips non-object items and turns a missing name into `""`.
  - `plan` filters labels with `isinstance` and reads `number` through `.get`, which crashes only under `--apply`.
  - The `--issue` path parses `gh api` output without a guard.
  - A DELETE is treated as done when any `404` appears in stderr.
  - The docstring documents exit 2 for "unreadable input, or an API failure", and exit 1 for an unmapped form answer, with the rest of the plan still applied.
- **`scripts/validate_harness.py`** (485 lines).
  - `check_labels` iterates `labels.json` without a list or object check and reads `str(item.get("name", ""))`. `check_spec_labels` parses the file again with `item["name"]`.
  - It parses the `spec_changes.py labels --taxonomy` output without a guard.
  - `read` ends in `return ""  # unreachable`.
  - A missing `git` raises `FileNotFoundError`.
  - `fail` exits 2. The script takes no arguments.
- **`.agents/skills/change-workflow/scripts/run_log_digest.py`** (178 lines), named in `change-workflow/SKILL.md` and `github-checks.md` `## Diagnosing a red check`.
  - It reads every run and job key through `.get` with a default, and a failed `--log-failed` fetch becomes a stderr note.
  - Its docstring gives the path as `scripts/run_log_digest.py`.
  - It is outside ruff's scope, and fails `ruff check` (UP037) and `ruff format --check` today.
  - `skills/meta/meta-github-workflow/assets/run_log_digest.py` is a management-code version of the same role. It records `log_error`, but reads each job's keys without a shape check.
- **Lint scope.** `just lint` and the pre-commit ruff `files` pattern cover `scripts/`, `skills/*/*/scripts`, and `skills/(meta|sdd)/*/assets`. `.agents/skills/` holds one real project-skill script, `change-workflow/scripts/run_log_digest.py`, beside symlinks to every public skill. Those skills' scripts are exempt from BLE001 and S110 only under their root-anchored `skills/*/*/scripts/**` path in `ruff.toml`.

**Data contracts** (checked 2026-09-30 against `octokit/webhooks` `payload-schemas/api.github.com` and `github/rest-api-description` `api.github.com.json` 1.1.4):
- webhook `pull_request`: `number`, `title`, `body`, `draft`, `head`, `base`, and `user` are required. `body` is `string | null`. `head.repo` is `repository | null`.
- webhook `issue`: `number`, `body`, and `user` are required, and `body` is `string | null`. `labels` is optional, an array of label objects whose `name` is required.
- REST `issue` (`gh api repos/{repo}/issues/{n}`): `number` and `labels` are required, the label items `string` or a label object. `body` is optional and nullable.
- `gh pr view --json` and `gh run view --json`: every requested field is present. Run 36371813053 shows jobs carrying `databaseId`, `name`, `status`, `conclusion`, and `steps[name, conclusion]`.

**Binding rules:**
- **Commits.** Every commit passes `just check` on its own, because branches are rebase-merged.
- **Base-branch runs.** `pr / policy` and `issues / triage` run the default branch's script, so this pull request cannot exercise them live.
- **`validate_harness.check_ruff`** reads the pre-commit ruff `files` pattern with `\S+` and must still match `skills/core/x/scripts/y.py`.
- **`validate_harness.labels_case_line`** reads the literal case line of `spec-labels.yml`, which this change does not touch.

## Placement

| What Changes bullet | File and section | Proof |
|---|---|---|
| `check_pr_policy.py` | the script: `load_event`, `load_pr_via_gh`, `commit_subjects`, the SHA check in `check`, `fail`, `strip_comments`, the docstring's exit codes | script harness below |
| `sync_issue_metadata.py` | the script: `load_label_names`, `load_issue_from_event`, `load_issue_via_gh`, `plan`, `run_gh`, `fail`, the docstring's exit codes | script harness below |
| `validate_harness.py` | the script: `fail`, `read`, `check_labels`, `check_spec_labels`, the `__main__` git check, the docstring's exit codes | script harness below; `just validate` |
| `run_log_digest.py` | `.agents/skills/change-workflow/scripts/run_log_digest.py`, whole file, including the docstring's usage path and output shape | script harness below |
| Exit codes | the three docstrings above and the digest's; `.agents/knowledge/github-checks.md`, the `issues / triage` row's Healthy run cell | readback; script harness below |
| Lint | `justfile` `lint` recipe; `.pre-commit-config.yaml` ruff-check and ruff-format `files`; `ruff.toml` header comment; `AGENTS.md` Validation row "Python style"; `ARCHITECTURE.md` `## Quality Gates` lint bullet; `.agents/knowledge/skill-quality.md` `## Management code`, last paragraph | `just lint`; negative test below; `just validate` (no ruff warning) |

## External impact

- **`.github/workflows/pr-policy.yml` and `issue-triage.yml`.** Their command lines stay as they are. A readback confirms that each still calls the flags the rewritten script accepts. The live behavior changes on the first pull request and the first issue event after the merge, and both runs are read back then.
- **`.agents/skills/change-workflow/SKILL.md`** and **`github-checks.md` `## Diagnosing a red check`** keep the digest's invocation. A readback confirms that the flags still match.
- **`.agents/skills/code-review/SKILL.md`** already states "Exit 2 is guaranteed only for bad arguments; an unexpected crash exits 1". A readback confirms that no edit is needed.
- **`.agents/knowledge/harness-maintenance.md`** registers no pair these scripts join, and its `check_pr_policy` line (it reads the template itself) stays true. A readback confirms this.
- **README pairs.** They mention neither these scripts nor the lint scope (`grep` is empty), so they are not touched.
- **Public skills.** None is touched. `meta-github-workflow`'s script and asset stay byte-for-byte, and `git diff --stat origin/main...HEAD -- skills` is empty.

## Decisions

- **Check each interface once, where its data enters, then read directly** (every script bullet).
  - The interfaces are the event payload file, the output of `gh pr view`, `gh api`, and `gh run view`, `git log` and `git rev-parse`, `labels.json`, the PR template, and the `spec_changes.py labels --taxonomy` output.
  - Only the keys a script reads are checked, each against its documented type.
  - Each script carries its own small check helper, as `spec_changes.py` and `sync_labels.py` do.
  - Rejected: a shared module. The scripts run standalone with `python3`, and `pr / policy` runs from a base-commit checkout, where a shared import would couple the gate to further files.
- **An interface failure exits 1; exit 2 stays only for bad arguments** (exit-codes bullet; maintainer decision).
  - An unreadable `--event`, `--labels`, or `--template` file counts as an interface failure: the file is data another step or commit wrote, and the argument itself parsed. This reading is the author's, not the maintainer's (see Open Questions).
  - `validate_harness.py` takes no arguments, so it exits 0 or 1 only.
  - The `issues / triage` row then describes exit 1 as either an unmapped answer (the rest of the plan printed and applied) or an interface failure (nothing applied), told apart by the stderr message.
  - This matches `sync_labels.py` and the code-review contract.
  - Rejected: keeping the documented exit 2 as a declared deviation, which leaves three scripts disagreeing with that contract. Rejected: exit 2 for an unreadable path argument, which splits one interface's failures across two codes.
- **The SHA check moves to the load** (`check_pr_policy.py` bullet; author decision, see Open Questions).
  - `base.sha` and `head.sha` are required strings in the webhook schema and `baseRefOid`/`headRefOid` are requested gh fields, so a missing SHA is a broken interface and fails where the payload is loaded, before any body check.
  - This covers a fork payload too: today a fork payload without SHAs is judged on its body and title alone, because the commit-range step is skipped for forks; after the change it fails.
  - Rejected: keeping the check inside the commit-range step, which reads a guaranteed key through a default on the fork path.
- **Documented nullables stay as domain logic, each written explicitly with a comment citing the schema** (`check_pr_policy.py` and `sync_issue_metadata.py` bullets; maintainer decision).
  - `pull_request.body` and `issue.body` (null means an empty body). A REST issue from `gh api` may also omit `body`, which reads the same way; a webhook issue without `body` is a broken payload and fails.
  - `pull_request.head.repo`: null for a deleted fork is read as "not a fork", as today. The workflow's own fetch step (`if: !…head.repo.fork`) makes the same reading, so the commit range is fetched and checked.
  - A webhook issue without `labels` has no labels.
  - A REST issue label is a string (the name) or an object with a string `name`. An object without a string `name` fails, naming `gh api` and the value.
  - Rejected: failing on any of these. They are states GitHub documents, not broken interfaces, and a red required check on a deleted fork would block a legitimate pull request.
- **Designed deferrals stay, written where they happen** (every script bullet).
  - `validate_harness.py` collects every finding and fails once.
  - `check_pr_policy.py` collects findings.
  - `sync_issue_metadata.py` prints and applies the rest of the plan when an answer is unmapped.
  - The digest's "no job concluded failure, so digest every completed job" branch stays.
  - The optional-by-design inputs of `validate_harness.py` stay: a form without `labels:`, a missing `dependabot.yml`, and a job named by its id.
  - Rejected: turning each deferral into fail-fast, for example stopping `validate_harness.py` at its first finding, or having `sync_issue_metadata.py` apply nothing when one answer is unmapped. Each deferral is designed behavior, which the fail-fast paragraph of `## Management code` keeps, and changing it would change output on valid input.
- **The DELETE-404 idempotence stays and is narrowed** (`sync_issue_metadata.py` bullet; author decision, see Open Questions). A DELETE whose gh stderr carries `HTTP 404` (gh 2.98.0 prints `gh: Not Found (HTTP 404)`) means already removed. Any other failure exits 1 naming the command. Rejected: matching `404` anywhere in stderr, which also matches an issue or label name.
- **`run_log_digest.py` records `log_error` per job** (digest bullet).
  - Maintainer decision: a job whose failed-step log cannot be fetched carries a `log_error` field with gh's message (or its exit status), and its `log_tail` is empty. The digest does not fail.
  - Author decision (see Open Questions): every `failed_jobs` entry carries the key, `null` when the log was fetched. The shape is then the same in every entry, which is what the meta-github-workflow asset already emits.
  - This departs from the issue's acceptance that output on valid input is unchanged: every digest of a real failed run gains `"log_error": null` per job. The old-versus-new proof therefore compares the new output with `log_error` removed.
  - Rejected: failing the whole digest, which loses the diagnosis of every other job when one never started. Rejected: keeping only the stderr note, which drops the reason from the digest an agent reads. Rejected: adding the key only on failure, which keeps a fully fetched digest byte-identical but gives consumers two entry shapes.
- **The digest starts from the meta-github-workflow asset and owns its copy** (digest bullet; R1).
  - It adds the per-job shape check the asset lacks.
  - No check binds the two, and they may drift.
  - Rejected: importing or copying the asset under a sync rule, which R1 forbids.
- **Lint scope names one directory** (lint bullet; maintainer decision).
  - `.agents/skills/change-workflow/scripts` goes into the `justfile` recipe and, as one more alternative, into the pre-commit `files` pattern.
  - `ruff.toml` needs no per-file change. The negated `!skills/*/*/scripts/**/*.py` entry already applies BLE001 and S110, and turns SIM105 off, for the path.
  - Rejected: `.agents/skills/*/scripts`. The shell glob expands through the symlinks to 33 public-skill scripts (34 files with the project skill's own), and the root-anchored exemption does not cover them there: `ruff check --select BLE001,S110` over that glob reports 8 findings today.
  - Rejected: `.agents/skills`. It reaches only the project skill's file today, because ruff's directory walk does not follow the symlinked skill directories (`ruff check --show-files .agents/skills` lists one file). But the scope would then rest on walker behavior rather than a name, and it would take in any future `.py` under a project skill that is not management code.
  - Rejected: a `validate_harness` check that every non-symlink project-skill script directory is linted. One such directory exists, so revisit when a second appears.
  - Rejected: leaving the script unlinted, which leaves R5 without machine enforcement there.
- **The dependabot author mismatch is out of scope** (maintainer decision). `gh pr view --json author` reports `app/dependabot` (verified on PR 91), while the event's `user.login` is `dependabot[bot]`, the only value in `BOT_AUTHORS`, so a local `--pr` dry run of a bot pull request reports findings that CI skips. Fixing it changes output on valid input, so it goes to a separate bug. Rejected: fixing it here, which would need its own exemption from the unchanged-output proof.
- **One commit per script** (every bullet).
  - Each script's rewrite, its docstring, and the exit-code documentation it owns (`github-checks.md` with `sync_issue_metadata.py`) land together as a `fix` commit.
  - The lint-scope widening lands in its own commit after the digest's, because the digest fails `ruff format --check` until it is rewritten.
  - Rejected: one commit for all four, which the one-change-per-commit rule and the issue both refuse.

## Risks / Trade-offs

- **[`pr / policy` is required, and a too-strict check turns a legitimate but rare payload red]** → Only the keys the script reads are checked, against the published schemas above. A `head.repo: null` fixture and a `body: null` fixture must pass, and the first pull request after the merge is read back.
- **[The two base-branch workflows cannot be exercised by this pull request]** → Old-versus-new runs on recorded payloads before ready, and a readback of the first `pr / policy` and `issues / triage` runs after the merge.
- **[Recorded REST objects differ from webhook payloads]** → The REST `pull-request` schema marks `draft` optional and `head.repo` non-null, while the webhook marks `draft` required and `head.repo` nullable. The fixtures are REST objects wrapped as `{"pull_request": …}`, so each fixture is checked to carry every key the webhook requires, and the webhook-only states (`head.repo: null`, labels absent) are produced by mutation.
- **[Exit 1 on `issues / triage` now also means nothing was applied]** → The stderr message names the interface, and the `github-checks.md` row states both meanings.
- **[The pre-commit pattern stops matching skill scripts]** → `just validate` warns through `check_ruff`, and the pattern gains an alternative without whitespace.
- **[Another open change edits the same lint lines]** (#98, if it widens the lint scope) → The content does not conflict, only the text. Whichever lands second rebases and re-runs `just lint`.
- **[The taxonomy guard meets a changed output shape]** (#96, #100, or #101 changing `labels --taxonomy`) → The guard is written against the current shape, and whichever lands second rebases.

## Verification plan

Per What Changes bullet. Fixtures, the stub `gh`, and disposable worktrees live under the session scratch directory, stay untracked, and are removed afterwards. **Old** means the script at `origin/main`, run from a disposable worktree.

**Fixtures:**
- **Pull request payloads.** `gh api repos/ryan-minato/skills/pulls/<n>` for a ready in-repo pull request (95), a draft, and dependabot pull requests (91, 83), each wrapped as `{"pull_request": …}`. Each is checked to carry the keys the webhook schema requires. The heads are fetched with `git fetch origin refs/pull/<n>/head` so the commit range resolves.
- **Issue payloads.** `gh api repos/ryan-minato/skills/issues/<n>` for issue 97 and an issue that still carries `status/needs-triage`, each wrapped as `{"issue": …, "repository": {"full_name": "ryan-minato/skills"}}`.
- **Mutations.** `jq` mutations of each recorded object.
- **Stub `gh`.** A shell script on `PATH` that answers by argv with canned stdout, stderr, and exit status, and logs every call.

Every malformed case must exit 1 with stderr naming the interface (the payload file and key path, or the command) and the value received, print no findings or plan, and make no write call.

- **`check_pr_policy.py`.**
  - `--help` exits 0; `--bogus` exits 2.
  - With `--event`, each of these fails:
    - a payload that is a JSON list, or not JSON;
    - `pull_request` missing;
    - `head` missing (today this continues with a default);
    - `head.sha` missing or null;
    - `base` missing, and `base.sha` missing, on an in-repo payload and on a fork payload (`head.repo.fork: true`; today the fork payload continues);
    - a `head.repo` object without `fork` (today this reads as not a fork);
    - `title` missing (today this continues with `""`);
    - `user` missing (today this continues);
    - `number` missing;
    - `draft` a string.
  - With `--event`, these run as today: `body: null`, and `head.repo: null`, which is checked over the commit range.
  - With `--pr` and the stub gh, each of these fails naming `gh pr view`: non-JSON, an object without `headRefOid`, exit 1 with a stderr message, and gh missing from `PATH`.
  - An unknown base SHA fails naming `git log <range>` with git's message.
  - A missing template, and one without the four role headings, fail naming the file.
  - On valid input, old and new stdout, stderr, and exit status are byte-identical on every recorded payload with `--event`, and with `--pr` on 95 and 91 (91 keeps the `app/dependabot` findings).
  - Repeat: a second run with the same `--event` payload, and a second `--pr` run against the same stub answers, are byte-identical to the first.
- **`sync_issue_metadata.py`.**
  - `--help` exits 0; `--bogus` exits 2; `--issue 1` without `--repo` exits 2.
  - With `--event`, each of these fails:
    - `issue` missing;
    - `repository.full_name` missing;
    - `number` missing (today this crashes only under `--apply`);
    - `body` missing (today this continues with `""`);
    - `labels` an object;
    - a label object without `name` (today it is skipped).
  - With `--event`, these run as today: `body: null`, and `labels` absent.
  - With `--issue` and the stub gh, each of these fails naming `gh api`: non-JSON, a list, an object without `number` (today this continues and crashes only under `--apply`), and a label that is a number. A label given as a string is read as its name. An object without `body` runs as an empty body, which the REST schema allows.
  - A `labels.json` that is an object, an item without `name`, and a numeric `name` each fail naming the file and the index.
  - With `--apply` and the stub gh:
    - a failing POST fails naming the command;
    - a DELETE answered with `gh: Not Found (HTTP 404)` succeeds;
    - a DELETE failing with any other message fails.
  - Without `--apply`, the stub logs no write call.
  - Repeat: a second dry run on the same input is byte-identical to the first. `--apply` runs twice: against a stub gh that reflects the first run's writes, the second run plans no write; against a stub that keeps answering the original labels, the second run's DELETEs meet `gh: Not Found (HTTP 404)` and it exits as the first did.
  - On valid input, old and new dry-run output and exit status are byte-identical with `--event` on every recorded payload, and with `--issue --repo` against the same issues (read-only).
- **`validate_harness.py`.** In a disposable worktree:
  - a `labels.json` whose top level is an object, and one whose item lacks `name`, each fail naming `.github/labels.json` and the index;
  - a stub `spec_changes.py` printing non-JSON, `{}`, and `{"managed": "x"}` each fail naming the command;
  - `git` hidden from `PATH` fails naming git;
  - `grep -n 'unreachable' scripts/validate_harness.py` is empty.
  - On the current tree, old and new output and exit status are byte-identical, and a second run of the new script is identical to the first.
- **`run_log_digest.py`.**
  - `--help` exits 0; `--bogus` and `--run-id 0` exit 2.
  - With the stub gh, each of these fails naming `gh run view` and the value:
    - non-JSON, a list, an object without `jobs`;
    - a run without `databaseId` (today the digest falls back to the `--run-id` argument), `status`, or `conclusion`;
    - a job without `databaseId`, `name`, `status`, `conclusion`, or `steps`;
    - a step without `conclusion`.
  - gh missing from `PATH` fails naming gh.
  - A failing `--log-failed` call yields that job's `log_error` with gh's message and an empty `log_tail`, and the run exits 0.
  - On real failed runs 36371813053 and 36558570731, the old stdout equals the new stdout with `log_error` removed (`jq 'del(.failed_jobs[].log_error)'`), and the exit statuses match. A job's `log_error` is `null` exactly where the old script printed no "no failed-step log" note. A second run gives identical output.
  - Neither real run takes the fallback branch, so a stub-gh run whose conclusion is `failure` and whose jobs are only `cancelled` or `skipped` is run old and new: the stdout matches under the same `jq` filter, both print the fallback note, and the exit statuses match.
- **Exit codes.** A readback shows each docstring states 0, 1, and 2 as above. The `issues / triage` row states both meanings of exit 1. No malformed case above exits 2.
- **Lint.**
  - `just lint` passes. `ruff check --show-files` over the recipe's paths lists `.agents/skills/change-workflow/scripts/run_log_digest.py` and no other path under `.agents/skills/`.
  - In a disposable worktree, adding `try: pass` and `except Exception: pass` to the digest fails `just lint` with BLE001 and S110, and a pre-commit run on that file fails the same way.
  - `just validate` reports no ruff warning. The `AGENTS.md`, `ARCHITECTURE.md`, `ruff.toml`, and `skill-quality.md` lines are read back against the recipe.
- **External impact.** The readbacks listed there, and `git diff --stat origin/main...HEAD -- skills` is empty.
- **Everything.** `just check` passes on every commit (`git rebase --exec 'just check' origin/main`), then `git diff --stat origin/main...HEAD`.

Skipped:
- Live runs of `pr / policy` and `issues / triage` before the merge, because both run from the default branch. They are replaced by the recorded-payload runs above and read back after the merge.
- Behavioral skill tests, because no `SKILL.md` changes behavior and a repository change has no domain.

## Open Questions

The maintainer settled the exit codes, `log_error`, the nullables, the lint scope, and the dependabot follow-up. These choices were made by the author, not the maintainer, and each is for the maintainer to confirm on the pull request. Either answer keeps the placement above, and only the matching verification case changes:
- **`log_error` present on every entry** (Decisions, `log_error`): `null` when the log was fetched, which changes every digest's output. The alternative is the key only on failure.
- **An unreadable `--event`, `--labels`, or `--template` file exits 1** (Decisions, exit codes), where the docstrings document exit 2 today.
- **The SHA check moves to the load** (Decisions, SHA check), so a fork payload without SHAs now fails.
- **The DELETE-404 match narrows to gh's `HTTP 404`** (Decisions, DELETE-404), so any other DELETE failure now exits 1.
