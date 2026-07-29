# CI runs, checks, and logs

Loaded when a workflow run, a check, or its logs must be read — whether
reached from a PR's failing checks or directly from run history. Use the
column chosen in "Choose your path"; substitute `O/R`, `RUN_ID`, `JOB_ID`.
Everything here reads; the only writes (rerun, cancel) are marked and run
only at the user's explicit direction.

## From a failing PR to the error lines

1. **Identify the failing checks.** MCP: the capability that reads a PR's
   check runs. gh: `gh pr checks N -R O/R --json
   name,state,bucket,link,workflow` — the failing checks are the rows with
   `bucket` = `fail`.
2. **Map each failing check to its Actions run.** Preferred: the check's
   `link` URL contains the ids directly, in the form
   `https://github.com/O/R/actions/runs/RUN_ID/job/JOB_ID`. If the link is
   missing or not an Actions URL, list recent runs on the PR's branch:
   `gh run list -R O/R --branch BRANCH --limit 10 --json
   databaseId,workflowName,conclusion` and pick the run whose
   `workflowName` matches the failing check's `workflow` and whose
   `conclusion` is `failure`; its `databaseId` is `RUN_ID`.
3. **Fetch failed logs only** — see the hard rule below.

## Runs and jobs

| Task | MCP capability | gh |
|---|---|---|
| List workflow runs | list workflow runs | `gh run list -R O/R --limit 20 --json databaseId,displayTitle,workflowName,headBranch,status,conclusion,createdAt` |
| Filtered run listing | list workflow runs — check the tool's own description for where the workflow id and branch/status/event filters go | `gh run list -R O/R --workflow NAME --branch BR --status failure --limit 20 --json databaseId,displayTitle,headBranch,status,conclusion,createdAt` |
| List workflows | list workflows | `gh workflow list -R O/R` |
| Inspect one run | read one workflow run | `gh run view RUN_ID -R O/R` |
| List a run's jobs | list a run's jobs | `gh run view RUN_ID -R O/R --json jobs` |
| Failed-log excerpt | read job logs, failed-only, with a tail limit (about 100 lines) | [scripts/run_log_digest.py](scripts/run_log_digest.py): `python3 scripts/run_log_digest.py --repo O/R --run-id RUN_ID [--tail 50]` |
| Run timing/usage | read a run's timing/usage | `gh api repos/O/R/actions/runs/RUN_ID/timing` |

On the MCP path, the job-log capability comes in two shapes — all failed
jobs in the run (run id + failed-only + a tail limit of about 100 lines),
or a single job (job id + the same tail limit). Never combine a single-job
id with the failed-only switch. On the gh path, `gh run view RUN_ID -R O/R
--log-failed | tail -n 100` (or `--job JOB_ID --log-failed`) works too;
the digest script wraps the same rule and emits one JSON object with the
run's status, failed jobs, failed steps, and each failed job's log tail.

**Hard rule:** never fetch a full run log — full logs can be megabytes and
will drown the context. Always request failed-only output and tail it;
start with 100 lines and raise the tail (for example to 300) only when the
actual error is not within it.

Done when (for a "why did it fail" task): the failing job and step are
named and the relevant error lines are quoted.

## Artifacts

| Task | MCP capability | gh |
|---|---|---|
| List a run's artifacts | list a run's artifacts | `gh api repos/O/R/actions/runs/RUN_ID/artifacts` |
| Download one artifact | — (no MCP download; use gh) | `gh run download RUN_ID -R O/R -n ARTIFACT_NAME -D DIR` |

Download into a scratch directory and read only the files the task needs;
artifacts can be large.

## Watch a live run

Only when the user explicitly asks to wait for a run — this blocks until
the run finishes:

```bash
gh run watch RUN_ID -R O/R --exit-status
```

`--exit-status` makes the command exit non-zero if the run fails, so the
outcome is machine-readable.

## Rerun / cancel (writes)

Rerunning or canceling a run is a write. Confirm with the user first, then
`gh run rerun RUN_ID -R O/R` (add `--failed` to rerun only failed jobs) or
`gh run cancel RUN_ID -R O/R` explicitly at their direction.

## REST fallback tier

The read rows map to `rest_read.py runs`, `run`, `jobs`, and
`run-failures --tail 50` — see [rest-fallback.md](rest-fallback.md).
Actions **log text** requires a token even for public repositories (GitHub
answers 403 "Must have admin rights"); without one, `run-failures` still
names the failed jobs and steps from the jobs API.

## Gotchas

- `gh run list` shows GitHub Actions workflow runs only; check runs
  reported by external CI apps do not appear there — they still show in
  `gh pr checks`.
- `--workflow` accepts the workflow name, its file name (`ci.yml`), or its
  id from `gh workflow list`.
