# Pipelines and jobs

Loaded when a pipeline or job must be read or its failure digested —
whether reached from an MR or directly. Never fetch a full job trace
unbounded: logs run to megabytes and flood the context. Everything here
reads; the only writes (retry, cancel, run) are marked and run only at
the user's explicit direction.

## From a failing pipeline to the error lines

1. **Identify the failing jobs.** glab:
   `glab ci get --merge-request N -R G/P -s failed -d -F json` (from an
   MR) or `glab ci get -R G/P -p ID -s failed -d -F json` (by pipeline
   id). `-s failed` limits the listing to failed jobs; `-d` adds job
   details. Record each failing job's `id`, `name`, `stage`, and
   `failure_reason`. MCP: read the MR's pipelines (18.4), then a
   pipeline's jobs (18.4).
2. **Fetch only the tail of each failed log.** The failure is almost
   always in the last lines. For a **finished** job:

   ```bash
   glab ci trace JOB_ID -R G/P | tail -n 100
   ```

   `glab ci trace` follows running jobs forever and will hang a
   non-interactive session — check the job is finished first (step 1
   shows its status). Equivalent without trace:
   `glab api "projects/:fullpath/jobs/JOB_ID/trace" | tail -n 100`.
   Or digest all failed jobs in one call with
   [scripts/pipeline_log_digest.py](scripts/pipeline_log_digest.py):

   ```bash
   python3 scripts/pipeline_log_digest.py --repo G/P --pipeline-id ID [--tail 50] [--hostname HOST]
   ```

   which prints one JSON object: pipeline status plus, per failed job,
   its stage, failure reason, and the ANSI-stripped last N trace lines.
   MCP alternative: the job-log capability (19.1) has **no tail
   parameter** and returns the whole trace; prefer glab for logs, and use
   the MCP tool only when glab is unavailable and the log is known to be
   small.
3. **Quote the error.** Extract the failing command and its error lines
   from the tail; raise the tail to 300 only when the error is not in the
   last 100 lines.

Done when: each failed job is named and its error lines are quoted, not
the whole log.

## Reads

| Task | glab command | MCP capability (min GitLab) |
|---|---|---|
| List pipelines | `glab ci list -R G/P -F json [-s failed]` | list pipelines (18.10) |
| Inspect one pipeline + jobs | `glab ci get -R G/P -p ID -d -F json` | read a pipeline's jobs (18.4) |
| Pipeline variables (Maintainer only) | `glab ci get -R G/P -p ID --with-variables` | — |

## Explicit writes (user-initiated only)

Run these solely on the user's explicit request, and report what was
triggered. The MCP pipeline-management capability (18.10) covers
run/retry/cancel where present, but glab stays the recommended path.

```bash
glab ci retry JOB_ID -R G/P             # retry one job (or by job name with -p/-b)
glab ci cancel pipeline ID -R G/P       # cancel a pipeline
glab ci cancel job JOB_ID -R G/P        # cancel one job
glab ci run -R G/P -b BRANCH            # start a new pipeline for a branch
```

## CI config lint

Validates `.gitlab-ci.yml` against the instance (catches include and
rule errors local YAML parsing cannot):

```bash
glab ci lint [PATH] -R G/P
```

## Artifacts

```bash
glab ci artifact REF JOB_NAME -R G/P    # download the last pipeline's artifacts for a ref
```

For a specific job id instead of a ref/name pair:
`glab api "projects/:fullpath/jobs/JOB_ID/artifacts" > artifacts.zip`.
Download into a scratch directory and read only the files the task needs.

## REST fallback tier

The read rows map to `rest_read.py pipelines`, `jobs`, and
`pipeline-failures --tail 50` — see [rest-fallback.md](rest-fallback.md).
Job traces on private projects always need a token; even on public
projects, trace access may require membership — the scripts report
per-job access errors as data.

## Gotchas

- `glab ci list` shows GitLab CI pipelines only; statuses posted by
  external CI systems live on commits, not here.
- A successful pipeline digests to an empty `failed_jobs` list — that is
  data, not a failure.
