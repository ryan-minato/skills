#!/usr/bin/env python3
"""Digest a GitLab CI pipeline's failures into one compact JSON object.

Part of this project's GitLab workflow skill. Shells out to the glab CLI
(installed and authenticated) and never fetches full pipeline logs: it
reads only the traces of failed jobs and keeps the last N lines per job —
cleaned of ANSI escape codes, GitLab section markers, and carriage-return
progress frames — so multi-megabyte logs stay out of agent context.

Usage, from the repository root:
    python3 <project skill>/scripts/pipeline_log_digest.py --repo GROUP/PROJECT \\
        --pipeline-id ID [--tail N] [--hostname HOST]

Output: a single JSON object on stdout:
    {"pipeline_id": ..., "status": ...,
     "failed_jobs": [{"job_id": ..., "name": ..., "stage": ...,
                      "failure_reason": ..., "log_tail": [...], "log_error": null}]}

A job whose trace glab cannot return (a private project, an expired
trace) keeps an empty log_tail and carries glab's message in log_error,
so the digest still reports every failed job. Every other glab call is
checked where its output enters: a non-zero exit, output that is not JSON,
or JSON of another shape ends the script naming the endpoint.

Exit codes:
    0  digest produced (a successful pipeline digests to an empty
       failed_jobs list — that is data, not a failure)
    1  glab missing, glab failed on the pipeline, or glab printed unexpected output
    2  bad arguments
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.parse

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\^_]")
SECTION_MARKER_RE = re.compile(r"section_(?:start|end):\d+:[\w.-]+(?:\[[^\]]*\])?")
JOBS_PER_PAGE = 100


def glab_api(endpoint: str, hostname: str | None) -> subprocess.CompletedProcess:
    command = ["glab", "api", *(["--hostname", hostname] if hostname else []), endpoint]
    try:
        return subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit(
            "pipeline_log_digest: error: `glab` is not installed; install the GitLab CLI and run `glab auth login`."
        )


def glab_json(endpoint: str, hostname: str | None, kind: type) -> dict | list:
    """One `glab api` read whose output must be JSON of ``kind``; anything else ends the script."""
    result = glab_api(endpoint, hostname)
    if result.returncode != 0:
        sys.exit(
            f"pipeline_log_digest: error: `glab api {endpoint}` exited {result.returncode}: {result.stderr.strip()}\n"
            "Check that the pipeline exists (`glab ci list`) and that glab is authenticated (`glab auth status`)."
        )
    try:
        data = json.loads(result.stdout)
    except ValueError as exc:
        sys.exit(
            f"pipeline_log_digest: error: `glab api {endpoint}` did not print JSON ({exc}): {result.stdout[:200]!r}"
        )
    if not isinstance(data, kind):
        expected = "a JSON object" if kind is dict else "a JSON list"
        sys.exit(f"pipeline_log_digest: error: `glab api {endpoint}` printed {result.stdout[:200]!r}, not {expected}.")
    return data


def clean_trace(text: str) -> list[str]:
    """Strip ANSI escapes, GitLab section markers, and CR progress frames."""
    text = SECTION_MARKER_RE.sub("", ANSI_ESCAPE_RE.sub("", text))
    # Progress lines rewrite themselves with \r; keep each line's final state.
    return [line.rsplit("\r", 1)[-1].rstrip() for line in text.split("\n")]


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Digest the failed jobs of a GitLab CI pipeline into one small JSON object (failed jobs, failure "
            "reasons, last N cleaned log lines each). Requires an installed and authenticated glab CLI."
        ),
        epilog="Exit codes: 0 digest produced (empty failed_jobs on a green pipeline), 1 glab failed, 2 bad arguments.",
    )
    parser.add_argument("--repo", required=True, help="project as its full path GROUP[/SUBGROUP]/NAME")
    parser.add_argument("--pipeline-id", required=True, type=positive, help="numeric pipeline id")
    parser.add_argument("--tail", type=positive, default=50, help="log lines to keep per failed job (default: 50)")
    parser.add_argument("--hostname", help="self-managed GitLab host, passed through as `glab api --hostname`")
    args = parser.parse_args()
    parts = args.repo.split("/")
    if len(parts) < 2 or not all(parts):
        parser.error(
            f"--repo must be the full GROUP[/SUBGROUP]/NAME path (got {args.repo!r}), e.g. --repo gitlab-org/cli"
        )

    project = urllib.parse.quote(args.repo, safe="")
    pipeline = glab_json(f"projects/{project}/pipelines/{args.pipeline_id}", args.hostname, dict)
    jobs = glab_json(
        f"projects/{project}/pipelines/{args.pipeline_id}/jobs?scope[]=failed&per_page={JOBS_PER_PAGE}",
        args.hostname,
        list,
    )
    if len(jobs) == JOBS_PER_PAGE:
        print(f"note: the first {JOBS_PER_PAGE} failed jobs are digested; the pipeline may have more.", file=sys.stderr)

    failed_jobs = []
    for job in jobs:
        trace = glab_api(f"projects/{project}/jobs/{job['id']}/trace", args.hostname)
        failed_jobs.append(
            {
                "job_id": job["id"],
                "name": job["name"],
                "stage": job["stage"],
                "failure_reason": job["failure_reason"],
                "log_tail": clean_trace(trace.stdout)[-args.tail :] if trace.returncode == 0 else [],
                "log_error": None
                if trace.returncode == 0
                else (trace.stderr.strip() or f"glab exited {trace.returncode}"),
            }
        )

    digest = {"pipeline_id": pipeline["id"], "status": pipeline["status"], "failed_jobs": failed_jobs}
    json.dump(digest, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
