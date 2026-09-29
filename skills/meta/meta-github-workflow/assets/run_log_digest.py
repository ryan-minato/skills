#!/usr/bin/env python3
"""Digest a GitHub Actions run's failures into one compact JSON object.

Part of this project's GitHub workflow skill. Shells out to the gh CLI
(installed and authenticated) and never fetches full run logs: it reads
only the failed-step logs of failed jobs and keeps the last N lines per
job, so multi-megabyte logs stay out of agent context.

Usage, from the repository root:
    python3 <project skill>/scripts/run_log_digest.py --repo OWNER/REPO --run-id ID [--tail N]

Output: a single JSON object on stdout:
    {"run_id": ..., "status": ..., "conclusion": ...,
     "failed_jobs": [{"name": ..., "job_id": ..., "failed_steps": [...],
                      "log_tail": [...], "log_error": null}]}

A job whose failed-step log gh cannot return (a job that never started
has none) keeps an empty log_tail and carries gh's message in log_error,
so the digest still reports every failed job. Diagnostics go to stderr.

Exit codes:
    0  digest produced (a successful run digests to an empty
       failed_jobs list — that is data, not a failure)
    1  gh missing, gh failed on the run, or gh printed unexpected output
    2  bad arguments
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

REPO_SLUG = re.compile(r"^[^/\s]+/[^/\s]+$")


def gh(*args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["gh", *args], capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("run_log_digest: error: `gh` is not installed; install the GitHub CLI and run `gh auth login`.")


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Digest the failed jobs of a GitHub Actions run into one small "
            "JSON object (failed jobs, failed steps, last N log lines each)."
        ),
        epilog="Exit codes: 0 digest produced (empty failed_jobs on a green run), 1 gh failed, 2 bad arguments.",
    )
    parser.add_argument("--repo", required=True, help="repository as OWNER/REPO")
    parser.add_argument("--run-id", required=True, type=positive, help="numeric workflow run id")
    parser.add_argument("--tail", type=positive, default=50, help="log lines to keep per failed job (default: 50)")
    args = parser.parse_args()
    if not REPO_SLUG.match(args.repo):
        parser.error(f"--repo must be OWNER/REPO (got {args.repo!r}), e.g. --repo octocat/hello-world")

    view = gh("run", "view", str(args.run_id), "-R", args.repo, "--json", "databaseId,conclusion,status,jobs")
    if view.returncode != 0:
        sys.exit(
            f"run_log_digest: error: `gh run view {args.run_id}` exited {view.returncode}: {view.stderr.strip()}\n"
            f"Check that the run exists (`gh run list -R {args.repo} --limit 20`) and that gh is authenticated."
        )
    try:
        run = json.loads(view.stdout)
        jobs = run["jobs"]
    except (ValueError, KeyError, TypeError) as exc:
        sys.exit(
            f"run_log_digest: error: `gh run view --json` printed unexpected output ({exc!r}): {view.stdout[:200]!r}"
        )

    failed = [job for job in jobs if job["conclusion"] == "failure"]
    if not failed and run["conclusion"] == "failure":
        # A run can fail with no job concluding "failure" (cancelled jobs, a
        # startup failure); every completed job is then worth a look.
        failed = [job for job in jobs if job["status"] == "completed"]
        print(f"note: no job concluded failure; digesting all {len(failed)} completed job(s).", file=sys.stderr)

    failed_jobs = []
    for job in failed:
        log = gh("run", "view", "-R", args.repo, "--job", str(job["databaseId"]), "--log-failed")
        failed_jobs.append(
            {
                "name": job["name"],
                "job_id": job["databaseId"],
                "failed_steps": [step["name"] for step in job["steps"] if step["conclusion"] == "failure"],
                "log_tail": log.stdout.splitlines()[-args.tail :] if log.returncode == 0 else [],
                "log_error": None if log.returncode == 0 else (log.stderr.strip() or f"gh exited {log.returncode}"),
            }
        )

    digest = {
        "run_id": run["databaseId"],
        "status": run["status"],
        "conclusion": run["conclusion"],
        "failed_jobs": failed_jobs,
    }
    json.dump(digest, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
