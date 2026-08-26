#!/usr/bin/env python3
"""Probe local GitHub tooling and, optionally, a target repository's
capability quadrant; print one JSON report.

Usage:
    python3 scripts/check_tooling.py [--repo OWNER/REPO] [--hostname HOST]
                                     [--skip-network]

Local probes (all read-only; nothing is installed or changed):
- gh:        whether the GitHub CLI is on PATH, its version, and whether
             `gh auth status` exits 0 for the target host.
- token_env: which of GH_TOKEN, GITHUB_TOKEN, GITHUB_PERSONAL_ACCESS_TOKEN,
             and GITHUB_PAT are set. Only names and set/unset booleans are
             reported; token values are never printed.
- network:   HTTPS reachability of the target API host (any HTTP response,
             including 401, counts as reachable). Disabled by --skip-network.

Target probes (--repo, read-only, via gh; each field degrades to "unknown"
with a `verify` hint naming the exact command or UI path a human must check):
- repository: owner type (User/Organization), visibility, default branch,
              allowed merge methods, whether issues/discussions/wiki are on.
- actions:    whether Actions is enabled, the allowed-actions policy, the
              default GITHUB_TOKEN workflow permissions.
- rules:      ruleset count and whether the default branch carries a legacy
              branch protection rule (404 means none visible to this token,
              not proof of absence).
- org:        for Organization owners, whether org issue types respond.
- plan:       never probed — plan/tier is not reliably readable by API.
              Reported as unknown with the manual check path.
- docs:       the docs.github.com `version=` hint derived from the host.

This script cannot see MCP session state. Whether a GitHub MCP server is
connected can only be answered by inspecting the agent's own tool list.

Output: one JSON object on stdout; diagnostics on stderr. Idempotent —
safe to re-run any number of times.

Exit codes: 0 = all probes ran (missing tooling, permissions, or features
are data, not a failure), 1 = a probe crashed unexpectedly, 2 = bad
arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

TOKEN_ENV_VARS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GITHUB_PERSONAL_ACCESS_TOKEN",
    "GITHUB_PAT",
)

REPO_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
SUBPROCESS_TIMEOUT = 30
NETWORK_TIMEOUT = 5


def run_command(argv: list[str], env: dict | None = None) -> tuple[int, str]:
    """Run a command; return (exit code, full stdout). (-1, "") if unrunnable."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"warning: `{' '.join(argv)}` failed to run: {exc}", file=sys.stderr)
        return -1, ""
    return proc.returncode, proc.stdout.strip()


def gh_env(hostname: str | None) -> dict:
    env = dict(os.environ)
    if hostname:
        env["GH_HOST"] = hostname
    return env


def gh_json(path: str, hostname: str | None):
    """GET a REST path through gh api; return parsed JSON or None."""
    code, out = run_command(["gh", "api", path], env=gh_env(hostname))
    if code != 0 or not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def probe_gh(hostname: str | None) -> dict:
    result: dict = {"installed": False, "version": None, "authenticated": False}
    if shutil.which("gh") is None:
        return result
    result["installed"] = True
    _, version = run_command(["gh", "--version"])
    result["version"] = version.splitlines()[0] if version else None
    argv = ["gh", "auth", "status"]
    if hostname:
        argv += ["--hostname", hostname]
    exit_code, _ = run_command(argv)
    result["authenticated"] = exit_code == 0
    return result


def probe_token_env() -> dict:
    # Booleans only: never read token values into the report.
    return {name: bool(os.environ.get(name)) for name in TOKEN_ENV_VARS}


def probe_network(skip: bool, hostname: str | None) -> dict:
    host = hostname or "github.com"
    api = "https://api.github.com" if host == "github.com" else f"https://{host}/api/v3"
    result: dict = {"skipped": skip, "api_endpoint": api, "reachable": None}
    if skip:
        return result
    request = urllib.request.Request(api, headers={"User-Agent": "check-tooling-probe"})
    try:
        with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT):
            result["reachable"] = True
    except urllib.error.HTTPError:
        result["reachable"] = True  # an HTTP status is still a response
    except (urllib.error.URLError, OSError) as exc:
        print(f"warning: {api} unreachable: {exc}", file=sys.stderr)
        result["reachable"] = False
    return result


def unknown(verify: str) -> dict:
    return {"value": "unknown", "verify": verify}


def probe_repository(repo: str, hostname: str | None) -> dict:
    data = gh_json(f"repos/{repo}", hostname)
    if data is None:
        return {
            "skipped": (
                "could not read the repository — check `gh auth status`, the "
                f"OWNER/REPO spelling, and access, then re-run: gh api repos/{repo}"
            )
        }
    return {
        "owner_type": data.get("owner", {}).get("type"),
        "visibility": data.get("visibility"),
        "default_branch": data.get("default_branch"),
        "is_fork": data.get("fork"),
        "merge_methods": {
            "merge_commit": data.get("allow_merge_commit"),
            "squash": data.get("allow_squash_merge"),
            "rebase": data.get("allow_rebase_merge"),
            "auto_merge": data.get("allow_auto_merge"),
        },
        "features": {
            "issues": data.get("has_issues"),
            "discussions": data.get("has_discussions"),
            "wiki": data.get("has_wiki"),
        },
    }


def probe_actions(repo: str, hostname: str | None) -> dict:
    perms = gh_json(f"repos/{repo}/actions/permissions", hostname)
    if perms is None:
        return {
            "enabled": unknown(
                f"gh api repos/{repo}/actions/permissions (needs admin scope), "
                "or Settings > Actions > General in the UI"
            )
        }
    result: dict = {
        "enabled": perms.get("enabled"),
        "allowed_actions": perms.get("allowed_actions"),
    }
    wf = gh_json(f"repos/{repo}/actions/permissions/workflow", hostname)
    if wf is None:
        result["default_workflow_permissions"] = unknown(
            f"gh api repos/{repo}/actions/permissions/workflow, or "
            "Settings > Actions > General > Workflow permissions"
        )
    else:
        result["default_workflow_permissions"] = wf.get("default_workflow_permissions")
        result["can_approve_pull_request_reviews"] = wf.get(
            "can_approve_pull_request_reviews"
        )
    return result


def probe_rules(repo: str, default_branch: str | None, hostname: str | None) -> dict:
    result: dict = {}
    rulesets = gh_json(f"repos/{repo}/rulesets", hostname)
    if rulesets is None:
        result["ruleset_count"] = unknown(
            f"gh api repos/{repo}/rulesets — a 403/404 here can mean plan, "
            "permissions, or none; it is not proof of absence"
        )
    else:
        result["ruleset_count"] = len(rulesets)
    if default_branch:
        protection = gh_json(
            f"repos/{repo}/branches/{default_branch}/protection", hostname
        )
        if protection is None:
            result["legacy_branch_protection"] = {
                "value": "absent-or-unreadable",
                "note": (
                    "404 means no rule visible to this token — plan, "
                    "permissions, or genuinely none. Verify: gh api "
                    f"repos/{repo}/branches/{default_branch}/protection"
                ),
            }
        else:
            result["legacy_branch_protection"] = "present"
    return result


def probe_org(owner: str, owner_type: str | None, hostname: str | None) -> dict:
    if owner_type != "Organization":
        return {"applicable": False}
    issue_types = gh_json(f"orgs/{owner}/issue-types", hostname)
    if issue_types is None:
        return {
            "applicable": True,
            "issue_types": unknown(
                f"gh api orgs/{owner}/issue-types — absence can be plan, "
                "permissions, or the feature being off"
            ),
        }
    count = len(issue_types.get("issue_types", issue_types)) if isinstance(
        issue_types, (list, dict)
    ) else None
    return {"applicable": True, "issue_types": {"responding": True, "count": count}}


def docs_hint(hostname: str | None) -> dict:
    if hostname and hostname != "github.com":
        return {
            "host": hostname,
            "version_param": (
                "enterprise-server@<version> — read the GHES version from "
                f"https://{hostname}/api/v3/meta (installed_version)"
            ),
        }
    return {
        "host": "github.com",
        "version_param": (
            "free-pro-team@latest; use enterprise-cloud@latest only after the "
            "plan is confirmed Enterprise"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe gh, token env vars, network, and (with --repo) the target "
            "repository's capability quadrant; print one JSON object. Exit "
            "codes: 0 = probes ran (missing tooling/permissions/features are "
            "data, not failure), 1 = a probe crashed unexpectedly, 2 = bad "
            "arguments."
        ),
        epilog=(
            "Example: python3 scripts/check_tooling.py --repo octo/widget"
        ),
    )
    parser.add_argument(
        "--repo",
        help="target repository as OWNER/REPO; enables the capability probes",
    )
    parser.add_argument(
        "--hostname",
        help="GitHub Enterprise Server host (default: github.com)",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="skip the HTTPS reachability probe (for offline environments)",
    )
    args = parser.parse_args()
    if args.repo and not REPO_RE.match(args.repo):
        parser.error("--repo must be OWNER/REPO (no spaces, exactly one '/')")

    try:
        gh = probe_gh(args.hostname)
        report: dict = {
            "gh": gh,
            "token_env": probe_token_env(),
            "network": probe_network(args.skip_network, args.hostname),
            "docs": docs_hint(args.hostname),
            "plan": unknown(
                "plan/tier is not reliably readable by API — check the owner's "
                "billing/settings page, or infer from which gated features "
                "respond"
            ),
        }
        if args.repo:
            if not (gh["installed"] and gh["authenticated"]):
                report["repository"] = {
                    "skipped": (
                        "gh is missing or unauthenticated — install gh, run "
                        "`gh auth login`, then re-run with --repo"
                    )
                }
            else:
                repository = probe_repository(args.repo, args.hostname)
                report["repository"] = repository
                if "skipped" not in repository:
                    owner = args.repo.split("/", 1)[0]
                    report["actions"] = probe_actions(args.repo, args.hostname)
                    report["rules"] = probe_rules(
                        args.repo, repository.get("default_branch"), args.hostname
                    )
                    report["org"] = probe_org(
                        owner, repository.get("owner_type"), args.hostname
                    )
    except Exception as exc:  # any probe crash is exit 1, per the contract
        print(f"error: a probe crashed unexpectedly: {exc}", file=sys.stderr)
        return 1

    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
