#!/usr/bin/env python3
"""Validate a pull request event against repository policy."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_HEADINGS = (
    "Summary",
    "Related issue",
    "Changes",
    "Validation",
    "Checklist",
)
REQUIRED_CHECKLIST_ITEMS = (
    "`just check` passes locally",
    "Task-specific tests are recorded and pass",
    "No secrets, credentials, or personal data are present",
    "Non-merge/revert commits are atomic and follow Conventional Commits",
    "Documentation and paired translations are updated where required",
    "Public Skill indexes and marketplace metadata are synchronized where required",
)
CONVENTIONAL_RE = re.compile(
    r"^(feat|fix|docs|refactor|chore|test|ci)"
    r"(\([a-z0-9][a-z0-9, -]*\))?!?: [a-z0-9].+"
)
CLOSING_RE = re.compile(
    r"\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\s+#\d+\b", re.IGNORECASE
)
NO_ISSUE_RE = re.compile(r"^N/A\s+[—-]\s+\S.+", re.IGNORECASE)


def section(body: str, name: str) -> str:
    pattern = re.compile(
        rf"^## {re.escape(name)}\s*$\n(.*?)(?=^## \S|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(body)
    if not match:
        return ""
    return re.sub(r"<!--[\s\S]*?-->", "", match.group(1)).strip()


def is_conventional(subject: str) -> bool:
    return bool(CONVENTIONAL_RE.fullmatch(subject))


def is_same_repository(pull_request: dict[str, Any]) -> bool:
    return (
        pull_request["head"]["repo"]["full_name"]
        == pull_request["base"]["repo"]["full_name"]
    )


def validate_pull_request(
    pull_request: dict[str, Any], commit_subjects: list[str] | None = None
) -> list[str]:
    body = pull_request.get("body") or ""
    title = pull_request.get("title") or ""
    errors: list[str] = []

    missing = [
        heading
        for heading in REQUIRED_HEADINGS
        if not re.search(rf"^## {re.escape(heading)}\s*$", body, re.MULTILINE)
    ]
    if missing:
        errors.append(
            ".github/PULL_REQUEST_TEMPLATE.md: PR body is missing sections: "
            + ", ".join(f"## {heading}" for heading in missing)
        )

    if not is_conventional(title):
        errors.append(
            "PR title must follow the repository Conventional Commit format; "
            "see AGENTS.md Core Conventions."
        )

    related = section(body, "Related issue")
    if not CLOSING_RE.search(related) and not NO_ISSUE_RE.match(related):
        errors.append(
            ".github/PULL_REQUEST_TEMPLATE.md: Related issue must contain "
            "`Closes #N` or `N/A — <reason>`."
        )

    if not pull_request.get("draft", False):
        validation = re.sub(
            r"^[-*]\s*$", "", section(body, "Validation"), flags=re.MULTILINE
        ).strip()
        if not validation:
            errors.append(
                ".github/PULL_REQUEST_TEMPLATE.md: ready PRs must record "
                "validation commands and results."
            )

        checklist_lines = [
            line.strip() for line in section(body, "Checklist").splitlines()
        ]
        incomplete = []
        for item in REQUIRED_CHECKLIST_ITEMS:
            line = next((line for line in checklist_lines if item in line), "")
            if not re.match(r"^- \[[xX]\]", line):
                incomplete.append(item)
        if incomplete:
            errors.append(
                ".github/PULL_REQUEST_TEMPLATE.md: ready PR checklist is "
                "incomplete: " + "; ".join(incomplete)
            )

    if is_same_repository(pull_request):
        if commit_subjects is None:
            errors.append(
                "Same-repository PR commit subjects were not supplied; fetch the "
                "full history and validate base..head."
            )
        else:
            invalid = [
                subject
                for subject in commit_subjects
                if not subject.startswith(("Merge ", "Revert "))
                and not is_conventional(subject)
            ]
            if invalid:
                errors.append(
                    "Same-repository PRs rebase into main, so every commit must "
                    "follow Conventional Commits unless it is a merge or revert. "
                    "Fix: " + "; ".join(repr(subject) for subject in invalid)
                )

    return errors


def git_commit_subjects(base_sha: str, head_sha: str) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--format=%s", f"{base_sha}..{head_sha}"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or "git log failed; checkout the PR with fetch-depth: 0."
        )
    return [line for line in result.stdout.splitlines() if line]


def load_event(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read pull request event {path}: {error}") from error
    if "pull_request" not in payload:
        raise ValueError(f"{path} does not contain a pull_request event payload")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=("Example: python3 scripts/check_pr_policy.py --event event.json"),
    )
    parser.add_argument("--event", type=Path, required=True, help="event JSON file")
    args = parser.parse_args(argv)

    try:
        payload = load_event(args.event)
        pull_request = payload["pull_request"]
        subjects = None
        if is_same_repository(pull_request):
            subjects = git_commit_subjects(
                pull_request["base"]["sha"], pull_request["head"]["sha"]
            )
        errors = validate_pull_request(pull_request, subjects)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"PR policy could not run: {error}", file=sys.stderr)
        return 1

    result = {"valid": not errors, "errors": errors}
    print(json.dumps(result, indent=2))
    if errors:
        for error in errors:
            print(f"::error::{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
