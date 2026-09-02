#!/usr/bin/env python3
"""Plan or apply managed issue labels from Issue Form sections."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PRIORITY_LABELS = {
    "critical": "priority/critical",
    "high": "priority/high",
    "medium": "priority/medium",
    "low": "priority/low",
}
CATALOG_LABELS = {
    "core": "catalog/core",
    "engineering": "catalog/engineering",
    "meta": "catalog/meta",
    "scaffold": "catalog/scaffold",
    "util": "catalog/util",
    "writing": "catalog/writing",
}
RETIRED_CATALOG_LABELS = {"catalog/ops"}


def section(body: str, name: str) -> list[str] | None:
    match = re.search(
        rf"^### {re.escape(name)}\s*$\n(.*?)(?=^### \S|\Z)",
        body,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not match:
        return None
    value = match.group(1).strip()
    if not value or value == "_No response_":
        return []
    return [
        re.sub(r"^[-*]\s*", "", item).strip().lower()
        for item in re.split(r"\r?\n|,", value)
        if re.sub(r"^[-*]\s*", "", item).strip()
    ]


def plan_labels(body: str, current: list[str]) -> dict[str, Any]:
    priorities = section(body, "Priority")
    catalogs = section(body, "Catalog")
    if priorities is None and catalogs is None:
        return {
            "changed": False,
            "add": [],
            "remove": [],
            "warning": "No managed Issue Form fields found; labels are unchanged.",
        }

    desired: set[str] = set()
    if priorities is not None:
        selected = [
            PRIORITY_LABELS[value] for value in priorities if value in PRIORITY_LABELS
        ]
        if len(selected) != 1:
            return {
                "changed": False,
                "add": [],
                "remove": [],
                "warning": "Priority is missing or invalid; managed labels are unchanged.",
            }
        desired.add(selected[0])

    if catalogs is not None:
        selected = [
            CATALOG_LABELS[value] for value in catalogs if value in CATALOG_LABELS
        ]
        if not selected:
            return {
                "changed": False,
                "add": [],
                "remove": [],
                "warning": "Catalog is missing or invalid; managed labels are unchanged.",
            }
        desired.update(selected)

    managed = set(PRIORITY_LABELS.values())
    managed.update(CATALOG_LABELS.values())
    managed.update(RETIRED_CATALOG_LABELS)

    current_set = set(current)
    remove = sorted(
        label
        for label in current
        if label in managed
        and label not in desired
        and (
            (label.startswith("priority/") and priorities is not None)
            or (label.startswith("catalog/") and catalogs is not None)
        )
    )
    add = sorted(desired - current_set)
    return {
        "changed": bool(add or remove),
        "add": add,
        "remove": remove,
        "warning": None,
    }


def api_request(
    method: str, endpoint: str, auth: str, data: dict[str, Any] | None = None
) -> Any:
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(
        f"https://api.github.com/{endpoint}",
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {auth}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
            return json.loads(content) if content else None
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"GitHub API {method} {endpoint} failed with {error.code}: {detail}"
        ) from error


def issue_state(
    event: dict[str, Any], repo: str, auth: str, apply: bool
) -> tuple[str, list[str]]:
    issue = event["issue"]
    if apply:
        issue = api_request("GET", f"repos/{repo}/issues/{issue['number']}", auth)
    labels = [
        label["name"] if isinstance(label, dict) else label
        for label in issue.get("labels", [])
    ]
    return issue.get("body") or "", labels


def apply_plan(repo: str, number: int, auth: str, plan: dict[str, Any]) -> None:
    for label in plan["remove"]:
        encoded = urllib.parse.quote(label, safe="")
        try:
            api_request(
                "DELETE", f"repos/{repo}/issues/{number}/labels/{encoded}", auth
            )
        except RuntimeError as error:
            if "failed with 404" not in str(error):
                raise
    if plan["add"]:
        api_request(
            "POST",
            f"repos/{repo}/issues/{number}/labels",
            auth,
            {"labels": plan["add"]},
        )


def load_event(path: Path) -> dict[str, Any]:
    try:
        event = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read issue event {path}: {error}") from error
    if "issue" not in event:
        raise ValueError(f"{path} does not contain an issue event payload")
    return event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: python3 scripts/sync_issue_metadata.py --event event.json; "
            "add --apply only in the approved workflow."
        ),
    )
    parser.add_argument("--event", type=Path, required=True, help="issue event JSON")
    parser.add_argument("--repo", help="OWNER/REPO; defaults to GITHUB_REPOSITORY")
    parser.add_argument(
        "--apply", action="store_true", help="apply the computed label changes"
    )
    args = parser.parse_args(argv)

    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
    auth = os.environ.get("GITHUB_TOKEN", "")
    if args.apply and (not repo or not auth):
        parser.error("--apply requires --repo/GITHUB_REPOSITORY and GITHUB_TOKEN")

    try:
        event = load_event(args.event)
        body, labels = issue_state(event, repo, auth, args.apply)
        plan = plan_labels(body, labels)
        result = {
            "repo": repo or None,
            "issue": event["issue"]["number"],
            "applied": args.apply,
            **plan,
        }
        if args.apply and plan["changed"]:
            apply_plan(repo, result["issue"], auth, plan)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"Issue metadata sync failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    if plan["warning"]:
        print(plan["warning"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
