#!/usr/bin/env python3
"""Plan or apply managed issue labels from the issue forms' answers.

The forms ask for Priority and Catalog; this script turns those answers into
`priority/<value>` and `catalog/<value>` labels. The mapping is derived from
.github/labels.json by prefix — there is no second list to keep in sync.
Only managed labels are ever added or removed: the two prefixed families and
`status/needs-triage`, which is removed once the issue carries a type label
(bug, enhancement, task) and exactly one priority label.

Usage:
    python3 scripts/sync_issue_metadata.py --event "$GITHUB_EVENT_PATH" [--apply]
    python3 scripts/sync_issue_metadata.py --issue 42 --repo OWNER/REPO [--apply]

Without --apply the plan is printed as JSON and nothing changes. With
--apply the plan is executed through `gh api` (GH_TOKEN or an authenticated
gh). Exit codes: 0 = plan computed (and applied), 1 = a form answer did not
map to a label (the rest of the plan still prints, and applies with
--apply), 2 = bad arguments, unreadable input, or an API failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LABELS_FILE = Path(".github/labels.json")
MANAGED_PREFIXES = ("priority/", "catalog/")
SECTION_FOR_PREFIX = {"priority/": "Priority", "catalog/": "Catalog"}
TYPE_LABELS = ("bug", "enhancement", "task")
NEEDS_TRIAGE = "status/needs-triage"


def fail(code: int, message: str) -> None:
    print(f"sync_issue_metadata: error: {message}", file=sys.stderr)
    sys.exit(code)


def load_label_names(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(2, f"cannot read {path}: {exc}")
    if not isinstance(data, list):
        fail(2, f"{path} must be a JSON array of label objects")
    return {str(item.get("name", "")).strip() for item in data if isinstance(item, dict)}


def section_values(body: str, name: str) -> list[str] | None:
    """Return the answers under `### <name>`, [] for no answer, None if absent."""
    match = re.search(
        rf"^### {re.escape(name)}\s*$\n(.*?)(?=^### \S|\Z)",
        body or "",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not match:
        return None
    raw = match.group(1).strip()
    if not raw or raw == "_No response_":
        return []
    values = []
    for piece in re.split(r"[\n,]", raw):
        value = piece.strip().lstrip("-*").strip().strip("`").lower()
        if value:
            values.append(value)
    return values


def load_issue_from_event(path: str) -> tuple[dict, str]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(2, f"cannot read event payload {path}: {exc}")
    issue = payload.get("issue")
    if not isinstance(issue, dict):
        fail(2, f"{path} is not an issues event payload")
    repo = (payload.get("repository") or {}).get("full_name")
    if not repo:
        fail(2, f"{path} carries no repository.full_name")
    return issue, repo


def load_issue_via_gh(number: int, repo: str) -> dict:
    try:
        out = subprocess.run(
            ["gh", "api", f"repos/{repo}/issues/{number}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(2, f"gh api failed: {exc}")
    return json.loads(out)


def plan(issue: dict, known: set[str]) -> tuple[dict, list[str]]:
    findings: list[str] = []
    current = {label["name"] for label in issue.get("labels") or [] if isinstance(label, dict)}
    body = issue.get("body") or ""
    add: set[str] = set()
    remove: set[str] = set()
    for prefix in MANAGED_PREFIXES:
        values = section_values(body, SECTION_FOR_PREFIX[prefix])
        if values is None:
            continue  # the form did not ask; leave this axis alone
        desired = set()
        for value in values:
            label = prefix + value
            if label in known:
                desired.add(label)
            else:
                findings.append(
                    f"'{value}' under '### {SECTION_FOR_PREFIX[prefix]}' has no label {label!r} in {LABELS_FILE}. "
                    f"Fix: add the label to {LABELS_FILE} and the form, or correct the issue."
                )
        managed_now = {label for label in current if label.startswith(prefix)}
        add |= desired - managed_now
        remove |= managed_now - desired
    after = (current | add) - remove
    has_type = any(label in after for label in TYPE_LABELS)
    priorities = [label for label in after if label.startswith("priority/")]
    if has_type and len(priorities) == 1 and NEEDS_TRIAGE in after:
        remove.add(NEEDS_TRIAGE)
    return {"number": issue.get("number"), "add": sorted(add), "remove": sorted(remove)}, findings


def apply(repo: str, number: int, add: list[str], remove: list[str]) -> None:
    base = f"repos/{repo}/issues/{number}/labels"
    if add:
        run_gh(["api", base, "--method", "POST", "--input", "-"], json.dumps({"labels": add}))
    for label in remove:
        run_gh(["api", f"{base}/{label}", "--method", "DELETE"])


def run_gh(args: list[str], stdin: str | None = None) -> None:
    try:
        subprocess.run(["gh", *args], check=True, capture_output=True, text=True, input=stdin)
    except subprocess.CalledProcessError as exc:
        if "404" in (exc.stderr or "") and "DELETE" in args:
            return  # already removed by a concurrent run
        fail(2, f"gh {' '.join(args)} failed: {exc.stderr.strip()}")
    except OSError as exc:
        fail(2, f"gh is not available: {exc}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event", help="path to the issues event payload (GITHUB_EVENT_PATH)")
    source.add_argument("--issue", type=int, help="issue number to fetch with gh")
    parser.add_argument("--repo", help="OWNER/REPO (required with --issue)")
    parser.add_argument("--labels", default=str(LABELS_FILE), help=f"label taxonomy (default {LABELS_FILE})")
    parser.add_argument("--apply", action="store_true", help="execute the plan through gh api")
    args = parser.parse_args(argv)

    if args.event:
        issue, repo = load_issue_from_event(args.event)
    else:
        if not args.repo:
            parser.error("--repo is required with --issue")
        issue, repo = load_issue_via_gh(args.issue, args.repo), args.repo

    known = load_label_names(Path(args.labels))
    result, findings = plan(issue, known)
    result["repo"] = repo
    result["applied"] = False
    if args.apply and (result["add"] or result["remove"]):
        apply(repo, int(result["number"]), result["add"], result["remove"])
        result["applied"] = True
    print(json.dumps(result, indent=2))
    for finding in findings:
        print(finding, file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
