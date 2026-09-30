#!/usr/bin/env python3
"""Resolve GitHub Projects v2 node IDs for gh project item-edit.

gh project item-edit takes GraphQL node IDs (--project-id, --field-id,
--single-select-option-id, --id), which the human-facing commands do not
surface directly. This script resolves them from the project number and
human-readable names via gh's JSON output.

Usage, from the repository root:
    python3 <project skill>/scripts/project_fields.py --owner OWNER --number N \\
        [--field "Status" [--option "In Progress"]] [--item-url URL]

Output: one JSON object on stdout —
    {"project_id": ..., "field_id": ..., "option_id": ..., "item_id": ...}
with null for anything not requested. Name matching is case-insensitive.

Every gh call is checked where its output enters: a missing gh, a non-zero
exit, output that is not JSON, or JSON without the list the script reads
ends the script naming the command.

Exit codes: 0 = resolved; 1 = a name or URL was not found (candidates are
listed on stderr) or gh failed; 2 = bad arguments.
Requires an authenticated gh CLI with the project token scope.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def gh_json(*args: str, key: str | None = None) -> dict:
    """Run one gh command and return its JSON object, which must carry a ``key`` list when one is named."""
    command = f"gh {' '.join(args[:2])}"
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("project_fields: error: `gh` is not installed; install the GitHub CLI and run `gh auth login`.")
    if result.returncode != 0:
        sys.exit(f"project_fields: error: `{command}` exited {result.returncode}: {result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
    except ValueError as exc:
        sys.exit(f"project_fields: error: `{command}` did not print JSON ({exc}): {result.stdout[:200]!r}")
    if not isinstance(data, dict) or (key is not None and not isinstance(data.get(key), list)):
        expected = f"an object with a `{key}` list" if key else "an object"
        sys.exit(f"project_fields: error: `{command}` printed {result.stdout[:200]!r}, not {expected}.")
    return data


def pick(kind: str, wanted: str, entries: list[dict]) -> dict:
    """The entry whose name matches case-insensitively; otherwise exit listing the candidates."""
    for entry in entries:
        if entry["name"].lower() == wanted.lower():
            return entry
    listing = ", ".join(sorted(entry["name"] for entry in entries)) or "(none)"
    sys.exit(f"project_fields: {kind} {wanted!r} not found; available: {listing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="project_fields.py",
        description="Resolve Projects v2 node IDs (project, field, option, item) for gh project item-edit",
    )
    parser.add_argument("--owner", required=True, help="user login, org, or @me")
    parser.add_argument("--number", type=int, required=True, help="project number")
    parser.add_argument("--field", help="field name, e.g. Status")
    parser.add_argument("--option", help="single-select option name; needs --field")
    parser.add_argument("--item-url", help="issue/PR URL to resolve to an item id")
    parser.add_argument("--limit", type=int, default=500, help="items scanned for --item-url")
    args = parser.parse_args()
    if args.option and not args.field:
        parser.error("--option requires --field")

    number = str(args.number)
    result = {"project_id": None, "field_id": None, "option_id": None, "item_id": None}
    result["project_id"] = gh_json("project", "view", number, "--owner", args.owner, "--format", "json")["id"]

    if args.field:
        fields = gh_json("project", "field-list", number, "--owner", args.owner, "--format", "json", key="fields")
        field = pick("field", args.field, fields["fields"])
        result["field_id"] = field["id"]
        if args.option:
            # Only a single-select field has options.
            if "options" not in field:
                sys.exit(f"project_fields: field {field['name']!r} is a {field['type']}, not a single-select field.")
            result["option_id"] = pick("option", args.option, field["options"])["id"]

    if args.item_url:
        listing = gh_json(
            "project",
            "item-list",
            number,
            "--owner",
            args.owner,
            "--limit",
            str(args.limit),
            "--format",
            "json",
            key="items",
        )
        # gh prints `content` as null for an item type it does not map (such
        # as an item redacted from a repository the token cannot read), and a
        # draft issue's content has no URL; neither can match.
        item = next(
            (i for i in listing["items"] if i["content"] is not None and i["content"].get("url") == args.item_url),
            None,
        )
        if item is None:
            sys.exit(f"project_fields: item with url {args.item_url} not found in the first {args.limit} items")
        result["item_id"] = item["id"]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
