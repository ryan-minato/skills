#!/usr/bin/env python3
"""Sync a GitHub repository's labels to a JSON taxonomy file through the gh CLI.

Reads the desired labels from a JSON array of {"name", "color",
"description"} objects, compares them with the labels currently in the
repository, and prints the resulting plan as JSON to stdout. Dry-run by
default: nothing changes without --apply. Idempotent: re-running after a
successful apply yields an all-skip plan.

Usage:
    python3 scripts/sync_labels.py --file .github/labels.json --repo OWNER/REPO
    python3 scripts/sync_labels.py --file .github/labels.json --repo OWNER/REPO --apply [--prune]

`gh` is checked where its output enters: a missing gh, a non-zero exit, or
output that is not a JSON list of labels ends the script naming the
command, before any plan is printed.

Exit codes: 0 plan printed or applied cleanly; 1 an invalid labels file or
a gh failure; 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

HEX_COLOR = re.compile(r"^[0-9a-f]{6}$")
REPO_SLUG = re.compile(r"^[^/\s]+/[^/\s]+$")
# gh paginates label listings; a truncated listing would classify existing
# labels as "create" and make the apply fail halfway through.
LABEL_PAGE_LIMIT = 1000


def gh(*args: str) -> str:
    """Run one gh command and return its stdout; a missing gh or a non-zero exit ends the script."""
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("sync_labels: error: `gh` is not installed; install the GitHub CLI and run `gh auth login`.")
    if result.returncode != 0:
        sys.exit(f"sync_labels: error: `gh {' '.join(args)}` exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def load_desired(path: str) -> list[dict]:
    """The labels file as normalized label objects; anything else in it ends the script naming the file."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        sys.exit(f"sync_labels: error: cannot read {path} as JSON: {exc}")
    if not isinstance(data, list) or not data:
        sys.exit(f"sync_labels: error: {path} must be a non-empty JSON array of label objects.")
    labels = []
    seen = set()
    for index, item in enumerate(data):
        where = f"{path}[{index}]"
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            sys.exit(f"sync_labels: error: {where} must be an object with a non-empty string 'name'.")
        name = item["name"].strip()
        color = item.get("color")
        if not isinstance(color, str) or not HEX_COLOR.match(color.strip().lstrip("#").lower()):
            sys.exit(
                f"sync_labels: error: {where} ({name!r}): color {color!r} is not a 6-digit hex string "
                '(use e.g. "d73a4a", without "#").'
            )
        description = item.get("description", "")
        if not isinstance(description, str):
            sys.exit(f"sync_labels: error: {where} ({name!r}): description {description!r} is not a string.")
        if name.lower() in seen:
            sys.exit(f"sync_labels: error: {where}: duplicate label name {name!r}.")
        seen.add(name.lower())
        labels.append({"name": name, "color": color.strip().lstrip("#").lower(), "description": description.strip()})
    return labels


def fetch_current(repo: str) -> dict[str, dict]:
    """The repository's labels by lower-cased name, read through `gh label list`."""
    command = "gh label list"
    out = gh("label", "list", "-R", repo, "--json", "name,color,description", "--limit", str(LABEL_PAGE_LIMIT))
    try:
        entries = json.loads(out)
    except ValueError as exc:
        sys.exit(f"sync_labels: error: `{command}` did not print JSON ({exc}): {out[:200]!r}")
    if not isinstance(entries, list):
        sys.exit(f"sync_labels: error: `{command}` printed {out[:200]!r}, not a JSON list of labels.")
    if len(entries) >= LABEL_PAGE_LIMIT:
        sys.exit(
            f"sync_labels: error: {repo} returned {len(entries)} labels, at or above the {LABEL_PAGE_LIMIT} "
            "listing limit — the listing may be truncated, which would turn existing labels into creates. "
            "Raise LABEL_PAGE_LIMIT and re-run."
        )
    return {
        entry["name"].lower(): {
            "name": entry["name"],
            "color": entry["color"].lower(),
            "description": entry["description"].strip(),
        }
        for entry in entries
    }


def build_plan(desired: list[dict], current: dict[str, dict]) -> tuple[list, list, list, list]:
    create, update, skip = [], [], []
    for label in desired:
        existing = current.get(label["name"].lower())
        if existing is None:
            create.append(label)
        elif existing["color"] != label["color"] or existing["description"] != label["description"]:
            update.append(label)
        else:
            skip.append(label["name"])
    desired_keys = {label["name"].lower() for label in desired}
    prune_candidates = [entry["name"] for key, entry in sorted(current.items()) if key not in desired_keys]
    return create, update, skip, prune_candidates


def apply_plan(repo: str, create: list, update: list, prune_candidates: list, do_prune: bool) -> list[str]:
    for label in create:
        print(f"create: {label['name']}", file=sys.stderr)
        gh(
            "label",
            "create",
            label["name"],
            "-R",
            repo,
            "--color",
            label["color"],
            "--description",
            label["description"],
        )
    for label in update:
        print(f"update: {label['name']}", file=sys.stderr)
        gh("label", "edit", label["name"], "-R", repo, "--color", label["color"], "--description", label["description"])
    pruned = []
    if do_prune:
        for name in prune_candidates:
            print(f"delete: {name}", file=sys.stderr)
            gh("label", "delete", name, "-R", repo, "--yes")
            pruned.append(name)
    return pruned


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sync a GitHub repository's labels to a JSON taxonomy file. "
            "Dry-run by default: prints the create/update/skip/prune plan "
            "as JSON to stdout without changing anything; re-run with "
            "--apply to execute it."
        ),
        epilog="Exit codes: 0 plan printed or applied cleanly, 1 invalid labels file or gh failure, 2 bad arguments.",
    )
    parser.add_argument(
        "--file",
        required=True,
        help='path to the labels JSON file: an array of {"name", "color", "description"} objects',
    )
    parser.add_argument("--repo", required=True, help="target repository as OWNER/REPO")
    parser.add_argument("--apply", action="store_true", help="execute the plan (default: dry-run, print the plan only)")
    parser.add_argument("--prune", action="store_true", help="with --apply, delete repo labels absent from the file")
    args = parser.parse_args()
    if not REPO_SLUG.match(args.repo):
        parser.error(f"--repo must be OWNER/REPO, got {args.repo!r}")
    if args.prune and not args.apply:
        parser.error("--prune requires --apply; the dry-run plan already reports prune candidates")

    desired = load_desired(args.file)
    current = fetch_current(args.repo)
    create, update, skip, prune_candidates = build_plan(desired, current)
    pruned = apply_plan(args.repo, create, update, prune_candidates, args.prune) if args.apply else []
    plan = {
        "repo": args.repo,
        "applied": args.apply,
        "create": create,
        "update": update,
        "skip": skip,
        "prune_candidates": prune_candidates,
        "pruned": pruned,
    }
    json.dump(plan, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
