#!/usr/bin/env python3
"""Sync a GitLab project's own labels to a JSON taxonomy file through the REST API.

Reads the desired labels from a JSON array of {"name", "color",
"description"} objects, compares them with the labels defined on the
project itself, and prints the resulting plan as JSON to stdout. Dry-run by
default: nothing changes without --apply, and nothing is deleted without
--prune. Idempotent: re-running after a successful apply yields an
all-skip plan. Labels inherited from ancestor groups are reported, never
edited: the group's configuration repository owns them.

The token comes from the GITLAB_TOKEN environment variable only and is
never printed: in CI a project access token stored as a masked, protected
variable; by hand, the maintainer's glab login passed for one run.

Usage:
    python3 scripts/sync_labels.py --file .gitlab/labels.json
    python3 scripts/sync_labels.py --file .gitlab/labels.json --apply [--prune]
    (--api-url and --project default to CI_API_V4_URL and CI_PROJECT_ID.)

Exit codes: 0 plan printed or applied cleanly; 1 an invalid labels file, a
missing token, or an API failure; 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import NoReturn

HEX_COLOR = re.compile(r"^[0-9a-f]{6}$")
PER_PAGE = 100


def die(message: str) -> NoReturn:
    sys.exit(f"sync_labels: error: {message}")


def load_desired(path: str) -> list[dict]:
    """The labels file as normalized label objects; anything else in it ends the script naming the file."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        die(f"cannot read {path} as JSON: {exc}")
    if not isinstance(data, list) or not data:
        die(f"{path} must be a non-empty JSON array of label objects.")
    labels, seen = [], set()
    for index, item in enumerate(data):
        where = f"{path}[{index}]"
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            die(f"{where} must be an object with a non-empty string 'name'.")
        name = item["name"].strip()
        color = item.get("color")
        if not isinstance(color, str) or not HEX_COLOR.match(color.strip().lstrip("#").lower()):
            die(f'{where} ({name!r}): color {color!r} is not a 6-digit hex string (e.g. "#d73a4a").')
        description = item.get("description", "")
        if not isinstance(description, str):
            die(f"{where} ({name!r}): description {description!r} is not a string.")
        if name.lower() in seen:
            die(f"{where}: duplicate label name {name!r}.")
        seen.add(name.lower())
        labels.append(
            {"name": name, "color": "#" + color.strip().lstrip("#").lower(), "description": description.strip()}
        )
    return labels


class Api:
    def __init__(self, base: str, project: str, access_token: str) -> None:
        self.base = base.rstrip("/")
        self.project = urllib.parse.quote(project, safe="")
        self.access_token = access_token

    def call(self, method: str, path: str, params: dict | None = None) -> tuple[object, dict]:
        url = f"{self.base}/projects/{self.project}/{path}"
        data = None
        if params and method == "GET":
            url += "?" + urllib.parse.urlencode(params)
        elif params:
            data = urllib.parse.urlencode(params).encode()
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.access_token}")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                headers = dict(response.headers)
        except urllib.error.HTTPError as exc:
            die(f"{method} {path} returned HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}")
        except urllib.error.URLError as exc:
            die(f"{method} {path} could not reach {self.base}: {exc.reason}")
        if not body:
            return None, headers
        try:
            return json.loads(body), headers
        except ValueError:
            die(f"{method} {path} did not return JSON: {body[:200]!r}")


def fetch_current(api: Api) -> tuple[dict[str, dict], list[str]]:
    """The project's own labels by lower-cased name, and the names of inherited group labels."""
    own, inherited, page = {}, [], "1"
    while page:
        entries, headers = api.call("GET", "labels", {"per_page": PER_PAGE, "page": page, "with_counts": "false"})
        if not isinstance(entries, list):
            die(f"GET labels returned {json.dumps(entries)[:200]}, not a JSON list of labels.")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                die(f"GET labels returned an entry without a name: {json.dumps(entry)[:200]}")
            if entry.get("is_project_label"):
                own[entry["name"].lower()] = {
                    "name": entry["name"],
                    "color": str(entry.get("color", "")).lower(),
                    "description": (entry.get("description") or "").strip(),
                }
            else:
                inherited.append(entry["name"])
        page = headers.get("X-Next-Page") or headers.get("x-next-page") or ""
    return own, inherited


def build_plan(desired: list[dict], own: dict[str, dict], inherited: list[str]) -> dict:
    inherited_keys = {name.lower() for name in inherited}
    create, update, skip, group_owned = [], [], [], []
    for label in desired:
        key = label["name"].lower()
        existing = own.get(key)
        if existing is None and key in inherited_keys:
            group_owned.append(label["name"])
        elif existing is None:
            create.append(label)
        elif existing["color"] != label["color"] or existing["description"] != label["description"]:
            update.append(label)
        else:
            skip.append(label["name"])
    desired_keys = {label["name"].lower() for label in desired}
    prune_candidates = [entry["name"] for key, entry in sorted(own.items()) if key not in desired_keys]
    return {
        "create": create,
        "update": update,
        "skip": skip,
        "group_owned": group_owned,
        "prune_candidates": prune_candidates,
    }


def apply_plan(api: Api, plan: dict, do_prune: bool) -> list[str]:
    for label in plan["create"]:
        print(f"create: {label['name']}", file=sys.stderr)
        api.call("POST", "labels", label)
    for label in plan["update"]:
        print(f"update: {label['name']}", file=sys.stderr)
        api.call(
            "PUT",
            f"labels/{urllib.parse.quote(label['name'], safe='')}",
            {"color": label["color"], "description": label["description"]},
        )
    pruned = []
    if do_prune:
        for name in plan["prune_candidates"]:
            print(f"delete: {name}", file=sys.stderr)
            api.call("DELETE", f"labels/{urllib.parse.quote(name, safe='')}")
            pruned.append(name)
    return pruned


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sync a GitLab project's own labels to a JSON taxonomy file. Dry-run by default: "
            "prints the plan as JSON without changing anything; --apply executes it."
        ),
        epilog="Exit codes: 0 plan printed or applied cleanly, 1 invalid file, missing token, or API failure, "
        "2 bad arguments. The token is read from GITLAB_TOKEN.",
    )
    parser.add_argument("--file", required=True, help='labels JSON: an array of {"name", "color", "description"}')
    parser.add_argument(
        "--api-url", default=os.environ.get("CI_API_V4_URL"), help="REST base, e.g. https://HOST/api/v4"
    )
    parser.add_argument("--project", default=os.environ.get("CI_PROJECT_ID"), help="project ID or full path")
    parser.add_argument("--apply", action="store_true", help="execute the plan (default: dry-run)")
    parser.add_argument("--prune", action="store_true", help="with --apply, delete project labels absent from the file")
    args = parser.parse_args()
    if not args.api_url or not args.project:
        parser.error("--api-url and --project are required outside a GitLab CI job")
    if args.prune and not args.apply:
        parser.error("--prune requires --apply; the dry-run plan already reports prune candidates")
    access_token = os.environ.get("GITLAB_TOKEN", "")
    if not access_token:
        die("GITLAB_TOKEN is not set; in CI use the project access token variable, by hand pass your glab login.")

    desired = load_desired(args.file)
    api = Api(args.api_url, args.project, access_token)
    own, inherited = fetch_current(api)
    plan = build_plan(desired, own, inherited)
    pruned = apply_plan(api, plan, args.prune) if args.apply else []
    json.dump({"project": args.project, "applied": args.apply, **plan, "pruned": pruned}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
