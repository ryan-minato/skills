#!/usr/bin/env python3
"""Sync an organization's issue types and issue fields to a JSON file.

Reads the desired taxonomy from a JSON object with "types" and "fields"
arrays, compares it with what the organization currently has, and prints
the resulting plan as JSON to stdout. Dry-run by default: nothing changes
without --apply. Idempotent: re-running after a successful apply yields an
all-skip plan.

Never deletes. Organization settings reach every repository in the
organization, so entries present in the organization but absent from the
file are reported as "unmanaged" and left untouched. Retire a type by
setting "is_enabled": false in the file instead.

Updating a single-select field replaces its whole option set, so this tool
carries each surviving option's existing id back into the request; editing
those options by hand through the API silently orphans their values.

Exit codes: 0 plan printed or applied cleanly, 1 gh missing or a gh
command failed, 2 bad arguments or an invalid taxonomy file.
"""

import argparse
import json
import subprocess
import sys

COLORS = {"gray", "blue", "green", "yellow", "orange", "red", "pink", "purple"}
DATA_TYPES = {"text", "number", "date", "single_select", "multi_select"}
VISIBILITIES = {"all", "organization_members_only"}
ORG_NAME_MAX = 39


def fail(code, message):
    print(f"sync_org_taxonomy: error: {message}", file=sys.stderr)
    sys.exit(code)


def run_gh(argv, hostname=None):
    """Run a gh api command; return stdout, or exit 1 with gh's error."""
    command = ["gh", "api"]
    if hostname:
        command += ["--hostname", hostname]
    command += argv
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        fail(1, "gh CLI not found. Install gh and run 'gh auth login'.")
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "403" in stderr or "Resource not accessible" in stderr:
            fail(
                1,
                "'gh api {}' was refused: {}. Reading organization issue "
                "types and fields needs the read:org scope. Run "
                "'gh auth refresh -s read:org'.".format(" ".join(argv), stderr),
            )
        fail(1, "'gh api {}' failed: {}".format(" ".join(argv), stderr))
    return proc.stdout


def api_json(argv, hostname=None):
    stdout = run_gh(argv, hostname)
    try:
        return json.loads(stdout) if stdout.strip() else []
    except json.JSONDecodeError:
        fail(1, "could not parse 'gh api {}' output as JSON".format(" ".join(argv)))


def as_list(payload, endpoint):
    """Tolerate both a bare array and an object wrapping the array."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("issue_types", "issue_fields", "items", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    fail(1, f"unexpected response shape from {endpoint}: expected a list")


def require_name(entry, context):
    name = str(entry.get("name", "")).strip()
    if not name:
        fail(2, f"{context}: every entry needs a non-empty \"name\"")
    return name


def check_unique(names, context):
    seen = set()
    for name in names:
        key = name.lower()
        if key in seen:
            fail(2, f"{context}: duplicate name {name!r} (names are case-insensitive)")
        seen.add(key)


def validate_types(raw):
    types = []
    for index, entry in enumerate(raw):
        context = f"types[{index}]"
        if not isinstance(entry, dict):
            fail(2, f"{context}: must be an object")
        name = require_name(entry, context)
        color = entry.get("color")
        if color is not None and str(color).lower() not in COLORS:
            fail(
                2,
                f"{context}: color {color!r} is not one of "
                f"{', '.join(sorted(COLORS))} (or null)",
            )
        types.append(
            {
                "name": name,
                "color": None if color is None else str(color).lower(),
                "description": str(entry.get("description") or "").strip(),
                "is_enabled": bool(entry.get("is_enabled", True)),
            }
        )
    check_unique([t["name"] for t in types], "types")
    return types


def validate_options(raw, context):
    if not isinstance(raw, list) or not raw:
        fail(2, f"{context}: a select field needs a non-empty \"options\" array")
    options = []
    for index, entry in enumerate(raw):
        item = f"{context}.options[{index}]"
        if not isinstance(entry, dict):
            fail(2, f"{item}: must be an object")
        name = require_name(entry, item)
        color = str(entry.get("color", "")).lower()
        if color not in COLORS:
            fail(
                2,
                f"{item}: color {entry.get('color')!r} is not one of "
                f"{', '.join(sorted(COLORS))}",
            )
        options.append(
            {
                "name": name,
                "color": color,
                "priority": int(entry.get("priority", index)),
                "description": str(entry.get("description") or "").strip(),
            }
        )
    check_unique([o["name"] for o in options], context + ".options")
    return options


def validate_fields(raw):
    fields = []
    for index, entry in enumerate(raw):
        context = f"fields[{index}]"
        if not isinstance(entry, dict):
            fail(2, f"{context}: must be an object")
        name = require_name(entry, context)
        data_type = str(entry.get("data_type", "")).lower()
        if data_type not in DATA_TYPES:
            fail(
                2,
                f"{context}: data_type {entry.get('data_type')!r} is not one "
                f"of {', '.join(sorted(DATA_TYPES))}",
            )
        visibility = str(entry.get("visibility", "all")).lower()
        if visibility not in VISIBILITIES:
            fail(
                2,
                f"{context}: visibility {entry.get('visibility')!r} is not one "
                f"of {', '.join(sorted(VISIBILITIES))}",
            )
        field = {
            "name": name,
            "data_type": data_type,
            "description": str(entry.get("description") or "").strip(),
            "visibility": visibility,
        }
        if data_type in ("single_select", "multi_select"):
            field["options"] = validate_options(entry.get("options"), context)
        fields.append(field)
    check_unique([f["name"] for f in fields], "fields")
    return fields


def load_desired(path):
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        fail(2, f"cannot read {path}: {exc}")
    except json.JSONDecodeError as exc:
        fail(2, f"{path} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        fail(
            2,
            f"{path} must be a JSON object with \"types\" and \"fields\" arrays",
        )
    types_raw = data.get("types", [])
    fields_raw = data.get("fields", [])
    if not isinstance(types_raw, list) or not isinstance(fields_raw, list):
        fail(2, f"{path}: \"types\" and \"fields\" must each be an array")
    if not types_raw and not fields_raw:
        fail(2, f"{path}: nothing to sync — both \"types\" and \"fields\" are empty")
    return validate_types(types_raw), validate_fields(fields_raw)


def fetch_current(org, hostname):
    types = as_list(
        api_json([f"orgs/{org}/issue-types"], hostname), f"orgs/{org}/issue-types"
    )
    fields = as_list(
        api_json([f"orgs/{org}/issue-fields"], hostname), f"orgs/{org}/issue-fields"
    )
    by_type = {
        str(entry["name"]).lower(): {
            "id": entry.get("id"),
            "name": str(entry["name"]),
            "color": (entry.get("color") or None),
            "description": str(entry.get("description") or "").strip(),
            "is_enabled": bool(entry.get("is_enabled", True)),
        }
        for entry in types
        if entry.get("name")
    }
    by_field = {
        str(entry["name"]).lower(): {
            "id": entry.get("id"),
            "name": str(entry["name"]),
            "data_type": str(entry.get("data_type") or "").lower(),
            "description": str(entry.get("description") or "").strip(),
            "visibility": str(entry.get("visibility") or "all").lower(),
            "options": [
                {
                    "id": option.get("id"),
                    "name": str(option.get("name", "")),
                    "color": str(option.get("color") or "").lower(),
                    "priority": option.get("priority"),
                    "description": str(option.get("description") or "").strip(),
                }
                for option in (entry.get("options") or [])
            ],
        }
        for entry in fields
        if entry.get("name")
    }
    return by_type, by_field


def merge_option_ids(desired_options, existing_options):
    """Carry surviving options' ids so an update edits rather than recreates."""
    ids = {str(o["name"]).lower(): o.get("id") for o in existing_options}
    merged = []
    for option in desired_options:
        item = dict(option)
        existing_id = ids.get(option["name"].lower())
        if existing_id is not None:
            item["id"] = existing_id
        merged.append(item)
    return merged


def options_differ(desired_options, existing_options):
    def shape(options):
        return [
            (o["name"].lower(), o.get("color"), o.get("description", ""))
            for o in options
        ]

    return shape(desired_options) != shape(existing_options)


def build_plan(desired_types, desired_fields, current_types, current_fields):
    plan = {
        "types": {"create": [], "update": [], "skip": []},
        "fields": {"create": [], "update": [], "skip": []},
        "unmanaged": {"types": [], "fields": []},
    }

    for item in desired_types:
        existing = current_types.get(item["name"].lower())
        if existing is None:
            plan["types"]["create"].append(item)
        elif (
            existing["color"] != item["color"]
            or existing["description"] != item["description"]
            or existing["is_enabled"] != item["is_enabled"]
        ):
            plan["types"]["update"].append(dict(item, id=existing["id"]))
        else:
            plan["types"]["skip"].append(item["name"])

    for item in desired_fields:
        existing = current_fields.get(item["name"].lower())
        if existing is None:
            plan["fields"]["create"].append(item)
            continue
        if existing["data_type"] and existing["data_type"] != item["data_type"]:
            fail(
                2,
                f"field {item['name']!r} already exists with data_type "
                f"{existing['data_type']!r}; the API cannot change a field's "
                "data type. Rename the field in the file or delete it in the "
                "organization settings by hand.",
            )
        changed = (
            existing["description"] != item["description"]
            or existing["visibility"] != item["visibility"]
        )
        if "options" in item:
            changed = changed or options_differ(item["options"], existing["options"])
        if changed:
            update = dict(item, id=existing["id"])
            if "options" in item:
                update["options"] = merge_option_ids(
                    item["options"], existing["options"]
                )
            plan["fields"]["update"].append(update)
        else:
            plan["fields"]["skip"].append(item["name"])

    desired_type_names = {t["name"].lower() for t in desired_types}
    desired_field_names = {f["name"].lower() for f in desired_fields}
    plan["unmanaged"]["types"] = sorted(
        entry["name"] for key, entry in current_types.items()
        if key not in desired_type_names
    )
    plan["unmanaged"]["fields"] = sorted(
        entry["name"] for key, entry in current_fields.items()
        if key not in desired_field_names
    )
    return plan


def post_json(endpoint, payload, hostname, method=None):
    argv = [endpoint, "--input", "-"]
    if method:
        argv = ["--method", method] + argv
    command = ["gh", "api"]
    if hostname:
        command += ["--hostname", hostname]
    command += argv
    try:
        proc = subprocess.run(
            command,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        fail(1, "gh CLI not found. Install gh and run 'gh auth login'.")
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "403" in stderr or "Resource not accessible" in stderr:
            fail(
                1,
                f"'gh api {endpoint}' was refused: {stderr}. Writing "
                "organization issue types or fields requires organization "
                "ownership and the admin:org scope; a repository admin is "
                "not an organization admin.",
            )
        fail(1, f"'gh api {endpoint}' failed: {stderr}")
    return proc.stdout


def apply_plan(org, plan, hostname):
    for item in plan["types"]["create"]:
        print(f"creating issue type {item['name']}", file=sys.stderr)
        post_json(f"orgs/{org}/issue-types", item, hostname, method="POST")
    for item in plan["types"]["update"]:
        print(f"updating issue type {item['name']}", file=sys.stderr)
        payload = {k: v for k, v in item.items() if k != "id"}
        post_json(
            f"orgs/{org}/issue-types/{item['id']}", payload, hostname, method="PUT"
        )
    for item in plan["fields"]["create"]:
        print(f"creating issue field {item['name']}", file=sys.stderr)
        post_json(f"orgs/{org}/issue-fields", item, hostname, method="POST")
    for item in plan["fields"]["update"]:
        print(f"updating issue field {item['name']}", file=sys.stderr)
        payload = {
            k: v for k, v in item.items() if k not in ("id", "data_type")
        }
        post_json(
            f"orgs/{org}/issue-fields/{item['id']}", payload, hostname, method="PATCH"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sync an organization's issue types and issue fields to a JSON "
            "taxonomy file via the gh CLI. Dry-run unless --apply is given; "
            "never deletes anything."
        ),
        epilog=(
            "Example: sync_org_taxonomy.py --file org-taxonomy.json "
            "--org my-org\n"
            "Exit codes: 0 plan printed or applied cleanly, 1 gh missing or "
            "failed, 2 bad arguments or invalid taxonomy file."
        ),
    )
    parser.add_argument(
        "--file",
        required=True,
        help=(
            'path to the taxonomy JSON: an object with "types" and "fields" '
            "arrays"
        ),
    )
    parser.add_argument(
        "--org", required=True, help="target organization login"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="execute the plan (default: dry-run, print the plan only)",
    )
    parser.add_argument(
        "--hostname",
        help="GitHub Enterprise Server hostname (default: gh's own host)",
    )
    args = parser.parse_args()

    org = args.org.strip()
    if not org or "/" in org or len(org) > ORG_NAME_MAX:
        fail(2, f"--org must be a bare organization login, got {args.org!r}")

    desired_types, desired_fields = load_desired(args.file)
    current_types, current_fields = fetch_current(org, args.hostname)
    plan = build_plan(desired_types, desired_fields, current_types, current_fields)

    if args.apply:
        apply_plan(org, plan, args.hostname)

    plan["org"] = org
    plan["applied"] = args.apply
    json.dump(plan, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
