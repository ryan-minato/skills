#!/usr/bin/env python3
"""Plan guard for the project's platform-settings tf.

Delivered as scripts/tf_plan_guard.py. It reads the JSON form of a saved
plan (`tofu show -json plan.out` or `terraform show -json plan.out`) and
lists, for the reviewer, every update to an attribute the configuration
does not declare (a reset to the provider's default).

Without state (the default), removing a declaration only stops managing
an object, so any delete or replace fails the guard. A create that
imports nothing also fails: every run starts from an empty state, so a
created object must get an import block before the next run, or the
next run would create it again. Pass --allow-create only for the first
creation of an object that does not exist yet.

With --allow-delete, for a configuration that keeps state, deletes and
replaces are listed under their own heading for the approver and pass,
and creates are ordinary, because the state records them.

Python 3.10+, standard library only.

Usage:
    tofu plan -out plan.out && tofu show -json plan.out > plan.json
    python3 scripts/tf_plan_guard.py plan.json
    tofu show -json plan.out | python3 scripts/tf_plan_guard.py - --allow-create
    tofu show -json plan.out | python3 scripts/tf_plan_guard.py - --allow-delete

Exit codes: 0 = nothing blocked (listings are informational);
1 = a blocked change is planned, or the input is not a readable plan
JSON; 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import NoReturn

INDEX_RE = re.compile(r"\[[^\]]*\]")


def fail(message: str) -> NoReturn:
    print(f"tf_plan_guard: error: {message}", file=sys.stderr)
    sys.exit(1)


def read_plan(source: str) -> dict:
    label = "standard input" if source == "-" else source
    try:
        text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"cannot read {label}: {exc.strerror}")
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as exc:
        fail(f"{label} is not JSON ({exc.msg} at line {exc.lineno}); pass the output of `show -json` on a saved plan")
    if (
        not isinstance(plan, dict)
        or "format_version" not in plan
        or not isinstance(plan.get("resource_changes", []), list)
    ):
        fail(f"{label} is not a plan JSON: expected an object with format_version and a resource_changes list")
    for rc in plan.get("resource_changes", []):
        change = rc.get("change") if isinstance(rc, dict) else None
        if (
            not isinstance(rc, dict)
            or not isinstance(rc.get("address"), str)
            or not isinstance(change, dict)
            or not isinstance(change.get("actions"), list)
        ):
            fail(f"{label}: a resource_changes entry lacks an address or change.actions: {json.dumps(rc)[:200]}")
    return plan


def declared_attributes(module: dict, prefix: str = "") -> dict[str, set[str]]:
    """Map each configured resource address (without index) to the attribute names it declares."""
    declared: dict[str, set[str]] = {}
    for resource in module.get("resources", []):
        if not isinstance(resource, dict) or not isinstance(resource.get("address"), str):
            fail(f"configuration resource without an address: {json.dumps(resource)[:200]}")
        declared[prefix + resource["address"]] = set(resource.get("expressions", {}))
    for name, call in module.get("module_calls", {}).items():
        declared.update(declared_attributes(call.get("module", {}), f"{prefix}module.{name}."))
    return declared


def changed_attributes(change: dict) -> set[str]:
    before = change.get("before") or {}
    after = change.get("after") or {}
    unknown = change.get("after_unknown") or {}
    keys = set(before) | set(after)
    return {key for key in keys if before.get(key) != after.get(key) and not unknown.get(key)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a tf plan JSON: without state, block deletes, replaces, and un-imported creates."
    )
    parser.add_argument("plan", help="path to the `show -json` output of a saved plan, or - for standard input")
    parser.add_argument(
        "--allow-create",
        action="store_true",
        help="without state: allow creates that import nothing, for the first creation of an object",
    )
    parser.add_argument(
        "--allow-delete",
        action="store_true",
        help="with state: list deletes and replaces for approval instead of failing; creates are ordinary",
    )
    args = parser.parse_args()

    plan = read_plan(args.plan)
    declared = declared_attributes(plan.get("configuration", {}).get("root_module", {}))

    deletes, creates, resets = [], [], []
    for rc in plan.get("resource_changes", []):
        address = rc["address"]
        change = rc["change"]
        actions = change["actions"]
        if "delete" in actions:
            kind = "replace" if "create" in actions else "delete"
            deletes.append(f"{address}: {kind} planned ({', '.join(actions)})")
        elif actions == ["create"] and not change.get("importing"):
            creates.append(f"{address}: created, not imported")
        elif "update" in actions:
            config_address = INDEX_RE.sub("", address)
            undeclared = sorted(changed_attributes(change) - declared.get(config_address, set()))
            if undeclared:
                resets.append(f"{address}: undeclared attributes change: {', '.join(undeclared)}")

    blocked_creates = creates and not (args.allow_create or args.allow_delete)
    blocked_deletes = deletes and not args.allow_delete
    sections = (
        ("Undeclared attributes that change (reset to the provider's value)", resets),
        ("Blocked: create without import" if blocked_creates else "Creates", creates),
        ("Blocked: delete or replace" if blocked_deletes else "Deletes and replaces to approve", deletes),
    )
    for title, lines in sections:
        if lines:
            print(f"{title}:")
            for line in lines:
                print(f"  {line}")
    if blocked_creates:
        print(
            "Without state, every created object needs an import block before the next run, or that run creates it "
            "again. If these objects do not exist yet, rerun with --allow-create and add their import blocks after "
            "the apply."
        )
    if blocked_deletes:
        print("Without state, removing a declaration stops managing the object; nothing may be deleted or replaced.")
    if blocked_creates or blocked_deletes:
        sys.exit(1)
    print("tf_plan_guard: nothing blocked.")


if __name__ == "__main__":
    main()
