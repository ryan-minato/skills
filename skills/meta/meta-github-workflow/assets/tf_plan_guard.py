#!/usr/bin/env python3
"""Plan guard for the project's stateless platform-settings tf.

Delivered as scripts/tf_plan_guard.py. It reads the JSON form of a saved
plan (`tofu show -json plan.out` or `terraform show -json plan.out`) and
enforces the rule that removing a declaration only stops managing an
object: any delete or replace action fails the guard. It also lists, for
the reviewer, every create that imports nothing (a first creation whose
ID must be written into an import block afterwards) and every update to
an attribute the configuration does not declare (a reset to the
provider's default). Python 3.10+, standard library only.

Usage:
    tofu plan -out plan.out && tofu show -json plan.out > plan.json
    python3 scripts/tf_plan_guard.py plan.json
    tofu show -json plan.out | python3 scripts/tf_plan_guard.py -

Exit codes: 0 = no delete or replace (listings are informational);
1 = a delete or replace is planned, or the input is not a readable plan
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
    if not isinstance(plan, dict) or not isinstance(plan.get("resource_changes", []), list):
        fail(f"{label} is not a plan JSON: expected an object with a resource_changes list")
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
        description="Fail on any delete or replace in a tf plan JSON; list un-imported creates and undeclared updates."
    )
    parser.add_argument("plan", help="path to the `show -json` output of a saved plan, or - for standard input")
    args = parser.parse_args()

    plan = read_plan(args.plan)
    declared = declared_attributes(plan.get("configuration", {}).get("root_module", {}))

    blocked, creates, resets = [], [], []
    for rc in plan.get("resource_changes", []):
        address = rc["address"]
        change = rc["change"]
        actions = change["actions"]
        if "delete" in actions:
            kind = "replace" if "create" in actions else "delete"
            blocked.append(f"{address}: {kind} planned ({', '.join(actions)})")
        elif actions == ["create"] and not change.get("importing"):
            creates.append(f"{address}: created, not imported; write its ID into an import block after the apply")
        elif "update" in actions:
            config_address = INDEX_RE.sub("", address)
            undeclared = sorted(changed_attributes(change) - declared.get(config_address, set()))
            if undeclared:
                resets.append(f"{address}: undeclared attributes change: {', '.join(undeclared)}")

    for title, lines in (
        ("Creates without import", creates),
        ("Undeclared attributes that change (reset to the provider's value)", resets),
        ("Blocked: delete or replace", blocked),
    ):
        if lines:
            print(f"{title}:")
            for line in lines:
                print(f"  {line}")
    if blocked:
        print("Removing a declaration stops managing the object; nothing may be deleted or replaced.")
        sys.exit(1)
    print("tf_plan_guard: no delete or replace planned.")


if __name__ == "__main__":
    main()
