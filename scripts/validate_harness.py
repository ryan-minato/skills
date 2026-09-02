#!/usr/bin/env python3
"""Validate repository harness synchronization points."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from check_pr_policy import REQUIRED_CHECKLIST_ITEMS, REQUIRED_HEADINGS
from sync_issue_metadata import CATALOG_LABELS, PRIORITY_LABELS

REPO_ROOT = Path(__file__).resolve().parent.parent
RUFF_VERSION = "0.16.5"


def read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def form_options(path: str, label: str) -> list[str] | None:
    lines = read(path).splitlines()
    in_item = False
    found_label = False
    in_options = False
    options: list[str] = []
    for line in lines:
        if line.startswith("  - type:"):
            if found_label:
                break
            in_item = True
            in_options = False
            continue
        if not in_item:
            continue
        if line.strip() == f"label: {label}":
            found_label = True
            continue
        if found_label and line.strip() == "options:":
            in_options = True
            continue
        if in_options:
            match = re.match(r"^\s{8}-\s+(.+?)\s*$", line)
            if match:
                options.append(match.group(1))
            elif line.strip() and len(line) - len(line.lstrip()) <= 6:
                break
    return options if found_label else None


def validate() -> list[str]:
    errors: list[str] = []

    agents = read("AGENTS.md")
    pointers = (
        ".agents/knowledge/project-workflow.md",
        ".agents/knowledge/agent-authority.md",
        ".agents/knowledge/github/checks.md",
        ".agents/knowledge/github/platform-settings.md",
    )
    for pointer in pointers:
        if pointer not in agents:
            errors.append(f"AGENTS.md: missing discovery pointer to `{pointer}`")
        if not (REPO_ROOT / pointer).exists():
            errors.append(f"{pointer}: discovery target does not exist")
    if "`change-workflow`" not in agents:
        errors.append("AGENTS.md: missing discovery route for `change-workflow`")
    if not (REPO_ROOT / ".agents/skills/change-workflow/SKILL.md").is_file():
        errors.append(".agents/skills/change-workflow/SKILL.md: target does not exist")

    template = read(".github/PULL_REQUEST_TEMPLATE.md")
    headings = re.findall(r"^## (.+?)\s*$", template, re.MULTILINE)
    if tuple(headings) != REQUIRED_HEADINGS:
        errors.append(
            ".github/PULL_REQUEST_TEMPLATE.md: headings differ from "
            "scripts/check_pr_policy.py REQUIRED_HEADINGS"
        )
    checklist_items = tuple(
        re.findall(r"^- \[[ xX]\] (.+?)\s*$", template, re.MULTILINE)
    )
    if checklist_items != REQUIRED_CHECKLIST_ITEMS:
        errors.append(
            ".github/PULL_REQUEST_TEMPLATE.md: checklist differs from "
            "scripts/check_pr_policy.py REQUIRED_CHECKLIST_ITEMS"
        )

    import json

    labels = {item["name"] for item in json.loads(read(".github/labels.json"))}
    expected_labels = set(PRIORITY_LABELS.values()) | set(CATALOG_LABELS.values())
    missing_labels = sorted(expected_labels - labels)
    if missing_labels:
        errors.append(
            ".github/labels.json: missing managed labels: " + ", ".join(missing_labels)
        )

    forms = {
        ".github/ISSUE_TEMPLATE/new-skill.yml": True,
        ".github/ISSUE_TEMPLATE/modify-skill.yml": True,
        ".github/ISSUE_TEMPLATE/project-improvement.yml": False,
    }
    for form, requires_catalog in forms.items():
        priority = form_options(form, "Priority")
        if priority != list(PRIORITY_LABELS):
            errors.append(
                f"{form}: Priority options must match scripts/sync_issue_metadata.py"
            )
        catalog = form_options(form, "Catalog")
        expected_catalog = list(CATALOG_LABELS) if requires_catalog else None
        if catalog != expected_catalog:
            errors.append(
                f"{form}: Catalog field must match scripts/sync_issue_metadata.py"
            )

    workflow_checks = {
        ".github/workflows/checks.yml": "name: checks / quality",
        ".github/workflows/pr-policy.yml": "name: pr / policy",
        ".github/workflows/secret.yml": "scan-secrets:",
    }
    checks_register = read(".agents/knowledge/github/checks.md")
    for workflow, marker in workflow_checks.items():
        if marker not in read(workflow):
            errors.append(f"{workflow}: missing stable check marker `{marker}`")
        check_name = marker.removeprefix("name: ").removesuffix(":")
        if f"`{check_name}`" not in checks_register:
            errors.append(
                ".agents/knowledge/github/checks.md: missing registered check "
                f"`{check_name}`"
            )

    checks_workflow = read(".github/workflows/checks.yml")
    devcontainer = read(".devcontainer/devcontainer.json")
    pre_commit = read(".pre-commit-config.yaml")
    tools_match = re.search(r'"toolsToInstall":\s*"([^"]+)"', devcontainer)
    devcontainer_tools = set(tools_match.group(1).split(",")) if tools_match else set()
    expected_tools = {"pre-commit", "rust-just", "ruff"}
    if devcontainer_tools != expected_tools:
        errors.append(
            ".devcontainer/devcontainer.json: toolsToInstall must use the latest "
            "pre-commit, rust-just, and ruff releases without version pins"
        )
    for package in ("pre-commit", "rust-just", "ruff"):
        if package not in checks_workflow:
            errors.append(
                f".github/workflows/checks.yml: missing `{package}` installation"
            )
        if re.search(rf"\b{re.escape(package)}==", checks_workflow):
            errors.append(
                ".github/workflows/checks.yml: check tools must use latest "
                f"releases; remove the `{package}` version pin"
            )
    if f"rev: v{RUFF_VERSION}" not in pre_commit:
        errors.append(
            ".pre-commit-config.yaml: Ruff revision differs from the harness "
            f"tool version {RUFF_VERSION}"
        )

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(
            f"FAIL: {len(errors)} harness synchronization problem(s)", file=sys.stderr
        )
        for error in errors:
            print(f"  * {error}", file=sys.stderr)
        return 1
    print("OK: repository harness synchronization checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
