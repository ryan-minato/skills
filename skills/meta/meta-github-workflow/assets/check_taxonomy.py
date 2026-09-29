#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6,<7"]
# ///
"""Check that the repository's harness and its committed taxonomy agree.

Delivered as scripts/check_taxonomy.py and run as the `taxonomy /
consistency` CI job. Sources checked against labels.json:

- .github/release.yml           category and exclude labels ('*' ignored)
- .github/ISSUE_TEMPLATE/*.yml  top-level `labels:` (config.yml skipped)
- .github/labeler.yml           top-level label keys

A referenced-but-undefined label is exactly the failure GitHub never
reports: forms and release.yml drop unknown labels silently. Issue-form
`type:` values are checked the same way against org-taxonomy.json when that
file is present, because an unknown type is dropped just as silently.

The reverse direction is checked too: a label that is defined but consumed
by nothing is dead weight the taxonomy claims is meaningful. A label a
person applies by hand declares that in the taxonomy with
"applied_by": "human", which records its applier instead of a consumer.

Parsing these YAML files needs PyYAML, declared in the inline metadata
above, so run the script with uv, which installs it:

    uv run scripts/check_taxonomy.py [--repo-root DIR] [--labels FILE] [--types FILE]

Every problem is printed with the file to fix before the script exits.
Exit codes: 0 = the taxonomy and its consumers agree; 1 = at least one
undefined reference or unconsumed definition, or a file that cannot be
read or does not have the documented shape; 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

try:
    import yaml
except ModuleNotFoundError:
    sys.exit("check_taxonomy: error: PyYAML is missing; run the script with `uv run scripts/check_taxonomy.py`.")

KINDS = {dict: "mapping", list: "list"}


def fail(message: str) -> NoReturn:
    sys.exit(f"check_taxonomy: error: {message}")


def shaped(value: object, kind: type, where: str) -> dict | list:
    """A YAML or JSON value of the documented kind; a key written with no value reads as empty."""
    if value is None:
        return kind()
    if not isinstance(value, kind):
        fail(f"{where} must be a {KINDS[kind]}, not {type(value).__name__}; fix the file.")
    return value


def load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        fail(f"cannot read {path}: {exc}")


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read {path}: {exc}")


def load_labels(path: Path) -> tuple[set[str], set[str]]:
    """Defined label names, and the subset a person applies by hand."""
    defined, human = set(), set()
    for index, entry in enumerate(shaped(load_json(path), list, str(path))):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            fail(f"{path}[{index}] must be an object with a string 'name'; fix the file.")
        defined.add(entry["name"])
        # `applied_by` is optional: only a label a person applies declares it.
        if str(entry.get("applied_by", "")).lower() == "human":
            human.add(entry["name"])
    return defined, human


def load_types(path: Path) -> set[str]:
    """Enabled issue type names from an organization taxonomy file."""
    data = shaped(load_json(path), dict, str(path))
    names = set()
    for index, entry in enumerate(shaped(data.get("types"), list, f"{path}: types")):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            fail(f"{path}: types[{index}] must be an object with a string 'name'; fix the file.")
        # `is_enabled` is optional and defaults to enabled.
        if entry.get("is_enabled", True):
            names.add(entry["name"])
    return names


def issue_forms(root: Path) -> list[Path]:
    template_dir = root / ".github" / "ISSUE_TEMPLATE"
    forms = sorted(template_dir.glob("*.yml")) + sorted(template_dir.glob("*.yaml"))
    return [form for form in forms if form.name not in ("config.yml", "config.yaml")]


def collect_references(root: Path) -> list[tuple[str, Path]]:
    """(label, source) pairs for every label reference found."""
    refs = []
    release = root / ".github" / "release.yml"
    if release.is_file():
        data = shaped(load_yaml(release), dict, str(release))
        changelog = shaped(data.get("changelog"), dict, f"{release}: changelog")
        exclude = shaped(changelog.get("exclude"), dict, f"{release}: changelog.exclude")
        refs += [(str(label), release) for label in shaped(exclude.get("labels"), list, f"{release}: exclude.labels")]
        for index, entry in enumerate(shaped(changelog.get("categories"), list, f"{release}: changelog.categories")):
            where = f"{release}: changelog.categories[{index}]"
            category = shaped(entry, dict, where)
            refs += [(str(label), release) for label in shaped(category.get("labels"), list, f"{where}.labels")]
            category_exclude = shaped(category.get("exclude"), dict, f"{where}.exclude")
            labels = shaped(category_exclude.get("labels"), list, f"{where}.exclude.labels")
            refs += [(str(label), release) for label in labels]
    for form in issue_forms(root):
        labels = shaped(load_yaml(form), dict, str(form)).get("labels")
        # A form may list its labels as a comma-separated string.
        if isinstance(labels, str):
            labels = [part.strip() for part in labels.split(",") if part.strip()]
        refs += [(str(label), form) for label in shaped(labels, list, f"{form}: labels")]
    labeler = root / ".github" / "labeler.yml"
    if labeler.is_file():
        refs += [(str(label), labeler) for label in shaped(load_yaml(labeler), dict, str(labeler))]
    return refs


def collect_type_references(root: Path) -> list[tuple[str, Path]]:
    """(type, source) pairs for every issue-form `type:` found."""
    refs = []
    for form in issue_forms(root):
        issue_type = shaped(load_yaml(form), dict, str(form)).get("type")
        if issue_type is not None and not isinstance(issue_type, str):
            fail(f"{form}: type must be a string; fix the file.")
        if issue_type and issue_type.strip():
            refs.append((issue_type.strip(), form))
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that release.yml, issue forms, and labeler.yml reference only labels defined in the committed "
            "taxonomy, and that every defined label has a consumer."
        ),
        epilog="Exit codes: 0 consistent, 1 problems (listed) or unreadable files, 2 bad arguments. "
        "Example: uv run scripts/check_taxonomy.py",
    )
    parser.add_argument("--repo-root", default=".", help="repository root (default: .)")
    parser.add_argument(
        "--labels",
        default=".github/labels.json",
        help="taxonomy file, relative to --repo-root (default: .github/labels.json)",
    )
    parser.add_argument(
        "--types",
        default=".github/org-taxonomy.json",
        help=(
            "organization taxonomy file, relative to --repo-root; issue-form `type:` values are checked against it "
            "when it exists (default: .github/org-taxonomy.json)"
        ),
    )
    args = parser.parse_args()
    root = Path(args.repo_root)
    if not root.is_dir():
        parser.error(f"--repo-root {root} is not a directory")
    labels_path = root / args.labels
    if not labels_path.is_file():
        parser.error(f"taxonomy file {labels_path} not found; pass --labels")

    defined, human_applied = load_labels(labels_path)
    references = collect_references(root)
    problems = []

    for label, source in references:
        if label not in defined and label != "*":
            problems.append(
                f"undefined label {label!r} referenced by {source} — add it to {labels_path} (and sync it to the "
                "repository) or fix the reference; GitHub drops unknown labels silently."
            )

    types_path = root / args.types
    type_references = collect_type_references(root)
    if types_path.is_file():
        defined_types = load_types(types_path)
        for issue_type, source in type_references:
            if issue_type not in defined_types:
                problems.append(
                    f"undefined issue type {issue_type!r} referenced by {source} — add it to {types_path} and sync "
                    "it to the organization, or fix the reference; GitHub drops unknown types silently."
                )
    elif type_references:
        count = len(type_references)
        print(
            f"note: {count} issue-form `type:` {'value is' if count == 1 else 'values are'} unchecked because "
            f"{types_path} does not exist; on a personal account remove the `type:` keys, and in an organization "
            "commit the taxonomy file.",
            file=sys.stderr,
        )

    # Reverse direction: a definition nothing consumes is not "meaningful".
    consumed = {label for label, _ in references}
    for label in sorted(defined - consumed - human_applied):
        problems.append(
            f"unconsumed label {label!r} is defined in {labels_path} but no release.yml category, issue form, or "
            'labeler rule references it — give it a consumer, delete it, or set "applied_by": "human" on it when '
            "a person applies it by hand."
        )

    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(
        f"taxonomy consistent: {len(defined)} labels defined, all references resolve, and every definition has a "
        "consumer."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
