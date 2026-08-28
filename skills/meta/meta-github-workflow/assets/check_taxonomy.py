#!/usr/bin/env python3
"""Check that the repository's harness and its committed taxonomy agree.

Copy into the target repository at scripts/check_taxonomy.py and run it as
the `taxonomy / consistency` CI job. Sources checked against labels.json:

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

Dependency exception: parsing workflow-adjacent YAML requires PyYAML,
which is not in the standard library. This is the repository's one
documented script dependency: install with `pip install pyyaml==6.0.2`
(the CI job does the same). --help works without it.

Usage:
    python3 scripts/check_taxonomy.py [--repo-root DIR] [--labels FILE]
                                      [--types FILE]

Exit codes: 0 = the taxonomy and its consumers agree, 1 = at least one
undefined reference or unconsumed definition (each printed with its source
file), 2 = bad arguments, unreadable inputs, or PyYAML missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fail(code: int, message: str) -> None:
    print(f"check_taxonomy: error: {message}", file=sys.stderr)
    sys.exit(code)


def import_yaml():
    """Import PyYAML after argument parsing so --help needs no dependency."""
    try:
        import yaml
    except ImportError:
        fail(
            2,
            "PyYAML is required — install with `pip install pyyaml==6.0.2` "
            "(the one documented dependency exception for this repository's "
            "scripts).",
        )
    return yaml


def load_defined(labels_path: Path) -> set:
    try:
        entries = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(2, f"cannot read {labels_path}: {exc}")
    if not isinstance(entries, list):
        fail(2, f"{labels_path} must be a JSON array of label objects")
    return {str(e.get("name", "")) for e in entries if isinstance(e, dict)}


def load_yaml_file(path: Path):
    yaml = import_yaml()
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        fail(2, f"cannot read {path}: {exc}")
    except yaml.YAMLError as exc:
        fail(2, f"cannot parse {path}: {exc}")


def as_dict(value):
    """A YAML key written with no value parses as None, not {}."""
    return value if isinstance(value, dict) else {}


def as_list(value):
    """A YAML key written with no value parses as None, not []."""
    return value if isinstance(value, list) else []


def collect_references(root: Path) -> list:
    """Return (label, source) pairs for every label reference found."""
    refs = []
    release = root / ".github" / "release.yml"
    if release.is_file():
        changelog = as_dict(as_dict(load_yaml_file(release)).get("changelog"))
        for label in as_list(as_dict(changelog.get("exclude")).get("labels")):
            refs.append((str(label), release))
        for entry in as_list(changelog.get("categories")):
            category = as_dict(entry)
            for label in as_list(category.get("labels")):
                refs.append((str(label), release))
            for label in as_list(as_dict(category.get("exclude")).get("labels")):
                refs.append((str(label), release))
    template_dir = root / ".github" / "ISSUE_TEMPLATE"
    if template_dir.is_dir():
        for form in sorted(template_dir.glob("*.yml")) + sorted(
            template_dir.glob("*.yaml")
        ):
            if form.name in ("config.yml", "config.yaml"):
                continue
            form_data = as_dict(load_yaml_file(form))
            labels = form_data.get("labels")
            if isinstance(labels, str):
                labels = [part.strip() for part in labels.split(",") if part.strip()]
            for label in as_list(labels):
                refs.append((str(label), form))
    labeler = root / ".github" / "labeler.yml"
    if labeler.is_file():
        for label in as_dict(load_yaml_file(labeler)):
            refs.append((str(label), labeler))
    return refs


def collect_type_references(root: Path) -> list:
    """Return (type, source) pairs for every issue-form `type:` found."""
    refs = []
    template_dir = root / ".github" / "ISSUE_TEMPLATE"
    if not template_dir.is_dir():
        return refs
    for form in sorted(template_dir.glob("*.yml")) + sorted(
        template_dir.glob("*.yaml")
    ):
        if form.name in ("config.yml", "config.yaml"):
            continue
        issue_type = as_dict(load_yaml_file(form)).get("type")
        if isinstance(issue_type, str) and issue_type.strip():
            refs.append((issue_type.strip(), form))
    return refs


def load_human_applied(path: Path) -> set:
    """Labels the taxonomy declares a person applies, so no file consumes them."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry["name"])
        for entry in data
        if isinstance(entry, dict)
        and entry.get("name")
        and str(entry.get("applied_by", "")).lower() == "human"
    }


def load_defined_types(path: Path) -> set:
    """Read the enabled issue type names from an org taxonomy file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(2, f"cannot read {path}: {exc}")
    entries = data.get("types") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        fail(2, f'{path}: expected an object with a "types" array')
    return {
        str(entry["name"])
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("name")
        and entry.get("is_enabled", True)
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that release.yml, issue forms, and labeler.yml reference "
            "only labels defined in the committed taxonomy. Exit codes: "
            "0 = consistent, 1 = undefined references (listed), 2 = bad "
            "arguments, unreadable inputs, or PyYAML missing."
        ),
        epilog="Example: python3 scripts/check_taxonomy.py",
    )
    parser.add_argument(
        "--repo-root", default=".", help="repository root (default: .)"
    )
    parser.add_argument(
        "--labels",
        default=".github/labels.json",
        help="taxonomy file, relative to --repo-root (default: .github/labels.json)",
    )
    parser.add_argument(
        "--types",
        default=".github/org-taxonomy.json",
        help=(
            "organization taxonomy file, relative to --repo-root; issue-form "
            "`type:` values are checked against it when it exists "
            "(default: .github/org-taxonomy.json)"
        ),
    )
    args = parser.parse_args()

    root = Path(args.repo_root)
    if not root.is_dir():
        fail(2, f"--repo-root {root} is not a directory")
    labels_path = root / args.labels
    if not labels_path.is_file():
        fail(2, f"taxonomy file {labels_path} not found — pass --labels")

    defined = load_defined(labels_path)
    human_applied = load_human_applied(labels_path)
    references = collect_references(root)
    problems = False

    for label, source in references:
        if label not in defined and label != "*":
            print(
                f"undefined label {label!r} referenced by {source} — add it to "
                f"{labels_path} (and sync it to the repository) or fix the "
                "reference; GitHub drops unknown labels silently.",
                file=sys.stderr,
            )
            problems = True

    types_path = root / args.types
    type_references = collect_type_references(root)
    if types_path.is_file():
        defined_types = load_defined_types(types_path)
        for issue_type, source in type_references:
            if issue_type not in defined_types:
                print(
                    f"undefined issue type {issue_type!r} referenced by {source} "
                    f"— add it to {types_path} and sync it to the organization, "
                    "or fix the reference; GitHub drops unknown types silently.",
                    file=sys.stderr,
                )
                problems = True
    elif type_references:
        count = len(type_references)
        print(
            f"note: {count} issue-form `type:` "
            f"{'value is' if count == 1 else 'values are'} "
            f"unchecked because {types_path} does not exist; on a personal "
            "account remove the `type:` keys, and in an organization commit "
            "the taxonomy file.",
            file=sys.stderr,
        )

    # Reverse direction: a definition nothing consumes is not "meaningful".
    consumed = {label for label, _ in references}
    for label in sorted(defined):
        if label in consumed or label in human_applied:
            continue
        print(
            f"unconsumed label {label!r} is defined in {labels_path} but no "
            "release.yml category, issue form, or labeler rule references it "
            '— give it a consumer, delete it, or set "applied_by": "human" on '
            "it when a person applies it by hand.",
            file=sys.stderr,
        )
        problems = True

    if problems:
        return 1
    print(
        f"taxonomy consistent: {len(defined)} labels defined, all references "
        "resolve, and every definition has a consumer."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
