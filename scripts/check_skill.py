#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Validate a single SKILL.md file against the skills specification.

Usage:
    uv run scripts/check_skill.py PATH

PATH can be a SKILL.md file or the skill directory containing it.

Exit codes:
    0  all checks passed (warnings may be present)
    1  one or more errors found
    2  bad arguments
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

KNOWN_KEYS = frozenset(
    {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
)
REQUIRED_KEYS = frozenset({"name", "description"})

NAME_MAX_LEN = 64
DESCRIPTION_WARN_MILD = 512
DESCRIPTION_WARN_STRONG = 768
DESCRIPTION_ERROR = 1024
COMPATIBILITY_MAX_LEN = 500
BODY_LINE_WARN = 500

_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md text.

    Returns (frontmatter_dict, body).
    Raises ValueError on missing or malformed frontmatter.
    """
    if not text.startswith("---"):
        raise ValueError(
            "FRONTMATTER  missing opening '---';"
            " SKILL.md must begin with YAML frontmatter"
        )
    lines = text.splitlines(keepends=True)
    end = next(
        (i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"),
        None,
    )
    if end is None:
        raise ValueError("FRONTMATTER  missing closing '---' delimiter")
    fm_raw = "".join(lines[1:end])
    body = "".join(lines[end + 1:])
    try:
        fm = yaml.safe_load(fm_raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"FRONTMATTER  YAML parse error: {exc}") from exc
    if fm is None:
        fm = {}
    if not isinstance(fm, dict):
        raise ValueError("FRONTMATTER  must be a YAML mapping, not a scalar")
    return fm, body


def check(skill_md: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    text = skill_md.read_text(encoding="utf-8")
    try:
        fm, body = _parse_frontmatter(text)
    except ValueError as exc:
        errors.append(str(exc))
        return errors, warnings

    # Unknown keys
    for key in sorted(set(fm.keys()) - KNOWN_KEYS):
        errors.append(
            f"FRONTMATTER  unknown key '{key}';"
            f" allowed: {', '.join(sorted(KNOWN_KEYS))}"
        )

    # Missing required keys
    for key in sorted(REQUIRED_KEYS - set(fm.keys())):
        errors.append(f"FRONTMATTER  missing required field '{key}'")

    # name
    if "name" in fm:
        name = fm["name"]
        if not isinstance(name, str):
            errors.append("FRONTMATTER  'name' must be a string")
        else:
            if not _NAME_RE.fullmatch(name):
                errors.append(
                    f"FRONTMATTER  'name' '{name}' is invalid;"
                    " use only lowercase a-z, digits 0-9, and hyphens;"
                    " no leading, trailing, or consecutive hyphens"
                )
            elif len(name) > NAME_MAX_LEN:
                errors.append(
                    f"FRONTMATTER  'name' is {len(name)} chars; max is {NAME_MAX_LEN}"
                )
            dir_name = skill_md.parent.name
            if name != dir_name:
                errors.append(
                    f"FRONTMATTER  'name' is '{name}' but parent directory is"
                    f" '{dir_name}'; they must match exactly"
                )

    # description
    if "description" in fm:
        desc = fm["description"]
        if not isinstance(desc, str):
            errors.append("FRONTMATTER  'description' must be a string")
        else:
            n = len(desc)
            if n > DESCRIPTION_ERROR:
                errors.append(
                    f"FRONTMATTER  'description' is {n} chars;"
                    f" hard limit is {DESCRIPTION_ERROR}"
                )
            elif n > DESCRIPTION_WARN_STRONG:
                warnings.append(
                    f"FRONTMATTER  'description' is {n} chars; strongly consider"
                    f" trimming (hard limit {DESCRIPTION_ERROR})"
                )
            elif n > DESCRIPTION_WARN_MILD:
                warnings.append(
                    f"FRONTMATTER  'description' is {n} chars;"
                    f" consider trimming (hard limit {DESCRIPTION_ERROR})"
                )

    # compatibility
    if "compatibility" in fm:
        compat = fm["compatibility"]
        if not isinstance(compat, str):
            errors.append("FRONTMATTER  'compatibility' must be a string")
        elif len(compat) > COMPATIBILITY_MAX_LEN:
            errors.append(
                f"FRONTMATTER  'compatibility' is {len(compat)} chars;"
                f" max is {COMPATIBILITY_MAX_LEN}"
            )

    # metadata
    if "metadata" in fm:
        meta = fm["metadata"]
        if not isinstance(meta, dict):
            errors.append("FRONTMATTER  'metadata' must be a YAML mapping")
        else:
            for k, v in meta.items():
                if not isinstance(k, str):
                    errors.append(
                        f"FRONTMATTER  metadata key {k!r} must be a string"
                    )
                if not isinstance(v, str):
                    errors.append(
                        f"FRONTMATTER  metadata['{k}'] must be a string,"
                        f" got {type(v).__name__}"
                    )

    # allowed-tools
    if "allowed-tools" in fm:
        if not isinstance(fm["allowed-tools"], str):
            errors.append(
                "FRONTMATTER  'allowed-tools' must be a space-separated string"
            )

    # Body line count
    body_lines = len(body.splitlines())
    if body_lines > BODY_LINE_WARN:
        warnings.append(
            f"BODY  {body_lines} lines exceeds {BODY_LINE_WARN};"
            " consider moving reference material to references/"
        )

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a SKILL.md file against the skills specification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:"
            "\n  0  all checks passed (warnings may be present)"
            "\n  1  one or more errors found"
            "\n  2  bad arguments"
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        metavar="PATH",
        help="path to a SKILL.md file or the skill directory containing it",
    )
    args = parser.parse_args()

    path = args.path.resolve()
    skill_md = path / "SKILL.md" if path.is_dir() else path

    if not skill_md.is_file():
        print(f"ERROR  {skill_md} not found", file=sys.stderr)
        sys.exit(2)

    errors, warnings = check(skill_md)

    label = str(args.path)
    for w in warnings:
        print(f"WARN   {label}: {w}", file=sys.stderr)
    for e in errors:
        print(f"ERROR  {label}: {e}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} error(s).", file=sys.stderr)
        sys.exit(1)
    elif warnings:
        print(f"\nPassed with {len(warnings)} warning(s).", file=sys.stderr)
    else:
        print(f"OK  {label}: all checks passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
