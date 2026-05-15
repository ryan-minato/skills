#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "pyyaml>=6.0",
# ]
# requires-python = ">=3.11"
# ///
"""Validate SKILL.md frontmatter fields and body length.

Usage:
  uv run scripts/check_metadata.py --skill PATH

PATH can be a skill directory or its SKILL.md file.

Exit codes:
  0  all checks passed (warnings may be present)
  1  one or more errors found
  2  bad arguments
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

KNOWN_FIELDS = frozenset(
    {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
)
REQUIRED_FIELDS = frozenset({"name", "description"})

_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

DESCRIPTION_WARN_MILD = 512
DESCRIPTION_WARN_STRONG = 768
DESCRIPTION_MAX = 1024
COMPATIBILITY_MAX = 500
BODY_WARN = 500
BODY_WARN_STRONG = 1000


def _issue(location: str, level: str, reason: str, suggestion: str) -> dict:
    return {"location": location, "level": level, "reason": reason, "suggestion": suggestion}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Returns (frontmatter_dict, body_text). Raises ValueError on parse failure."""
    if not text.startswith("---"):
        raise ValueError("missing opening '---'")
    lines = text.splitlines(keepends=True)
    end = next(
        (i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"),
        None,
    )
    if end is None:
        raise ValueError("missing closing '---' delimiter")
    fm_raw = "".join(lines[1:end])
    body = "".join(lines[end + 1:])
    try:
        fm = yaml.safe_load(fm_raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc
    if fm is None:
        fm = {}
    if not isinstance(fm, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return fm, body


def validate(skill_md: Path) -> list[dict]:
    issues: list[dict] = []
    skill_dir = skill_md.parent
    base = _rel(skill_md)

    text = skill_md.read_text(encoding="utf-8")
    try:
        fm, body = _parse_frontmatter(text)
    except ValueError as exc:
        issues.append(_issue(
            f"{base}:frontmatter",
            "error",
            f"Cannot parse frontmatter: {exc}.",
            "Ensure SKILL.md begins with '---', contains valid YAML, and closes the block with '---' on its own line.",
        ))
        return issues

    # Unknown fields
    for key in sorted(fm):
        if key not in KNOWN_FIELDS:
            issues.append(_issue(
                f"{base}:frontmatter:{key}",
                "warning",
                f"Unknown frontmatter field '{key}'.",
                f"Remove '{key}'. Supported fields: {', '.join(sorted(KNOWN_FIELDS))}.",
            ))

    # Required fields
    for key in sorted(REQUIRED_FIELDS):
        if key not in fm:
            issues.append(_issue(
                f"{base}:frontmatter",
                "error",
                f"Required field '{key}' is missing.",
                f"Add '{key}' to the frontmatter block.",
            ))

    # name
    name = fm.get("name")
    if name is not None:
        if not isinstance(name, str):
            issues.append(_issue(
                f"{base}:frontmatter:name",
                "error",
                f"'name' must be a string, got {type(name).__name__}.",
                "Set 'name' to a lowercase hyphenated string matching the directory name.",
            ))
        else:
            if len(name) > 64:
                issues.append(_issue(
                    f"{base}:frontmatter:name",
                    "error",
                    f"'name' is {len(name)} chars; maximum is 64.",
                    "Shorten the skill name to 64 characters or fewer.",
                ))
            if not _NAME_RE.fullmatch(name):
                issues.append(_issue(
                    f"{base}:frontmatter:name",
                    "error",
                    f"'name' value '{name}' contains invalid characters or structure.",
                    "Use only lowercase a-z, digits 0-9, and single hyphens. "
                    "No leading, trailing, or consecutive hyphens (e.g. 'my-skill', not 'My_Skill' or 'my--skill').",
                ))
            if name != skill_dir.name:
                issues.append(_issue(
                    f"{base}:frontmatter:name",
                    "error",
                    f"'name' is '{name}' but the parent directory is '{skill_dir.name}'. They must match exactly.",
                    f"Either rename the directory to '{name}' or change 'name' to '{skill_dir.name}'.",
                ))

    # description
    desc = fm.get("description")
    if desc is not None:
        if not isinstance(desc, str):
            issues.append(_issue(
                f"{base}:frontmatter:description",
                "error",
                f"'description' must be a string, got {type(desc).__name__}.",
                "Use a YAML string, optionally with '>' for a folded block scalar.",
            ))
        else:
            length = len(desc)
            if length > DESCRIPTION_MAX:
                issues.append(_issue(
                    f"{base}:frontmatter:description",
                    "error",
                    f"'description' is {length} chars; maximum is {DESCRIPTION_MAX}.",
                    "Trim to 1024 chars or fewer. "
                    "Keep the first sentence as the capability summary and the most distinctive trigger conditions.",
                ))
            elif length > DESCRIPTION_WARN_STRONG:
                issues.append(_issue(
                    f"{base}:frontmatter:description",
                    "warning",
                    f"'description' is {length} chars (>{DESCRIPTION_WARN_STRONG}). "
                    "All skill descriptions are loaded into the system prompt simultaneously; "
                    "long descriptions consume significant context budget.",
                    "Trim to under 512 chars. Remove redundant trigger phrasings; keep the most distinctive ones.",
                ))
            elif length > DESCRIPTION_WARN_MILD:
                issues.append(_issue(
                    f"{base}:frontmatter:description",
                    "warning",
                    f"'description' is {length} chars (>{DESCRIPTION_WARN_MILD}). "
                    "Shorter descriptions are preferable for context efficiency.",
                    "Consider trimming to under 512 chars while preserving key trigger conditions.",
                ))

    # compatibility
    compat = fm.get("compatibility")
    if compat is not None and isinstance(compat, str) and len(compat) > COMPATIBILITY_MAX:
        issues.append(_issue(
            f"{base}:frontmatter:compatibility",
            "error",
            f"'compatibility' is {len(compat)} chars; maximum is {COMPATIBILITY_MAX}.",
            "Trim 'compatibility' to 500 chars or fewer.",
        ))

    # metadata must be a flat dict
    meta = fm.get("metadata")
    if meta is not None and not isinstance(meta, dict):
        issues.append(_issue(
            f"{base}:frontmatter:metadata",
            "error",
            f"'metadata' must be a YAML mapping (key: value pairs), got {type(meta).__name__}.",
            "Use a flat YAML mapping, e.g.:\nmetadata:\n  author: my-org\n  version: '1.0'",
        ))

    # body length
    body_lines = len(body.splitlines())
    if body_lines > BODY_WARN_STRONG:
        issues.append(_issue(
            f"{base}:body",
            "warning",
            f"SKILL.md body is {body_lines} lines (>{BODY_WARN_STRONG}). "
            "Agents loading very long skills may follow irrelevant clauses, wasting context and causing incorrect behavior.",
            "Split content into 'references/' files. Keep SKILL.md to workflow steps, gotchas, and always-needed context. "
            "Move detailed or conditional content to 'references/<topic>.md' with conditional load instructions.",
        ))
    elif body_lines > BODY_WARN:
        issues.append(_issue(
            f"{base}:body",
            "warning",
            f"SKILL.md body is {body_lines} lines (>{BODY_WARN}).",
            "Consider moving content needed only under specific conditions to 'references/<topic>.md' "
            "and adding a conditional load instruction in SKILL.md.",
        ))

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate SKILL.md frontmatter fields and body length.",
        epilog="Example: uv run scripts/check_metadata.py --skill skills/my-skill/",
    )
    parser.add_argument(
        "--skill",
        required=True,
        metavar="PATH",
        help="Path to a skill directory or its SKILL.md file.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output results as JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    path = Path(args.skill)
    if path.is_dir():
        skill_md = path / "SKILL.md"
    elif path.name == "SKILL.md":
        skill_md = path
    else:
        parser.error(
            f"--skill must be a skill directory or a SKILL.md file. Got: {args.skill!r}"
        )

    if not skill_md.exists():
        print(f"Error: '{skill_md}' does not exist.", file=sys.stderr)
        sys.exit(1)

    issues = validate(skill_md)
    errors = [i for i in issues if i["level"] == "error"]

    if args.json_output:
        print(json.dumps(issues, indent=2))
    else:
        if not issues:
            print(f"OK  {_rel(skill_md)}: all metadata checks passed.")
        for item in issues:
            label = "ERROR  " if item["level"] == "error" else "WARNING"
            print(f"{label}  [{item['location']}]")
            print(f"           Reason:     {item['reason']}")
            print(f"           Suggestion: {item['suggestion']}")
            print()

    if errors:
        print(f"{len(errors)} error(s) found.", file=sys.stderr)
        sys.exit(1)
    elif issues:
        print(f"{len(issues)} warning(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
