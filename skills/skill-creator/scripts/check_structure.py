#!/usr/bin/env -S uv run
# /// script
# dependencies = []
# requires-python = ">=3.11"
# ///
"""Validate a skill directory's file and folder structure.

Usage:
  uv run scripts/check_structure.py --skill PATH

Exit codes:
  0  all checks passed (warnings may be present)
  1  one or more errors found
  2  bad arguments
"""

import argparse
import json
import sys
from pathlib import Path

ALLOWED_ROOT = frozenset({"SKILL.md", "references", "assets", "scripts"})
ALLOWED_SCRIPT_EXTS = frozenset({".py", ".rb", ".ts"})


def _issue(location: str, level: str, reason: str, suggestion: str) -> dict:
    return {"location": location, "level": level, "reason": reason, "suggestion": suggestion}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def validate(skill_dir: Path) -> list[dict]:
    issues: list[dict] = []
    base = _rel(skill_dir)

    # Root-level entries
    for entry in sorted(skill_dir.iterdir()):
        name = entry.name
        if name in ALLOWED_ROOT:
            continue
        loc = f"{base}/{name}"
        if name == "README.md":
            issues.append(_issue(
                loc,
                "warning",
                "README.md found in the skill root.",
                "Skill directories are not typically distributed with a README — "
                "the 'description' field and SKILL.md body serve that purpose. "
                "Remove README.md unless the deployment environment explicitly requires it.",
            ))
        elif name.startswith("."):
            issues.append(_issue(
                loc,
                "warning",
                f"Hidden entry '{name}' found in the skill root.",
                "Skill roots should contain only SKILL.md, references/, assets/, and scripts/. "
                "Remove this entry if it is not required at runtime.",
            ))
        else:
            issues.append(_issue(
                loc,
                "warning",
                f"Unexpected entry '{name}' at the skill root.",
                "Skill roots should contain only SKILL.md, references/, assets/, and scripts/. "
                "Move this into one of those subdirectories or remove it.",
            ))

    # scripts/ directory contents
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.is_dir():
        for entry in sorted(scripts_dir.iterdir()):
            loc = f"{base}/scripts/{entry.name}"
            if entry.is_dir():
                issues.append(_issue(
                    f"{loc}/",
                    "warning",
                    f"Subdirectory '{entry.name}' found inside scripts/.",
                    "scripts/ should contain only flat script files. "
                    "Inline any shared logic or move it to the skill root if it is a support file.",
                ))
                continue
            ext = entry.suffix.lower()
            if ext == ".js":
                issues.append(_issue(
                    loc,
                    "warning",
                    f"'{entry.name}' is a plain JavaScript file (.js).",
                    "Use TypeScript (.ts) instead. Both Deno and Bun execute .ts files natively "
                    "with no compilation step, providing type safety at no extra cost.",
                ))
            elif ext == ".sh":
                issues.append(_issue(
                    loc,
                    "warning",
                    f"'{entry.name}' is a shell script (.sh) that runs only on Unix-like systems.",
                    "Shell scripts are not portable across all agent environments. "
                    "Consider rewriting in Python (uv) or TypeScript (Deno/Bun) for cross-platform support. "
                    "If Unix-only execution is intentional, document the requirement in SKILL.md's 'compatibility' field.",
                ))
            elif ext == ".ps1":
                issues.append(_issue(
                    loc,
                    "warning",
                    f"'{entry.name}' is a PowerShell script (.ps1) that runs only in environments with PowerShell installed.",
                    "PowerShell scripts are not portable across all agent environments. "
                    "Consider rewriting in Python (uv) or TypeScript (Deno/Bun). "
                    "If PowerShell-only execution is intentional, document the requirement in SKILL.md's 'compatibility' field.",
                ))
            elif ext not in ALLOWED_SCRIPT_EXTS:
                issues.append(_issue(
                    loc,
                    "warning",
                    f"Unexpected file type '{ext}' in scripts/ ('{entry.name}').",
                    f"scripts/ should contain only .py (Python/uv), .rb (Ruby), or .ts (Deno/Bun) files. "
                    f"Remove '{entry.name}' or convert it to a supported format.",
                ))

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a skill directory's file and folder structure.",
        epilog="Example: uv run scripts/check_structure.py --skill skills/my-skill/",
    )
    parser.add_argument(
        "--skill",
        required=True,
        metavar="PATH",
        help="Path to the skill directory.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output results as JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    skill_dir = Path(args.skill)
    if not skill_dir.is_dir():
        parser.error(f"--skill must be a skill directory. Got: {args.skill!r}")

    if not (skill_dir / "SKILL.md").exists():
        print(f"Error: '{skill_dir / 'SKILL.md'}' does not exist.", file=sys.stderr)
        sys.exit(1)

    issues = validate(skill_dir)
    errors = [i for i in issues if i["level"] == "error"]

    if args.json_output:
        print(json.dumps(issues, indent=2))
    else:
        if not issues:
            print(f"OK  {_rel(skill_dir)}: all structure checks passed.")
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
