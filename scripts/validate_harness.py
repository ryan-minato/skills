#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Validate the harness structure of the skills repository.

Usage:
    uv run scripts/validate_harness.py [--root PATH]

Exit codes:
    0  all checks passed
    1  one or more errors found
    2  bad arguments
"""

import argparse
import os
import sys
from pathlib import Path

REQUIRED_ROOT_FILES = ["AGENTS.md", "ARCHITECTURE.md", "SECURITY.md"]
REQUIRED_DOCS_FILES = ["docs/QUALITY.md", "docs/WORKFLOW.md", "docs/REFERENCES.md"]

AGENTS_MD_WARN = 100
AGENTS_MD_LIMIT = 120

ALLOWED_SCRIPT_SUFFIXES = {".py", ".ts"}


def check(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    # Required root and docs files
    for path in [*REQUIRED_ROOT_FILES, *REQUIRED_DOCS_FILES]:
        if not (root / path).is_file():
            errors.append(f"MISSING  {path}")

    # docs/specs/ directory
    specs_dir = root / "docs" / "specs"
    if not specs_dir.is_dir():
        errors.append("MISSING  docs/specs/")

    # AGENTS.md line budget
    agents_md = root / "AGENTS.md"
    if agents_md.is_file():
        n = len(agents_md.read_text(encoding="utf-8").splitlines())
        if n > AGENTS_MD_LIMIT:
            errors.append(
                f"AGENTS.md  {n} lines exceeds hard limit {AGENTS_MD_LIMIT};"
                " move content to docs/ and add pointers"
            )
        elif n > AGENTS_MD_WARN:
            warnings.append(
                f"AGENTS.md  {n} lines; approaching hard limit {AGENTS_MD_LIMIT}"
            )

    # .agents/skills symlink
    link = root / ".agents" / "skills"
    if not link.is_symlink():
        errors.append("MISSING  .agents/skills symlink")
    else:
        target = os.readlink(link)
        if target != "../skills":
            warnings.append(
                f".agents/skills points to '{target}'; expected '../skills'"
            )

    # Collect skill and spec names for bidirectional pairing check
    skills_dir = root / "skills"
    skill_names: set[str] = set()
    if skills_dir.is_dir():
        skill_names = {
            e.name
            for e in skills_dir.iterdir()
            if e.is_dir() and not e.name.startswith(".")
        }

    spec_names: set[str] = set()
    if specs_dir.is_dir():
        spec_names = {
            e.stem
            for e in specs_dir.iterdir()
            if e.is_file() and e.suffix == ".md" and not e.name.startswith(".")
        }

    # Bidirectional spec <-> skill pairing
    for name in sorted(skill_names - spec_names):
        errors.append(f"MISSING  docs/specs/{name}.md  (no spec for skills/{name}/)")
    for name in sorted(spec_names - skill_names):
        errors.append(f"MISSING  skills/{name}/  (no skill for docs/specs/{name}.md)")

    # Per-skill structural checks
    if skills_dir.is_dir():
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            name = entry.name

            if not (entry / "SKILL.md").is_file():
                errors.append(f"MISSING  skills/{name}/SKILL.md")

            # Nested skill check (skills/ must be flat)
            for sub in entry.iterdir():
                if (
                    sub.is_dir()
                    and sub.name not in ("scripts", "references", "assets")
                    and (sub / "SKILL.md").exists()
                ):
                    errors.append(
                        f"STRUCTURE  skills/{name}/{sub.name}/ looks like a nested"
                        " skill; skills/ must be flat"
                    )

            # Script language policy (skill-level)
            scripts_subdir = entry / "scripts"
            if scripts_subdir.is_dir():
                for script in sorted(scripts_subdir.iterdir()):
                    if (
                        script.is_file()
                        and not script.name.startswith(".")
                        and script.suffix not in ALLOWED_SCRIPT_SUFFIXES
                    ):
                        errors.append(
                            f"POLICY  skills/{name}/scripts/{script.name}"
                            "  only .py (uv) and .ts (deno) scripts are allowed"
                        )

    # Script language policy (repo-level scripts/)
    root_scripts = root / "scripts"
    if root_scripts.is_dir():
        for script in sorted(root_scripts.iterdir()):
            if (
                script.is_file()
                and not script.name.startswith(".")
                and script.suffix not in ALLOWED_SCRIPT_SUFFIXES
            ):
                errors.append(
                    f"POLICY  scripts/{script.name}"
                    "  only .py (uv) and .ts (deno) scripts are allowed"
                )

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate harness structure of the skills repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 = pass, 1 = errors found, 2 = bad arguments",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        metavar="PATH",
        help="repository root to validate (default: current directory)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR  {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    errors, warnings = check(root)

    for w in warnings:
        print(f"WARN   {w}", file=sys.stderr)
    for e in errors:
        print(f"ERROR  {e}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} error(s). Fix before committing.", file=sys.stderr)
        sys.exit(1)
    elif warnings:
        print(f"\nPassed with {len(warnings)} warning(s).", file=sys.stderr)
    else:
        print("OK  All checks passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
