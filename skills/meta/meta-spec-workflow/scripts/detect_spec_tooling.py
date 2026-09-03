#!/usr/bin/env python3
"""Read-only sweep of a checkout for spec-driven-development tooling.

Reports which spec tool layouts exist (Spec-Kit, OpenSpec, Kiro, plain
committed spec directories), how many specs and change records each holds,
the tool-owned files, every agent-entrypoint line that points at a spec
artifact, and every markdown or template file that looks like it restates
requirements or acceptance criteria outside the tool's own directories.

Output is JSON on stdout; diagnostics go to stderr. The script never writes
to the checkout, so repeated runs are identical.

Exit codes: 0 success, 1 unexpected failure, 2 bad arguments.

Usage:
    python3 detect_spec_tooling.py --root .
    python3 detect_spec_tooling.py --root /path/to/repo --max-candidates 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git",
    "node_modules",
    "vendor",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
}

ENTRYPOINT_CANDIDATES = (
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    ".agents/AGENTS.md",
    ".github/copilot-instructions.md",
    ".kiro/steering",
)

SPEC_WORDS = re.compile(
    r"(\.specify|openspec|\.kiro|spec-workflow|\bspecs?/|specification|"
    r"constitution|steering)",
    re.IGNORECASE,
)

REQUIREMENT_SIGNALS = (
    ("acceptance", re.compile(r"acceptance\s+criteria", re.IGNORECASE)),
    ("shall", re.compile(r"\b(?:SHALL|MUST)\b")),
    ("user-story", re.compile(r"\buser\s+stor(?:y|ies)\b", re.IGNORECASE)),
    (
        "requirements-heading",
        re.compile(r"^#+\s+.*requirements?\b", re.IGNORECASE | re.MULTILINE),
    ),
    ("scenario", re.compile(r"^#+\s+scenario\b", re.IGNORECASE | re.MULTILINE)),
)

TEMPLATE_GLOBS = (
    ".github/ISSUE_TEMPLATE/*.yml",
    ".github/ISSUE_TEMPLATE/*.yaml",
    ".github/ISSUE_TEMPLATE/*.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/pull_request_template.md",
    ".gitlab/issue_templates/*.md",
    ".gitlab/merge_request_templates/*.md",
)


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def count_files(directory: Path, name: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.rglob(name))


def detect_spec_kit(root: Path) -> dict | None:
    hidden = root / ".specify"
    specs = root / "specs"
    numbered = []
    if specs.is_dir():
        numbered = sorted(
            p for p in specs.iterdir() if p.is_dir() and re.match(r"^\d{3,}-", p.name)
        )
    if not hidden.is_dir() and not numbered:
        return None
    constitution = hidden / "memory" / "constitution.md"
    owned = [rel(root, hidden)] if hidden.is_dir() else []
    if numbered:
        owned.append(rel(root, specs))
    return {
        "tool": "spec-kit",
        "constitution": rel(root, constitution) if constitution.is_file() else None,
        "feature_directories": [rel(root, p) for p in numbered],
        "feature_specs": sum(1 for p in numbered if (p / "spec.md").is_file()),
        "owned_paths": owned,
    }


def detect_openspec(root: Path) -> dict | None:
    base = root / "openspec"
    if not base.is_dir():
        return None
    changes = base / "changes"
    archive = changes / "archive"
    active = []
    if changes.is_dir():
        active = sorted(
            rel(root, p)
            for p in changes.iterdir()
            if p.is_dir() and p.name != "archive"
        )
    config = None
    for candidate in ("config.yaml", "config.yml", "project.md"):
        if (base / candidate).is_file():
            config = rel(root, base / candidate)
            break
    archived = (
        sum(1 for p in archive.iterdir() if p.is_dir()) if archive.is_dir() else 0
    )
    return {
        "tool": "openspec",
        "main_specs": count_files(base / "specs", "spec.md"),
        "active_changes": active,
        "archived_changes": archived,
        "config": config,
        "owned_paths": [rel(root, base)],
    }


def detect_kiro(root: Path) -> dict | None:
    base = root / ".kiro"
    if not base.is_dir():
        return None
    specs = base / "specs"
    features = (
        sorted(p for p in specs.iterdir() if p.is_dir()) if specs.is_dir() else []
    )
    steering = base / "steering"
    hooks = base / "hooks"
    required = ("requirements.md", "design.md", "tasks.md")
    return {
        "tool": "kiro",
        "feature_directories": [rel(root, p) for p in features],
        "complete_specs": sum(
            1 for p in features if all((p / f).is_file() for f in required)
        ),
        "steering_files": sorted(rel(root, p) for p in steering.glob("*.md"))
        if steering.is_dir()
        else [],
        "hook_files": sorted(rel(root, p) for p in hooks.glob("*.json"))
        if hooks.is_dir()
        else [],
        "owned_paths": [rel(root, base)],
    }


def detect_committed_documents(root: Path, owned: set[str]) -> dict | None:
    names = (
        "specs",
        "spec",
        "specifications",
        "docs/specs",
        "docs/specifications",
        "docs/spec",
    )
    candidates = []
    for name in names:
        directory = root / name
        if not directory.is_dir() or rel(root, directory) in owned:
            continue
        markdown = count_files(directory, "*.md")
        if markdown:
            candidates.append(
                {"path": rel(root, directory), "markdown_files": markdown}
            )
    if not candidates:
        return None
    return {"tool": "committed-documents", "directories": candidates}


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        print(f"warning: cannot read {path}: {error}", file=sys.stderr)
        return None


def entrypoint_pointers(root: Path) -> list[dict]:
    hits = []
    for candidate in ENTRYPOINT_CANDIDATES:
        path = root / candidate
        if path.is_dir():
            files = sorted(path.glob("*.md"))
        elif path.is_file():
            files = [path]
        else:
            continue
        for file in files:
            text = read_text(file)
            if text is None:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if SPEC_WORDS.search(line):
                    hits.append(
                        {
                            "file": rel(root, file),
                            "line": number,
                            "text": line.strip()[:200],
                        }
                    )
    return hits


def iter_markdown(root: Path, owned: set[str]):
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS and rel(root, current / d) not in owned
        ]
        for filename in filenames:
            if filename.lower().endswith(".md"):
                yield current / filename


def requirement_candidates(
    root: Path, owned: set[str], limit: int
) -> tuple[list[dict], int]:
    found = []
    for file in iter_markdown(root, owned):
        text = read_text(file)
        if text is None:
            continue
        signals = {
            name: len(pattern.findall(text)) for name, pattern in REQUIREMENT_SIGNALS
        }
        signals = {k: v for k, v in signals.items() if v}
        if signals:
            found.append({"file": rel(root, file), "signals": signals})
    for pattern in TEMPLATE_GLOBS:
        for file in root.glob(pattern):
            if not file.is_file():
                continue
            text = read_text(file)
            if text is not None and re.search(r"acceptance", text, re.IGNORECASE):
                found.append(
                    {
                        "file": rel(root, file),
                        "signals": {"acceptance-field": 1},
                        "template": True,
                    }
                )
    found.sort(key=lambda item: item["file"])
    return found[:limit], max(0, len(found) - limit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Example: python3 detect_spec_tooling.py --root .",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", default=".", help="checkout to inspect (default: current directory)"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=40,
        help="cap on requirement-restating files listed (default: 40); the total is always reported",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(
            f"error: --root {args.root!r} is not a directory; pass the checkout to inspect "
            "(for example --root .)",
            file=sys.stderr,
        )
        return 2
    if args.max_candidates < 1:
        print("error: --max-candidates must be at least 1", file=sys.stderr)
        return 2
    root = root.resolve()

    try:
        tools = [
            t
            for t in (detect_spec_kit(root), detect_openspec(root), detect_kiro(root))
            if t
        ]
        owned = {p for t in tools for p in t.get("owned_paths", [])}
        committed = detect_committed_documents(root, owned)
        if committed:
            tools.append(committed)
        candidates, truncated = requirement_candidates(root, owned, args.max_candidates)
        report = {
            "root": str(root),
            "tools": tools,
            "tool_owned_paths": sorted(owned),
            "entrypoint_pointers": entrypoint_pointers(root),
            "requirement_restating_candidates": candidates,
            "candidates_not_listed": truncated,
        }
    except OSError as error:
        print(f"error: inspection failed: {error}", file=sys.stderr)
        return 1

    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    names = ", ".join(t["tool"] for t in tools) or "none"
    print(
        f"tools: {names}; entrypoint pointers: {len(report['entrypoint_pointers'])}; "
        f"requirement-restating candidates: {len(candidates)} (+{truncated} not listed)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
