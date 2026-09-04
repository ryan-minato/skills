#!/usr/bin/env python3
"""Archive every OpenSpec change whose tasks are all complete.

Scans ``openspec/changes/`` (or the directory given with ``--changes-dir``)
for changes whose ``tasks.md`` has at least one completed task and no open
task, archives each through the OpenSpec CLI non-interactively, and runs
the strict validator afterwards. Designed for an automation job on the
integration branch after merge: serialize the job on the host, rescan on
every run, and let a rejected push fail without retry.

Exit codes: 0 archived or nothing to archive; 1 the CLI or validator
failed; 2 bad arguments or an unusable changes directory.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

OPEN_TASK = re.compile(r"^\s*-\s*\[ \]\s", re.MULTILINE)
DONE_TASK = re.compile(r"^\s*-\s*\[[xX]\]\s", re.MULTILINE)
SKIP_SPECS = re.compile(r"^\s*skip_specs\s*:\s*true\s*$", re.MULTILINE)


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def completed_changes(changes_dir: Path) -> list[Path]:
    completed = []
    for change in sorted(p for p in changes_dir.iterdir() if p.is_dir() and p.name != "archive"):
        tasks = change / "tasks.md"
        if not tasks.is_file():
            continue
        text = tasks.read_text(encoding="utf-8")
        if DONE_TASK.search(text) and not OPEN_TASK.search(text):
            completed.append(change)
    return completed


def skips_specs(change: Path) -> bool:
    marker = change / ".openspec.yaml"
    return marker.is_file() and bool(SKIP_SPECS.search(marker.read_text(encoding="utf-8")))


def run(cmd: list[str], cwd: Path) -> None:
    env = {**os.environ, "OPENSPEC_NO_UPDATE_CHECK": "1"}
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        fail(f"`{' '.join(cmd)}` exited {result.returncode}", 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="archive_completed_changes.py",
        description="Archive every OpenSpec change whose tasks are all complete, then validate strictly.",
        epilog="Example: python3 scripts/archive_completed_changes.py --dry-run",
    )
    parser.add_argument("--root", default=".", help="repository root holding openspec/ (default: current directory)")
    parser.add_argument("--changes-dir", default=None, help="changes directory (default: <root>/openspec/changes)")
    parser.add_argument("--dry-run", action="store_true", help="list what would be archived; change nothing")
    parser.add_argument("--openspec", default="openspec", help="OpenSpec CLI executable (default: openspec)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments and 0 on --help
        return int(exc.code or 0)

    root = Path(args.root).resolve()
    changes_dir = Path(args.changes_dir).resolve() if args.changes_dir else root / "openspec" / "changes"
    if not changes_dir.is_dir():
        fail(f"{changes_dir} is not a directory")

    completed = completed_changes(changes_dir)
    if not completed:
        return 0
    for change in completed:
        if args.dry_run:
            print(f"would archive {change.name}")
            continue
        cmd = [args.openspec, "archive", change.name, "--yes"]
        if skips_specs(change):
            cmd.append("--skip-specs")
        print(f"archiving {change.name}")
        run(cmd, root)
    if not args.dry_run:
        run([args.openspec, "validate", "--all", "--strict", "--no-interactive"], root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
