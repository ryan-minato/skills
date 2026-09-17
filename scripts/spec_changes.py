#!/usr/bin/env python3
"""Operate on the OpenSpec changes a pull or merge request touches.

A request's *related changes* are the directories under the changes
directory (default ``openspec/changes/``) whose files the diff between
``--base`` and ``--head`` touches; a directory under ``archive/`` counts
under its change name with the leading ``YYYY-MM-DD-`` stripped. Each
change is ``active`` (its directory exists at the head), ``archived`` (an
archive directory for it exists at the head), or ``removed``.

Every subcommand except ``archive`` reads the head through git plumbing, so
the head never needs a checkout: ``related``, ``status``, ``show``, and
``labels`` are pure reads; ``check`` also runs the strict validator over
the working tree; ``archive`` edits the working tree through the OpenSpec
CLI (``openspec archive <name> --yes``, plus ``--skip-specs`` for a change
whose ``.openspec.yaml`` sets ``skip_specs: true``; flags verified against
OpenSpec 1.12.0 on 2026-09-17).

Exit codes: 0 success; 1 a failure, a finding, or a refusal; 2 bad
arguments, an unresolvable ref, or a tree that is not a git repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

OPEN_TASK = re.compile(r"^\s*-\s*\[ \]\s+(.*)$", re.MULTILINE)
DONE_TASK = re.compile(r"^\s*-\s*\[[xX]\]\s", re.MULTILINE)
SKIP_SPECS = re.compile(r"^\s*skip_specs\s*:\s*true\s*$", re.MULTILINE)
ARCHIVE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}-")
DOCS = ("proposal", "design", "tasks", "specs")

TRIGGER_LABEL = "spec/archive"
ARCHIVED_LABELS = ("spec/unarchived", "spec/archived")
PROGRESS_LABELS = ("spec/not-started", "spec/in-progress", "spec/done")
MANAGED_LABELS = ARCHIVED_LABELS + PROGRESS_LABELS


class Usage(Exception):
    """Bad arguments or an unusable repository (exit 2)."""


class Failure(Exception):
    """A tool failure, a finding, or a refusal (exit 1)."""


# --- git plumbing ---------------------------------------------------------


def git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise Failure(f"`git {' '.join(args)}` failed: {result.stderr.strip()}")
    return result.stdout


def resolve(root: Path, ref: str, what: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Usage(
            f"{what} {ref!r} does not resolve to a commit in {root}; fetch it first "
            "(for a pull request head: `git fetch origin refs/pull/<n>/head`)."
        )
    return result.stdout.strip()


def diff_names(root: Path, base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", "--no-renames", f"{base}...{head}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Usage(
            f"`git diff {base}...{head}` failed ({result.stderr.strip()}); the merge base must be reachable — "
            "fetch with full history (fetch-depth 0) rather than a shallow clone."
        )
    return [line for line in result.stdout.splitlines() if line]


def tree_entries(root: Path, ref: str, path: str) -> list[str]:
    out = git(root, "ls-tree", "--name-only", ref, f"{path.rstrip('/')}/", check=False)
    return [line.rsplit("/", 1)[-1] for line in out.splitlines() if line]


def tree_files(root: Path, ref: str, path: str) -> list[str]:
    out = git(root, "ls-tree", "-r", "--name-only", ref, f"{path.rstrip('/')}/", check=False)
    return [line for line in out.splitlines() if line]


def is_tree(root: Path, ref: str, path: str) -> bool:
    result = subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{ref}:{path}"], capture_output=True)
    return result.returncode == 0


def read_file(root: Path, ref: str, path: str) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{path}"], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


# --- related changes ------------------------------------------------------


def change_names(paths: list[str], changes_dir: str) -> list[str]:
    prefix = changes_dir.rstrip("/") + "/"
    names: list[str] = []
    for path in paths:
        if not path.startswith(prefix):
            continue
        rest = path[len(prefix) :]
        parts = rest.split("/")
        if len(parts) < 2:
            continue
        if parts[0] == "archive":
            if len(parts) < 3:
                continue
            name = ARCHIVE_DATE.sub("", parts[1])
        else:
            name = parts[0]
        if name and name not in names:
            names.append(name)
    return sorted(names)


def archive_dir(root: Path, ref: str, changes_dir: str, name: str) -> str | None:
    matches = [e for e in tree_entries(root, ref, f"{changes_dir}/archive") if ARCHIVE_DATE.sub("", e) == name]
    return f"{changes_dir}/archive/{sorted(matches)[-1]}" if matches else None


def state_at(root: Path, ref: str, changes_dir: str, name: str) -> tuple[str, str | None]:
    active = f"{changes_dir}/{name}"
    if is_tree(root, ref, active):
        return "active", active
    archived = archive_dir(root, ref, changes_dir, name)
    if archived:
        return "archived", archived
    return "removed", None


def task_counts(text: str | None) -> dict:
    if text is None:
        return {"done": 0, "open": 0, "open_tasks": [], "progress": "not-started"}
    done = len(DONE_TASK.findall(text))
    open_tasks = [m.strip() for m in OPEN_TASK.findall(text)]
    if not open_tasks and done > 0:
        progress = "done"
    elif done == 0:
        progress = "not-started"
    else:
        progress = "in-progress"
    return {"done": done, "open": len(open_tasks), "open_tasks": open_tasks, "progress": progress}


def related_changes(root: Path, base: str, head: str, changes_dir: str) -> list[dict]:
    changes = []
    for name in change_names(diff_names(root, base, head), changes_dir):
        state, path = state_at(root, head, changes_dir, name)
        base_state, _ = state_at(root, base, changes_dir, name)
        tasks = task_counts(read_file(root, head, f"{path}/tasks.md")) if path else task_counts(None)
        marker = read_file(root, head, f"{path}/.openspec.yaml") if path else None
        changes.append(
            {
                "name": name,
                "state": state,
                "path": path,
                "at_base": base_state,
                "skip_specs": bool(marker and SKIP_SPECS.search(marker)),
                "tasks": tasks,
            }
        )
    return changes


def select(changes: list[dict], wanted: list[str]) -> list[dict]:
    if not wanted:
        return changes
    by_name = {c["name"]: c for c in changes}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise Failure(
            f"change(s) {', '.join(missing)} are not related to this diff; related: {', '.join(by_name) or '(none)'}."
        )
    return [by_name[w] for w in wanted]


# --- rendering ------------------------------------------------------------


def status_markdown(changes: list[dict]) -> str:
    if not changes:
        return "No related OpenSpec change: the diff touches nothing under the changes directory.\n"
    lines = ["| Change | State | Tasks | Progress |", "|---|---|---|---|"]
    for c in changes:
        t = c["tasks"]
        lines.append(f"| `{c['name']}` | {c['state']} | {t['done']} done, {t['open']} open | {t['progress']} |")
    for c in changes:
        if c["tasks"]["open_tasks"]:
            lines.append("")
            lines.append(f"Open tasks of `{c['name']}` (first ten):")
            for task in c["tasks"]["open_tasks"][:10]:
                lines.append(f"- {task[:117] + '...' if len(task) > 120 else task}")
    return "\n".join(lines) + "\n"


def fence_for(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def doc_paths(root: Path, head: str, change: dict, doc: str) -> list[str]:
    path = change["path"]
    if path is None:
        return []
    if doc == "specs":
        return [p for p in tree_files(root, head, f"{path}/specs") if p.endswith("/spec.md")]
    return [f"{path}/{doc}.md"]


def show_markdown(root: Path, head: str, changes: list[dict], doc: str, url_prefix: str, max_chars: int) -> str:
    docs = DOCS if doc == "all" else (doc,)
    out: list[str] = []
    used = 0
    overflow = False

    def emit(block: str) -> None:
        nonlocal used
        out.append(block)
        used += len(block)

    for change in changes:
        if change["path"] is None:
            emit(f"### `{change['name']}` — removed at the head; nothing to show.\n\n")
            continue
        for d in docs:
            paths = doc_paths(root, head, change, d)
            if not paths:
                emit(f"**{change['name']}/{d}.md** — _no {d}.md yet_\n\n")
                continue
            for p in paths:
                link = f"{url_prefix.rstrip('/')}/{p}" if url_prefix else p
                header = f"**{p}** ([view]({link}))\n\n" if url_prefix else f"**{p}**\n\n"
                if overflow:
                    emit(f"**{p}** — omitted for size ([view]({link}))\n\n")
                    continue
                text = read_file(root, head, p) or ""
                fence = fence_for(text)
                block = f"{header}{fence}markdown\n{text.rstrip()}\n{fence}\n\n"
                budget = max_chars - used
                if len(block) > budget:
                    overflow = True
                    keep = text[: max(0, budget - len(header) - len(fence) * 2 - 80)]
                    keep = keep[: keep.rfind("\n")] if "\n" in keep else keep
                    emit(f"{header}{fence}markdown\n{keep}\n{fence}\n… truncated — full file: {link}\n\n")
                    continue
                emit(block)
    return "".join(out) or "No related OpenSpec change: the diff touches nothing under the changes directory.\n"


# --- commands -------------------------------------------------------------


def run_openspec(root: Path, executable: str, *args: str) -> None:
    env = {**os.environ, "OPENSPEC_NO_UPDATE_CHECK": "1"}
    try:
        result = subprocess.run([executable, *args], cwd=root, env=env)
    except OSError as exc:
        raise Failure(f"cannot run `{executable}`: {exc}; install the pinned OpenSpec CLI first.") from exc
    if result.returncode != 0:
        raise Failure(f"`{executable} {' '.join(args)}` exited {result.returncode}.")


def cmd_related(args, root, changes_dir) -> int:
    changes = related_changes(root, args.base, args.head, changes_dir)
    if args.json:
        print(json.dumps({"base": args.base, "head": args.head, "changes": changes}, indent=2))
    else:
        for c in changes:
            print(f"{c['name']}\t{c['state']}\t{c['tasks']['done']}/{c['tasks']['open']}")
    return 0


def cmd_status(args, root, changes_dir) -> int:
    changes = select(related_changes(root, args.base, args.head, changes_dir), args.change)
    if args.json:
        print(json.dumps({"base": args.base, "head": args.head, "changes": changes}, indent=2))
    else:
        sys.stdout.write(status_markdown(changes))
    return 0


def cmd_show(args, root, changes_dir) -> int:
    changes = select(related_changes(root, args.base, args.head, changes_dir), [args.change] if args.change else [])
    sys.stdout.write(show_markdown(root, args.head, changes, args.doc, args.url_prefix, args.max_chars))
    return 0


def cmd_check(args, root, changes_dir) -> int:
    findings: list[str] = []
    if not args.no_validate:
        run_openspec(root, args.openspec, "validate", "--all", "--strict", "--no-interactive")
    if args.all:
        live = root / changes_dir
        if live.is_dir():
            for entry in sorted(p for p in live.iterdir() if p.is_dir() and p.name != "archive"):
                findings.append(f"unarchived change {entry.name}: the integration branch holds only archived changes.")
    else:
        for c in related_changes(root, args.base, args.head, changes_dir):
            if c["state"] == "active":
                message = (
                    f"unarchived change {c['name']}: {c['tasks']['open']} open task(s) — archive it in this "
                    f"pull request (by hand, or with the {TRIGGER_LABEL} label) before it is marked ready."
                )
                if args.draft:
                    print(f"warning: {message}")
                else:
                    findings.append(message)
            elif c["state"] == "archived" and c["tasks"]["open"]:
                print(f"warning: archived change {c['name']} still has {c['tasks']['open']} open task(s).")
    for finding in findings:
        print(finding)
    return 1 if findings else 0


def cmd_archive(args, root, changes_dir) -> int:
    head_sha = resolve(root, args.head, "--head")
    if git(root, "rev-parse", "HEAD").strip() != head_sha:
        raise Usage("archive edits the working tree, so --head must be the checked-out HEAD; check it out first.")
    if git(root, "status", "--porcelain", "--", changes_dir).strip():
        raise Usage(f"{changes_dir} has uncommitted changes; commit or stash them before archiving.")
    changes = related_changes(root, args.base, args.head, changes_dir)
    plan = {"archived": [], "skipped": [], "refused": [], "validated": False}
    todo: list[dict] = []
    for c in changes:
        if c["state"] != "active":
            plan["skipped"].append({"name": c["name"], "reason": c["state"]})
        elif c["tasks"]["open"] or c["tasks"]["done"] == 0:
            plan["refused"].append(
                {"name": c["name"], "open": c["tasks"]["open"], "tasks": c["tasks"]["open_tasks"][:5]}
                if c["tasks"]["open"]
                else {"name": c["name"], "open": 0, "tasks": ["no ticked task in tasks.md"]}
            )
        else:
            todo.append(c)
    if plan["refused"]:
        if args.json:
            print(json.dumps(plan, indent=2))
        for r in plan["refused"]:
            print(f"refused {r['name']}: {r['open']} open task(s) — {'; '.join(r['tasks'])}", file=sys.stderr)
        print("nothing archived: every related change must be complete first.", file=sys.stderr)
        return 1
    for c in todo:
        cmd = ["archive", c["name"], "--yes"] + (["--skip-specs"] if c["skip_specs"] else [])
        if args.dry_run:
            print(f"would run: {args.openspec} {' '.join(cmd)}")
            continue
        print(f"archiving {c['name']}", file=sys.stderr)
        run_openspec(root, args.openspec, *cmd)
        plan["archived"].append(c["name"])
    if todo and not args.dry_run:
        run_openspec(root, args.openspec, "validate", "--all", "--strict", "--no-interactive")
        plan["validated"] = True
    if args.json:
        print(json.dumps(plan, indent=2))
    elif not todo:
        print("nothing to archive: every related change is already archived or removed.")
    return 0


def desired_labels(changes: list[dict]) -> list[str]:
    live = [c for c in changes if c["state"] != "removed"]
    if not live:
        return []
    archived = ARCHIVED_LABELS[1] if all(c["state"] == "archived" for c in live) else ARCHIVED_LABELS[0]
    progress_set = {c["tasks"]["progress"] for c in live}
    if progress_set == {"done"}:
        progress = PROGRESS_LABELS[2]
    elif progress_set == {"not-started"}:
        progress = PROGRESS_LABELS[0]
    else:
        progress = PROGRESS_LABELS[1]
    return [archived, progress]


def cmd_labels(args, root, changes_dir) -> int:
    if args.taxonomy:
        print(json.dumps({"trigger": TRIGGER_LABEL, "managed": list(MANAGED_LABELS)}, indent=2))
        return 0
    if not (args.base and args.head):
        raise Usage("labels needs --base and --head (or --taxonomy).")
    desired = desired_labels(related_changes(root, args.base, args.head, changes_dir))
    current = [label.strip() for label in (args.current or "").split(",") if label.strip()]
    add = [label for label in desired if label not in current]
    remove = [label for label in current if label in MANAGED_LABELS and label not in desired]
    result = {"desired": desired, "add": add, "remove": remove}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"desired: {' '.join(desired) or '(none)'}")
        print(f"add: {' '.join(add) or '(none)'}")
        print(f"remove: {' '.join(remove) or '(none)'}")
    return 0


# --- argument parsing -----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spec_changes.py",
        description="Resolve, show, check, archive, and label the OpenSpec changes a pull or merge request touches.",
        epilog=(
            "Examples: python3 scripts/spec_changes.py status --base origin/main --head HEAD; "
            "python3 scripts/spec_changes.py check --base origin/main --head HEAD --draft; "
            "python3 scripts/spec_changes.py archive --base origin/main --head HEAD --json"
        ),
    )
    parser.add_argument("--root", default=".", help="git work tree holding the changes directory (default: .)")
    parser.add_argument("--changes-dir", default="openspec/changes", help="changes directory relative to --root")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def refs(p: argparse.ArgumentParser, required: bool = True) -> None:
        p.add_argument("--base", required=required, help="base ref (the target branch)")
        p.add_argument("--head", required=required, help="head ref (the request's tip)")

    p = sub.add_parser("related", help="list the related changes with their state and task counts")
    refs(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_related)

    p = sub.add_parser("status", help="print a progress table (markdown) for the related changes")
    refs(p)
    p.add_argument("--change", action="append", default=[], help="limit to this change (repeatable)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("show", help="print a change's documents inside fenced blocks")
    refs(p)
    p.add_argument("--change", help="one related change (default: every related change)")
    p.add_argument("--doc", choices=(*DOCS, "all"), default="all")
    p.add_argument("--url-prefix", default="", help="link prefix, e.g. https://github.com/o/r/blob/<sha>")
    p.add_argument("--max-chars", type=int, default=60000, help="truncate the output at this size (default 60000)")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("check", help="strict validation plus the unarchived-change rule")
    refs(p, required=False)
    p.add_argument("--all", action="store_true", help="fail on any change outside archive/ (integration branch)")
    p.add_argument("--draft", action="store_true", help="report unarchived related changes as warnings")
    p.add_argument("--no-validate", action="store_true", help="skip the OpenSpec validator")
    p.add_argument("--openspec", default="openspec", help="OpenSpec CLI executable (default: openspec)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("archive", help="archive every complete related change through the OpenSpec CLI")
    refs(p)
    p.add_argument("--dry-run", action="store_true", help="print the plan; change nothing")
    p.add_argument("--openspec", default="openspec", help="OpenSpec CLI executable (default: openspec)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_archive)

    p = sub.add_parser("labels", help="compute the archive-axis and progress-axis labels")
    refs(p, required=False)
    p.add_argument("--current", default="", help="comma-separated labels currently on the request")
    p.add_argument("--taxonomy", action="store_true", help="print the trigger and managed label names")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_labels)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments and 0 on --help
        return int(exc.code or 0)
    root = Path(args.root).resolve()
    changes_dir = args.changes_dir.strip("/")
    try:
        if not (root / ".git").exists() and git(root, "rev-parse", "--git-dir", check=False).strip() == "":
            raise Usage(f"{root} is not a git work tree.")
        if args.command == "check" and not args.all and not (args.base and args.head):
            raise Usage("check needs --base and --head, or --all.")
        if getattr(args, "base", None) and getattr(args, "head", None) and args.command != "archive":
            resolve(root, args.base, "--base")
            resolve(root, args.head, "--head")
        return args.func(args, root, changes_dir)
    except Usage as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Failure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
