#!/usr/bin/env python3
"""Operate on the Spec-Kit features a pull or merge request touches.

A request's *touched features* are the numbered feature directories under
the kit's specs directory (default ``specs/``, entries named
``<NNN>-<name>``) whose files the diff between ``--base`` and ``--head``
touches. Every subcommand reads the head through git plumbing, so the head
never needs a checkout. Layout verified against Spec-Kit's templates and
feature script on 2026-09-17: ``spec.md`` and ``plan.md`` are required,
``tasks.md`` lists tasks as ``- [ ] T001 ...`` checkboxes, and the feature
script creates the directory and no git branch. The kit ships no validator
and no archive operation: completion is every task ticked.

Exit codes: 0 success; 1 a finding or a failure; 2 bad arguments, an
unresolvable ref, or a tree that is not a git repository.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

OPEN_TASK = re.compile(r"^\s*-\s*\[ \]\s+(.*)$", re.MULTILINE)
DONE_TASK = re.compile(r"^\s*-\s*\[[xX]\]\s", re.MULTILINE)
FEATURE_DIR = re.compile(r"^\d{3,}-[A-Za-z0-9._-]+$")
DOCS = ("spec", "plan", "tasks")
REQUIRED = ("spec.md", "plan.md")
PROGRESS_LABELS = ("spec/not-started", "spec/in-progress", "spec/done")


class Usage(Exception):
    """Bad arguments or an unusable repository (exit 2)."""


class Failure(Exception):
    """A finding or a failure (exit 1)."""


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


def read_file(root: Path, ref: str, path: str) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{path}"], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def feature_names(paths: list[str], specs_dir: str) -> list[str]:
    prefix = specs_dir.rstrip("/") + "/"
    names: list[str] = []
    for path in paths:
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix) :].split("/")
        if len(parts) >= 2 and FEATURE_DIR.match(parts[0]) and parts[0] not in names:
            names.append(parts[0])
    return sorted(names)


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


def describe(root: Path, ref: str, specs_dir: str, name: str) -> dict:
    path = f"{specs_dir}/{name}"
    present = set(tree_entries(root, ref, path))
    return {
        "name": name,
        "path": path,
        "state": "present" if present else "removed",
        "missing": [f for f in REQUIRED if f not in present] if present else [],
        "tasks": task_counts(read_file(root, ref, f"{path}/tasks.md")) if present else task_counts(None),
    }


def touched_features(root: Path, base: str, head: str, specs_dir: str) -> list[dict]:
    return [describe(root, head, specs_dir, n) for n in feature_names(diff_names(root, base, head), specs_dir)]


def select(features: list[dict], wanted: list[str]) -> list[dict]:
    if not wanted:
        return features
    by_name = {f["name"]: f for f in features}
    missing = [w for w in wanted if w not in by_name]
    if missing:
        raise Failure(
            f"feature(s) {', '.join(missing)} are not touched by this diff; touched: {', '.join(by_name) or '(none)'}."
        )
    return [by_name[w] for w in wanted]


def status_markdown(features: list[dict]) -> str:
    if not features:
        return "No touched Spec-Kit feature: the diff touches nothing under the specs directory.\n"
    lines = ["| Feature | Files | Tasks | Progress |", "|---|---|---|---|"]
    for f in features:
        if f["state"] == "removed":
            files = "removed"
        else:
            files = "missing " + ", ".join(f["missing"]) if f["missing"] else "complete"
        t = f["tasks"]
        lines.append(f"| `{f['name']}` | {files} | {t['done']} done, {t['open']} open | {t['progress']} |")
    for f in features:
        if f["tasks"]["open_tasks"]:
            lines.append("")
            lines.append(f"Open tasks of `{f['name']}` (first ten):")
            for task in f["tasks"]["open_tasks"][:10]:
                lines.append(f"- {task[:117] + '...' if len(task) > 120 else task}")
    return "\n".join(lines) + "\n"


def fence_for(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def show_markdown(root: Path, head: str, features: list[dict], doc: str, url_prefix: str, max_chars: int) -> str:
    docs = DOCS if doc == "all" else (doc,)
    out: list[str] = []
    used = 0
    overflow = False
    for f in features:
        if f["state"] == "removed":
            block = f"### `{f['name']}` — removed at the head; nothing to show.\n\n"
            out.append(block)
            used += len(block)
            continue
        for d in docs:
            p = f"{f['path']}/{d}.md"
            link = f"{url_prefix.rstrip('/')}/{p}" if url_prefix else p
            header = f"**{p}** ([view]({link}))\n\n" if url_prefix else f"**{p}**\n\n"
            text = read_file(root, head, p)
            if text is None:
                block = f"**{p}** — _no {d}.md yet_\n\n"
            elif overflow:
                block = f"**{p}** — omitted for size ([view]({link}))\n\n"
            else:
                fence = fence_for(text)
                block = f"{header}{fence}markdown\n{text.rstrip()}\n{fence}\n\n"
                if len(block) > max_chars - used:
                    overflow = True
                    keep = text[: max(0, max_chars - used - len(header) - len(fence) * 2 - 80)]
                    keep = keep[: keep.rfind("\n")] if "\n" in keep else keep
                    block = f"{header}{fence}markdown\n{keep}\n{fence}\n… truncated — full file: {link}\n\n"
            out.append(block)
            used += len(block)
    return "".join(out) or "No touched Spec-Kit feature: the diff touches nothing under the specs directory.\n"


def cmd_related(args, root, specs_dir) -> int:
    features = touched_features(root, args.base, args.head, specs_dir)
    if args.json:
        print(json.dumps({"base": args.base, "head": args.head, "features": features}, indent=2))
    else:
        for f in features:
            print(f"{f['name']}\t{f['state']}\t{f['tasks']['done']}/{f['tasks']['open']}")
    return 0


def cmd_status(args, root, specs_dir) -> int:
    features = select(touched_features(root, args.base, args.head, specs_dir), args.feature)
    if args.json:
        print(json.dumps({"base": args.base, "head": args.head, "features": features}, indent=2))
    else:
        sys.stdout.write(status_markdown(features))
    return 0


def cmd_show(args, root, specs_dir) -> int:
    features = select(touched_features(root, args.base, args.head, specs_dir), [args.feature] if args.feature else [])
    sys.stdout.write(show_markdown(root, args.head, features, args.doc, args.url_prefix, args.max_chars))
    return 0


def cmd_check(args, root, specs_dir) -> int:
    findings: list[str] = []
    if args.all:
        live = root / specs_dir
        if live.is_dir():
            for entry in sorted(p for p in live.iterdir() if p.is_dir() and FEATURE_DIR.match(p.name)):
                for required in REQUIRED:
                    if not (entry / required).is_file():
                        findings.append(f"feature {entry.name}: missing {required}.")
    else:
        for f in touched_features(root, args.base, args.head, specs_dir):
            if f["state"] == "removed":
                continue
            for m in f["missing"]:
                findings.append(f"feature {f['name']}: missing {m} — the specification and the plan form the package.")
            if f["tasks"]["open"]:
                message = (
                    f"feature {f['name']}: {f['tasks']['open']} open task(s) — every task is ticked before the "
                    "pull request is marked ready."
                )
                if args.draft:
                    print(f"warning: {message}")
                else:
                    findings.append(message)
    for finding in findings:
        print(finding)
    return 1 if findings else 0


def desired_labels(features: list[dict]) -> list[str]:
    live = [f for f in features if f["state"] != "removed"]
    if not live:
        return []
    progress_set = {f["tasks"]["progress"] for f in live}
    if progress_set == {"done"}:
        return [PROGRESS_LABELS[2]]
    if progress_set == {"not-started"}:
        return [PROGRESS_LABELS[0]]
    return [PROGRESS_LABELS[1]]


def cmd_labels(args, root, specs_dir) -> int:
    if args.taxonomy:
        print(json.dumps({"managed": list(PROGRESS_LABELS)}, indent=2))
        return 0
    if not (args.base and args.head):
        raise Usage("labels needs --base and --head (or --taxonomy).")
    desired = desired_labels(touched_features(root, args.base, args.head, specs_dir))
    current = [label.strip() for label in (args.current or "").split(",") if label.strip()]
    result = {
        "desired": desired,
        "add": [label for label in desired if label not in current],
        "remove": [label for label in current if label in PROGRESS_LABELS and label not in desired],
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {' '.join(value) or '(none)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spec_kit_features.py",
        description="Resolve, show, check, and label the Spec-Kit features a pull or merge request touches.",
        epilog=(
            "Examples: python3 scripts/spec_kit_features.py status --base origin/main --head HEAD; "
            "python3 scripts/spec_kit_features.py check --base origin/main --head HEAD --draft"
        ),
    )
    parser.add_argument("--root", default=".", help="git work tree holding the specs directory (default: .)")
    parser.add_argument("--specs-dir", default="specs", help="the kit's specs directory relative to --root")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def refs(p: argparse.ArgumentParser, required: bool = True) -> None:
        p.add_argument("--base", required=required, help="base ref (the target branch)")
        p.add_argument("--head", required=required, help="head ref (the request's tip)")

    p = sub.add_parser("related", help="list the touched features with their files and task counts")
    refs(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_related)

    p = sub.add_parser("status", help="print a progress table (markdown) for the touched features")
    refs(p)
    p.add_argument("--feature", action="append", default=[], help="limit to this feature (repeatable)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("show", help="print a feature's documents inside fenced blocks")
    refs(p)
    p.add_argument("--feature", help="one touched feature (default: every touched feature)")
    p.add_argument("--doc", choices=(*DOCS, "all"), default="all")
    p.add_argument("--url-prefix", default="", help="link prefix, e.g. https://github.com/o/r/blob/<sha>")
    p.add_argument("--max-chars", type=int, default=60000, help="truncate the output at this size (default 60000)")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("check", help="required files present; open tasks warn on a draft and fail when ready")
    refs(p, required=False)
    p.add_argument("--all", action="store_true", help="check every feature's required files in the working tree")
    p.add_argument("--draft", action="store_true", help="report open tasks as warnings")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("labels", help="compute the progress label")
    refs(p, required=False)
    p.add_argument("--current", default="", help="comma-separated labels currently on the request")
    p.add_argument("--taxonomy", action="store_true", help="print the managed label names")
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
    specs_dir = args.specs_dir.strip("/")
    try:
        if git(root, "rev-parse", "--git-dir", check=False).strip() == "":
            raise Usage(f"{root} is not a git work tree.")
        if args.command == "check" and not args.all and not (args.base and args.head):
            raise Usage("check needs --base and --head, or --all.")
        if getattr(args, "base", None) and getattr(args, "head", None):
            resolve(root, args.base, "--base")
            resolve(root, args.head, "--head")
        return args.func(args, root, specs_dir)
    except Usage as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Failure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
