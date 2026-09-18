#!/usr/bin/env python3
"""Operate on the Spec-Kit features a pull or merge request touches.

A request's *touched features* are the numbered feature directories under
the kit's specs directory (default ``specs/``, entries named
``<NNN>-<name>``) whose files the request touches. Layout verified against
Spec-Kit's templates and feature script on 2026-09-17: ``spec.md`` and
``plan.md`` are required, ``tasks.md`` lists tasks as ``- [ ] T001 ...``
checkboxes, and the feature script creates the directory and no git
branch. The kit ships no validator and no archive operation: completion is
every task ticked.

The head is read through one of two sources, never through a checkout:
``--base``/``--head`` with git plumbing (``GitSource``), or
``--snapshot FILE`` built from the GitHub REST API (``SnapshotSource``).
Each class says where it belongs. ``check --all`` is the one command that
reads the working tree instead. No command writes.

Exit codes: 0 success; 1 a finding or a failure; 2 bad arguments, an
unresolvable ref, or a tree that is not a git repository.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OPEN_TASK = re.compile(r"^\s*-\s*\[ \]\s*(.*)$", re.MULTILINE)
DONE_TASK = re.compile(r"^\s*-\s*\[[xX]\]\s", re.MULTILINE)
FEATURE_DIR = re.compile(r"^\d{3,}-[A-Za-z0-9._-]+$")
DOCS = ("spec", "plan", "tasks")
REQUIRED = ("spec.md", "plan.md")
PROGRESS_LABELS = ("spec/not-started", "spec/in-progress", "spec/done")

SNAPSHOT_SCHEMA = "spec-kit-snapshot/1"
SAFE_REPO = re.compile(r"^[A-Za-z0-9._-]{1,100}/[A-Za-z0-9._-]{1,100}$")
MAX_FILES = 3000
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 8_000_000
# The platform token's REST budget is shared by every workflow of the repository.
MAX_CALLS = 200


class Usage(Exception):
    """Bad arguments or an unusable repository (exit 2)."""


class Failure(Exception):
    """A finding or a failure (exit 1)."""


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
            f"{what} {ref!r} does not resolve to a commit in {root}; fetch it first, or read the head "
            "through `snapshot` instead of fetching it into a privileged workflow."
        )
    return result.stdout.strip()


# --- sources --------------------------------------------------------------


class Source:
    """A read-only view of the head."""

    base_label = "base"
    head_label = "head"

    def changed_paths(self) -> list[str]:
        raise NotImplementedError

    def entries(self, path: str) -> list[str]:
        raise NotImplementedError

    def read(self, path: str) -> str | None:
        raise NotImplementedError


class GitSource(Source):
    """Reads the two commits out of a local git object store.

    Use it where the head is already trusted or already present: a
    developer's clone, or an unprivileged pull-request check that checks
    the head out the ordinary way.
    """

    def __init__(self, root: Path, base: str, head: str) -> None:
        self.root = root
        self.base = base
        self.head = head
        self.base_label = base
        self.head_label = head

    def changed_paths(self) -> list[str]:
        result = subprocess.run(
            ["git", "-C", str(self.root), "diff", "--name-only", "--no-renames", f"{self.base}...{self.head}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Usage(
                f"`git diff {self.base}...{self.head}` failed ({result.stderr.strip()}); the merge base must be "
                "reachable — fetch with full history (fetch-depth 0) rather than a shallow clone."
            )
        return [line for line in result.stdout.splitlines() if line]

    def entries(self, path: str) -> list[str]:
        out = git(self.root, "ls-tree", "--name-only", self.head, f"{path.rstrip('/')}/", check=False)
        return [line.rsplit("/", 1)[-1] for line in out.splitlines() if line]

    def read(self, path: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(self.root), "show", f"{self.head}:{path}"], capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else None


class SnapshotSource(Source):
    """Reads a snapshot document built from the GitHub REST API.

    Use it in every privileged workflow (``pull_request_target``,
    ``issue_comment``, ``workflow_run``): the head's bytes arrive as data
    to parse, so no object authored by the request ever reaches the
    runner's git store, and nothing it carries is ever executed.
    """

    def __init__(self, doc: dict, specs_dir: str | None = None) -> None:
        if doc.get("schema") != SNAPSHOT_SCHEMA:
            raise Usage(f"snapshot schema {doc.get('schema')!r} is not {SNAPSHOT_SCHEMA!r}; rebuild it.")
        recorded = doc.get("specs_dir")
        if specs_dir is not None and recorded is not None and recorded != specs_dir:
            raise Usage(
                f"the snapshot was built for the specs directory {recorded!r}, not {specs_dir!r}; "
                "rebuild it with the same --specs-dir."
            )
        self.doc = doc
        self.head = doc.get("head") or {}
        self.base_label = (doc.get("base") or {}).get("sha", "base")
        self.head_label = self.head.get("sha", "head")

    def changed_paths(self) -> list[str]:
        return list(self.doc.get("changed_paths") or [])

    def entries(self, path: str) -> list[str]:
        prefix = path.rstrip("/") + "/"
        names: list[str] = []
        for item in list(self.head.get("dirs") or []) + list(self.head.get("paths") or []):
            if item.startswith(prefix):
                name = item[len(prefix) :].split("/")[0]
                if name and name not in names:
                    names.append(name)
        return names

    def read(self, path: str) -> str | None:
        return (self.head.get("files") or {}).get(path)


# --- github rest api ------------------------------------------------------


class Api:
    """The few REST reads the snapshot needs, over the standard library.

    A fork's head SHA resolves from the base repository, so nothing here
    needs the fork. ``max_calls`` bounds what one request can spend of the
    repository's shared REST budget.
    """

    def __init__(self, repo: str, api_url: str, auth: str, max_calls: int = MAX_CALLS) -> None:
        if not SAFE_REPO.match(repo):
            raise Usage(f"--repo {repo!r} is not an OWNER/NAME pair.")
        self.repo = repo
        self.api_url = api_url.rstrip("/")
        self.auth = auth
        self.max_calls = max_calls
        self.calls = 0

    def get(self, path: str, allow_404: bool = False):
        url = f"{self.api_url}/{path.lstrip('/')}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.auth}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "spec-kit-features",
            },
        )
        for attempt in range(3):
            if self.calls >= self.max_calls:
                raise Failure(
                    f"the snapshot reached the --max-calls cap ({self.max_calls} API requests); nothing is "
                    "reported from a partial snapshot. Split the request, or raise the cap deliberately."
                )
            try:
                self.calls += 1
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and allow_404:
                    return None
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                detail = exc.read().decode("utf-8", "replace")[:200].replace("\n", " ")
                raise Failure(
                    f"GitHub API {exc.code} for {path}: {detail}; check the token's permissions "
                    "(a read needs `contents: read` and `pull-requests: read`)."
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise Failure(f"cannot reach the GitHub API at {url}: {exc.reason}") from exc
        raise Failure(f"GitHub API kept failing for {path}.")

    def pull(self, number: int) -> dict:
        data = self.get(f"repos/{self.repo}/pulls/{number}")
        if not isinstance(data, dict):
            raise Failure(f"pull request {number} did not return an object.")
        return data

    def pull_files(self, number: int, max_files: int) -> list[str]:
        paths: list[str] = []
        page = 1
        while True:
            batch = self.get(f"repos/{self.repo}/pulls/{number}/files?per_page=100&page={page}")
            if not batch:
                break
            for item in batch:
                if "filename" in item:
                    paths.append(item["filename"])
                # `git diff --no-renames` reports a rename as its old path plus
                # its new one, so the git source sees both. The API reports it as
                # one entry carrying `previous_filename`; keeping only `filename`
                # would hide a record moved out of its directory from every
                # privileged job while the unprivileged check still saw it.
                if item.get("previous_filename"):
                    paths.append(item["previous_filename"])
            if len(batch) < 100 or len(paths) > max_files:
                break
            page += 1
        if len(paths) > max_files:
            raise Failure(
                f"the request touches at least {len(paths)} files, over the --max-files cap "
                f"({max_files}); split the request, or raise the cap deliberately."
            )
        return paths

    def tree(self, sha: str, recursive: bool = False) -> dict | None:
        query = "?recursive=1" if recursive else ""
        data = self.get(f"repos/{self.repo}/git/trees/{urllib.parse.quote(sha, safe='')}{query}", allow_404=True)
        if data and data.get("truncated"):
            raise Failure(
                f"the tree {sha} came back truncated; nothing is reported from a partial snapshot. "
                "Narrow the specs directory."
            )
        return data

    def blob_text(self, sha: str) -> str | None:
        data = self.get(f"repos/{self.repo}/git/blobs/{urllib.parse.quote(sha, safe='')}", allow_404=True)
        if not data or data.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(data["content"]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None


def tree_sha_at(api: Api, commit: str, segments: list[str]) -> str | None:
    """Walk a commit's tree down a path, one tree read per segment.

    Returns None when a path segment is absent — a project that does not
    use the tool looks exactly like that, so it is not an error. The
    commit's own tree is different: a request whose head cannot be read
    (a force-push or a deleted fork mid-run) would otherwise be reported
    as a project with nothing in it, and a label derived from that would
    say the opposite of the truth.
    """
    root = api.tree(commit)
    if root is None:
        raise Failure(
            f"the head commit {commit} is not readable; nothing is reported from a partial snapshot. "
            "The branch was probably force-pushed or deleted while this ran — retry on the current head."
        )
    sha = commit
    for segment in segments:
        data = api.tree(sha) if sha != commit else root
        if not data:
            return None
        entry = next((e for e in data.get("tree", []) if e.get("path") == segment and e.get("type") == "tree"), None)
        if entry is None:
            return None
        sha = entry["sha"]
    return sha


def build_snapshot(api: Api, number: int, specs_dir: str, caps: dict) -> dict:
    """Read the request's touched features into a document the sources can replay.

    Only the documents the commands read are fetched, feature names are
    checked against the numbered-directory pattern before they reach a
    URL, and a snapshot that would be partial — a cap reached, a truncated
    tree — raises instead of being reported on: a label derived from half
    the request is worse than no answer.
    """
    pull = api.pull(number)
    base_sha = pull["base"]["sha"]
    head_sha = pull["head"]["sha"]
    changed = api.pull_files(number, caps["max_files"])

    dirs: list[str] = []
    shas: dict[str, str] = {}
    root = tree_sha_at(api, head_sha, specs_dir.split("/"))
    if root:
        for entry in (api.tree(root) or {}).get("tree", []):
            if entry.get("type") == "tree":
                dirs.append(f"{specs_dir}/{entry['path']}")
                shas[f"{specs_dir}/{entry['path']}"] = entry["sha"]

    skipped: list[dict] = []
    paths: list[str] = []
    files: dict[str, str] = {}
    total = 0
    for name in feature_names(changed, specs_dir):
        path = f"{specs_dir}/{name}"
        if path not in shas:
            continue
        listing = api.tree(shas[path], recursive=True) or {"tree": []}
        for entry in listing.get("tree", []):
            if entry.get("type") != "blob":
                continue
            full = f"{path}/{entry['path']}"
            paths.append(full)
            # Only the documents the commands read are fetched; the rest are listed by path.
            if entry["path"] not in {f"{d}.md" for d in DOCS}:
                continue
            size = int(entry.get("size") or 0)
            if size > caps["max_file_bytes"]:
                raise Failure(
                    f"{full} is {size} bytes, over the --max-file-bytes cap ({caps['max_file_bytes']}); "
                    "nothing is reported from a partial snapshot."
                )
            if total + size > caps["max_total_bytes"]:
                raise Failure(
                    f"the documents reached the --max-total-bytes cap ({caps['max_total_bytes']}) at {full}; "
                    "nothing is reported from a partial snapshot."
                )
            text = api.blob_text(entry["sha"])
            if text is None:
                raise Failure(f"{full} is not decodable as UTF-8 text; nothing is reported from a partial snapshot.")
            files[full] = text
            total += size

    return {
        "schema": SNAPSHOT_SCHEMA,
        "repo": api.repo,
        "pull_request": number,
        "specs_dir": specs_dir,
        "changed_paths": changed,
        "base": {"sha": base_sha},
        "head": {"sha": head_sha, "dirs": dirs, "paths": sorted(paths), "files": files},
        "skipped": skipped,
        "bytes": total,
    }


# --- touched features -----------------------------------------------------


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
    open_tasks = [m.strip() or "(untitled task)" for m in OPEN_TASK.findall(text)]
    if not open_tasks and done > 0:
        progress = "done"
    elif done == 0:
        progress = "not-started"
    else:
        progress = "in-progress"
    return {"done": done, "open": len(open_tasks), "open_tasks": open_tasks, "progress": progress}


def describe(source: Source, specs_dir: str, name: str) -> dict:
    path = f"{specs_dir}/{name}"
    present = set(source.entries(path))
    return {
        "name": name,
        "path": path,
        "state": "present" if present else "removed",
        "missing": [f for f in REQUIRED if f not in present] if present else [],
        "tasks": task_counts(source.read(f"{path}/tasks.md")) if present else task_counts(None),
    }


def touched_features(source: Source, specs_dir: str) -> list[dict]:
    return [describe(source, specs_dir, n) for n in feature_names(source.changed_paths(), specs_dir)]


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


def code_span(text: str, in_table: bool = False) -> str:
    """Inline code that request-authored text cannot break out of (no links, mentions, or markup)."""
    text = " ".join(text.split())
    if in_table:  # a table splits cells before it parses code spans
        text = text.replace("|", "\\|")
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * (longest + 1)
    pad = " " if not text or text.startswith("`") or text.endswith("`") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


def link_to(url_prefix: str, path: str) -> str:
    return f"{url_prefix.rstrip('/')}/{urllib.parse.quote(path, safe='/')}"


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
        name = code_span(f["name"], in_table=True)
        lines.append(f"| {name} | {files} | {t['done']} done, {t['open']} open | {t['progress']} |")
    for f in features:
        if f["tasks"]["open_tasks"]:
            lines.append("")
            lines.append(f"Open tasks of {code_span(f['name'])} (first ten):")
            for task in f["tasks"]["open_tasks"][:10]:
                lines.append(f"- {code_span(task[:117] + '...' if len(task) > 120 else task)}")
    return "\n".join(lines) + "\n"


def fence_for(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def show_markdown(source: Source, features: list[dict], doc: str, url_prefix: str, max_chars: int) -> str:
    docs = DOCS if doc == "all" else (doc,)
    out: list[str] = []
    used = 0
    overflow = False
    for f in features:
        if f["state"] == "removed":
            block = f"### {code_span(f['name'])} — removed at the head; nothing to show.\n\n"
            out.append(block)
            used += len(block)
            continue
        for d in docs:
            p = f"{f['path']}/{d}.md"
            link = link_to(url_prefix, p) if url_prefix else ""
            view = f" ([view]({link}))" if url_prefix else ""
            header = f"**{code_span(p)}**{view}\n\n"
            text = source.read(p)
            if text is None:
                block = f"**{code_span(p)}** — _no {d}.md yet_\n\n"
            elif overflow:
                block = f"**{code_span(p)}** — omitted for size{view}\n\n"
            else:
                fence = fence_for(text)
                block = f"{header}{fence}markdown\n{text.rstrip()}\n{fence}\n\n"
                if len(block) > max_chars - used:
                    overflow = True
                    keep = text[: max(0, max_chars - used - len(header) - len(fence) * 2 - 80)]
                    keep = keep[: keep.rfind("\n")] if "\n" in keep else keep
                    tail = f": {link}" if url_prefix else ""
                    block = f"{header}{fence}markdown\n{keep}\n{fence}\n… truncated — full file{tail}\n\n"
            out.append(block)
            used += len(block)
    return "".join(out) or "No touched Spec-Kit feature: the diff touches nothing under the specs directory.\n"


# --- commands -------------------------------------------------------------


def cmd_snapshot(args, root, specs_dir) -> int:
    auth = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not auth:
        raise Usage("snapshot needs a GitHub token in GH_TOKEN or GITHUB_TOKEN.")
    if args.pr <= 0:
        raise Usage("--pr must be a positive pull request number.")
    if min(args.max_files, args.max_file_bytes, args.max_total_bytes, args.max_calls) <= 0:
        raise Usage("every snapshot cap must be a positive number.")
    api = Api(args.repo, args.api_url, auth, args.max_calls)
    caps = {
        "max_files": args.max_files,
        "max_file_bytes": args.max_file_bytes,
        "max_total_bytes": args.max_total_bytes,
    }
    doc = build_snapshot(api, args.pr, specs_dir, caps)
    text = json.dumps(doc, indent=2)
    if args.out:
        try:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            raise Usage(f"cannot write the snapshot to {args.out}: {exc}") from exc
        print(
            f"snapshot of {api.repo}#{args.pr} at {doc['head']['sha'][:7]}: "
            f"{len(doc['head']['files'])} file(s), {doc['bytes']} byte(s), {api.calls} API call(s) → {args.out}",
            file=sys.stderr,
        )
    else:
        print(text)
    for item in doc["skipped"]:
        print(f"skipped {item.get('path') or item.get('name')}: {item['reason']}", file=sys.stderr)
    return 0


def cmd_related(args, root, specs_dir) -> int:
    source = args.source
    features = touched_features(source, specs_dir)
    if args.json:
        print(json.dumps({"base": source.base_label, "head": source.head_label, "features": features}, indent=2))
    else:
        for f in features:
            print(f"{f['name']}\t{f['state']}\t{f['tasks']['done']}/{f['tasks']['open']}")
    return 0


def cmd_status(args, root, specs_dir) -> int:
    source = args.source
    features = select(touched_features(source, specs_dir), args.feature)
    if args.json:
        print(json.dumps({"base": source.base_label, "head": source.head_label, "features": features}, indent=2))
    else:
        sys.stdout.write(status_markdown(features))
    return 0


def cmd_show(args, root, specs_dir) -> int:
    source = args.source
    features = select(touched_features(source, specs_dir), [args.feature] if args.feature else [])
    sys.stdout.write(show_markdown(source, features, args.doc, args.url_prefix, args.max_chars))
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
        for f in touched_features(args.source, specs_dir):
            if f["state"] == "removed":
                continue
            for m in f["missing"]:
                findings.append(f"feature {f['name']}: missing {m} — the specification and the plan form the package.")
            # "Finished" is every task ticked, so a feature with no ticked task
            # is unfinished even when it has no open one: a missing tasks.md and
            # a tasks.md with no checkbox both mean nothing was implemented, and
            # the progress label already reads not-started for them.
            if f["tasks"]["open"] or f["tasks"]["done"] == 0:
                if f["tasks"]["open"]:
                    message = (
                        f"feature {f['name']}: {f['tasks']['open']} open task(s) — every task is ticked before the "
                        "pull request is marked ready."
                    )
                else:
                    message = (
                        f"feature {f['name']}: no completed task — tasks.md is missing or carries no checkbox, so "
                        "the request implemented nothing and is not finished."
                    )
                # The one request that may merge unfinished is the split shape's
                # specification request: it carries the specification and the plan
                # and no implementation, and the implementation requests that
                # follow tick the tasks. A request that implemented something is
                # not that request, whatever its shape.
                if args.draft:
                    print(f"warning: {message}")
                elif args.shape == "split" and f["tasks"]["done"] == 0:
                    print(
                        f"warning: feature {f['name']} has no implemented task; "
                        "allowed for a specification request under the split shape."
                    )
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
    desired = desired_labels(touched_features(args.source, specs_dir))
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


# --- argument parsing -----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spec_kit_features.py",
        description="Resolve, show, check, and label the Spec-Kit features a pull or merge request touches.",
        epilog=(
            "Examples: python3 scripts/spec_kit_features.py status --base origin/main --head HEAD; "
            "python3 scripts/spec_kit_features.py snapshot --repo owner/name --pr 12 --out snapshot.json; "
            "python3 scripts/spec_kit_features.py labels --snapshot snapshot.json"
        ),
    )
    parser.add_argument("--root", default=".", help="git work tree holding the specs directory (default: .)")
    parser.add_argument("--specs-dir", default="specs", help="the kit's specs directory relative to --root")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def reads(p: argparse.ArgumentParser) -> None:
        group = p.add_argument_group("head source (one of)")
        group.add_argument("--base", help="base ref, read with git plumbing (with --head)")
        group.add_argument("--head", help="head ref, read with git plumbing (with --base)")
        group.add_argument(
            "--snapshot",
            help="snapshot file from the snapshot command; the only source a privileged workflow may use",
        )

    p = sub.add_parser("snapshot", help="build a head snapshot from the GitHub REST API (no checkout, no fetch)")
    p.add_argument("--repo", required=True, help="OWNER/NAME of the repository the pull request targets")
    p.add_argument("--pr", required=True, type=int, help="pull request number")
    p.add_argument("--out", help="write the snapshot here (default: stdout)")
    p.add_argument("--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    p.add_argument("--max-files", type=int, default=MAX_FILES, help=f"cap on touched files (default {MAX_FILES})")
    p.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES, help="per-file byte cap")
    p.add_argument("--max-total-bytes", type=int, default=MAX_TOTAL_BYTES, help="total byte cap")
    p.add_argument("--max-calls", type=int, default=MAX_CALLS, help=f"cap on API requests (default {MAX_CALLS})")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("related", help="list the touched features with their files and task counts")
    reads(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_related)

    p = sub.add_parser("status", help="print a progress table (markdown) for the touched features")
    reads(p)
    p.add_argument("--feature", action="append", default=[], help="limit to this feature (repeatable)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("show", help="print a feature's documents inside fenced blocks")
    reads(p)
    p.add_argument("--feature", help="one touched feature (default: every touched feature)")
    p.add_argument("--doc", choices=(*DOCS, "all"), default="all")
    p.add_argument("--url-prefix", default="", help="link prefix, e.g. https://github.com/o/r/blob/<sha>")
    p.add_argument("--max-chars", type=int, default=60000, help="truncate the output at this size (default 60000)")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("check", help="required files present; open tasks warn on a draft and fail when ready")
    reads(p)
    p.add_argument("--all", action="store_true", help="check every feature's required files in the working tree")
    p.add_argument("--draft", action="store_true", help="report open tasks as warnings")
    p.add_argument(
        "--shape",
        choices=("combined", "split"),
        default="combined",
        help=(
            "the change request shape the project's contract records (default: combined). "
            "Under split, a touched feature with no implemented task is allowed unfinished: "
            "that is the specification request, which the implementation requests finish later."
        ),
    )
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("labels", help="compute the progress label")
    reads(p)
    p.add_argument("--current", default="", help="comma-separated labels currently on the request")
    p.add_argument("--taxonomy", action="store_true", help="print the managed label names")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_labels)
    return parser


def open_source(args, root: Path, specs_dir: str) -> Source | None:
    """Build the head source the command asked for, or None when it needs none."""
    snapshot = getattr(args, "snapshot", None)
    base, head = getattr(args, "base", None), getattr(args, "head", None)
    if snapshot and (base or head):
        raise Usage("--snapshot reads the head on its own; do not pass --base/--head with it.")
    if snapshot:
        try:
            doc = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        except OSError as exc:
            raise Usage(f"cannot read the snapshot {snapshot}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise Usage(f"snapshot {snapshot} is not valid JSON: {exc}") from exc
        return SnapshotSource(doc, specs_dir)
    if base and head:
        if git(root, "rev-parse", "--git-dir", check=False).strip() == "":
            raise Usage(f"{root} is not a git work tree.")
        resolve(root, base, "--base")
        resolve(root, head, "--head")
        return GitSource(root, base, head)
    if base or head:
        raise Usage("--base and --head go together; pass both, or pass --snapshot instead.")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments and 0 on --help
        return int(exc.code or 0)
    root = Path(args.root).resolve()
    specs_dir = args.specs_dir.strip("/")
    try:
        if args.command == "snapshot":
            return args.func(args, root, specs_dir)
        args.source = open_source(args, root, specs_dir)
        needs_no_source = (args.command == "check" and args.all) or (args.command == "labels" and args.taxonomy)
        if args.source is None and not needs_no_source:
            raise Usage(
                f"{args.command} needs a head source: --base with --head, or --snapshot from the snapshot command."
            )
        if args.command == "check" and args.all and args.source is not None:
            raise Usage("check --all reads the working tree; do not pass a head source with it.")
        return args.func(args, root, specs_dir)
    except Usage as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Failure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
