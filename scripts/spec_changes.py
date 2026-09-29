#!/usr/bin/env python3
"""Operate on the OpenSpec changes a pull request touches.

This repository's own management script: `checks / spec`, `spec / command`,
and `spec / labels` call it, `just spec-check` and `just spec-changes` wrap
it, and the archive executor runs its `archive` command.

A request's *related changes* are the directories under the changes
directory (default ``openspec/changes/``) whose files the request touches;
a directory under ``archive/`` counts under its change name with the
leading ``YYYY-MM-DD-`` stripped. Each change is ``active`` (its directory
exists at the head), ``archived`` (an archive directory for it exists at
the head), or ``removed``.

The head is read through one of two sources, never through a checkout:
``--base``/``--head`` with git plumbing (``GitSource``), or ``--snapshot
FILE`` built from the GitHub REST API by the ``snapshot`` command
(``SnapshotSource``). ``archive`` is the one command that writes, and it
takes the git source only.

Every interface is checked where its data enters — git's exit status, the
API's status and JSON shape, the snapshot document's keys, the OpenSpec
CLI's presence and exit status — and fails naming it; nothing is reported
from a partial read.

Exit codes: 0 success; 1 a failure, a finding, or a refusal; 2 bad
arguments, an unknown change name included. An unexpected error ends with
its traceback (exit 1).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import itertools
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
SKIP_SPECS = re.compile(r"^\s*skip_specs\s*:\s*true\s*$", re.MULTILINE)
ARCHIVE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}-")
DOCS = ("proposal", "design", "tasks", "specs")

ARCHIVED_LABELS = ("spec/unarchived", "spec/archived")
PROGRESS_LABELS = ("spec/not-started", "spec/in-progress", "spec/done")
MANAGED_LABELS = ARCHIVED_LABELS + PROGRESS_LABELS

SNAPSHOT_SCHEMA = "spec-snapshot/1"
# Every key the snapshot readers index, with its JSON type. The base side
# records directory names only, because no base document is ever read.
SNAPSHOT_KEYS = {
    ("changed_paths",): list,
    ("base", "sha"): str,
    ("base", "dirs"): list,
    ("head", "sha"): str,
    ("head", "dirs"): list,
    ("head", "paths"): list,
    ("head", "files"): dict,
}
# A request-authored directory name is read only when it is a plain change name.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
SAFE_REPO = re.compile(r"^[A-Za-z0-9._-]{1,100}/[A-Za-z0-9._-]{1,100}$")
# The documents inside a change directory the commands read; nothing else is fetched.
CHANGE_DOCS = ("proposal.md", "design.md", "tasks.md", ".openspec.yaml")
MAX_FILES = 3000
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 8_000_000
# The platform token's REST budget is shared by every workflow of the repository.
MAX_CALLS = 200
ATTEMPTS = 3
TRANSIENT_STATUSES = (429, 500, 502, 503, 504)
JSON_KINDS = {dict: "an object", list: "a list", str: "a string"}
HTTP_HINTS = {
    401: "the token is missing or expired",
    403: "the token lacks `contents: read` or `pull-requests: read`, or the rate limit is spent",
    404: "the object is gone, as when the head is force-pushed or deleted while this runs, so rerun on the "
    "current head; or the token cannot see it",
}


class Usage(Exception):
    """Bad arguments (exit 2)."""


class Failure(Exception):
    """A tool failure, a finding, or a refusal (exit 1)."""


# --- git plumbing ---------------------------------------------------------


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise Failure(f"`git {' '.join(args)}` exited {result.returncode}: {result.stderr.strip()}")
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
    """A read-only view of the base and the head: directories on both sides, documents at the head."""

    base_label = "base"
    head_label = "head"

    def changed_paths(self) -> list[str]:
        raise NotImplementedError

    def exists_dir(self, side: str, path: str) -> bool:
        raise NotImplementedError

    def entries(self, side: str, path: str) -> list[str]:
        """Names directly under a directory; empty when the directory is absent."""
        raise NotImplementedError

    def head_files(self, path: str) -> list[str]:
        """Every file path under a directory at the head, sorted."""
        raise NotImplementedError

    def head_text(self, path: str) -> str | None:
        """A change document's text at the head, or None when the head has no such document."""
        raise NotImplementedError


class GitSource(Source):
    """Reads the two commits out of a local git object store.

    Use it where the head is already trusted or already present: a
    developer's clone, or an unprivileged pull-request check that checks
    the head out the ordinary way.
    """

    def __init__(self, root: Path, base: str, head: str) -> None:
        self.root = root
        self.shas = {"base": resolve(root, base, "--base"), "head": resolve(root, head, "--head")}
        self.base_label = base
        self.head_label = head

    def ls(self, side: str, path: str, recursive: bool = False) -> list[tuple[str, str]]:
        """``(type, path)`` records of ``git ls-tree``, which lists nothing, and exits 0, for an absent path."""
        out = git(self.root, "ls-tree", "-z", *(["-r"] if recursive else []), self.shas[side], "--", path)
        records = []
        for record in filter(None, out.split("\0")):
            meta, path = record.split("\t", 1)
            records.append((meta.split()[1], path))
        return records

    def changed_paths(self) -> list[str]:
        base, head = self.shas["base"], self.shas["head"]
        result = subprocess.run(
            ["git", "-C", str(self.root), "diff", "-z", "--name-only", "--no-renames", f"{base}...{head}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Failure(
                f"`git diff {self.base_label}...{self.head_label}` failed ({result.stderr.strip()}); the merge "
                "base must be reachable, so fetch with full history (fetch-depth 0) rather than a shallow clone."
            )
        return [path for path in result.stdout.split("\0") if path]

    def exists_dir(self, side: str, path: str) -> bool:
        return self.ls(side, path) == [("tree", path)]

    def entries(self, side: str, path: str) -> list[str]:
        return [child.rsplit("/", 1)[-1] for _, child in self.ls(side, f"{path}/")]

    def head_files(self, path: str) -> list[str]:
        return [child for kind, child in self.ls("head", f"{path}/", recursive=True) if kind == "blob"]

    def head_text(self, path: str) -> str | None:
        if self.ls("head", path) != [("blob", path)]:
            return None
        return git(self.root, "show", f"{self.shas['head']}:{path}")


class SnapshotSource(Source):
    """Reads a snapshot document built from the GitHub REST API.

    Use it in every privileged workflow (``pull_request_target``,
    ``issue_comment``, ``workflow_run``): the head's bytes arrive as data
    to parse, so no object authored by the request ever reaches the
    runner's git store, and nothing it carries is ever executed.
    """

    def __init__(self, path: Path, changes_dir: str) -> None:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise Failure(f"cannot read the snapshot {path}: {exc}") from exc
        problem = snapshot_problem(doc, changes_dir)
        if problem:
            raise Failure(
                f"snapshot {path}: {problem}; rebuild it with the snapshot command and the same --changes-dir."
            )
        self.changed = doc["changed_paths"]
        self.dirs = {"base": doc["base"]["dirs"], "head": doc["head"]["dirs"]}
        self.paths = {"base": [], "head": doc["head"]["paths"]}
        self.texts = doc["head"]["files"]
        self.base_label = doc["base"]["sha"]
        self.head_label = doc["head"]["sha"]

    def changed_paths(self) -> list[str]:
        return self.changed

    def exists_dir(self, side: str, path: str) -> bool:
        return path in self.dirs[side] or any(p.startswith(f"{path}/") for p in self.paths[side])

    def entries(self, side: str, path: str) -> list[str]:
        prefix = f"{path}/"
        names: list[str] = []
        for item in self.dirs[side] + self.paths[side]:
            if item.startswith(prefix):
                name = item[len(prefix) :].split("/")[0]
                if name not in names:
                    names.append(name)
        return names

    def head_files(self, path: str) -> list[str]:
        return [p for p in self.paths["head"] if p.startswith(f"{path}/")]

    def head_text(self, path: str) -> str | None:
        # The snapshot holds every change document the head has, so a missing key is a missing document.
        return self.texts.get(path)


def snapshot_problem(doc: object, changes_dir: str) -> str | None:
    """What makes a loaded snapshot document unusable for this changes directory, or None."""
    if not isinstance(doc, dict) or doc.get("schema") != SNAPSHOT_SCHEMA:
        return f"it is not a {SNAPSHOT_SCHEMA} document"
    if doc.get("changes_dir") != changes_dir:
        return f"it was built for the changes directory {doc.get('changes_dir')!r}, not {changes_dir!r}"
    for keys, kind in SNAPSHOT_KEYS.items():
        value = doc
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if not isinstance(value, kind):
            return f"`{'.'.join(keys)}` is missing or not {JSON_KINDS[kind]}"
    return None


# --- github rest api ------------------------------------------------------


class Api:
    """The few REST reads the snapshot needs, over the standard library.

    A fork's head SHA resolves from the base repository, so nothing here
    needs the fork. ``max_calls`` bounds what one request can spend of the
    repository's shared REST budget.
    """

    def __init__(self, repo: str, api_url: str, auth: str, max_calls: int) -> None:
        self.repo = repo
        self.api_url = api_url.rstrip("/")
        self.auth = auth
        self.max_calls = max_calls
        self.calls = 0

    def get(self, endpoint: str, kind: type) -> dict | list:
        """One GET whose body must be JSON of ``kind``; anything else fails naming the endpoint."""
        body = self._read(endpoint)
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise Failure(f"GET {endpoint} did not return JSON: {exc}") from exc
        if not isinstance(data, kind):
            raise Failure(
                f"GET {endpoint} returned {JSON_KINDS.get(type(data), type(data).__name__)} where the API "
                f"documents {JSON_KINDS[kind]}; nothing is reported from a partial snapshot."
            )
        return data

    def _read(self, endpoint: str) -> bytes:
        """The response body; a rate limit, a server error, or a network error is retried, nothing else."""
        request = urllib.request.Request(
            f"{self.api_url}/{endpoint}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.auth}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "spec-changes",
            },
        )
        attempt = 0
        while True:
            attempt += 1
            if self.calls >= self.max_calls:
                raise Failure(
                    f"the snapshot reached the --max-calls cap ({self.max_calls} API requests) at GET {endpoint}; "
                    "nothing is reported from a partial snapshot. Split the request, or raise the cap deliberately."
                )
            self.calls += 1
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code not in TRANSIENT_STATUSES or attempt == ATTEMPTS:
                    detail = exc.read().decode("utf-8", "replace")[:200].replace("\n", " ")
                    hint = HTTP_HINTS.get(exc.code, "the API refused the read")
                    raise Failure(f"GET {endpoint} answered HTTP {exc.code}: {detail}. Likely cause: {hint}.") from exc
            except urllib.error.URLError as exc:
                if attempt == ATTEMPTS:
                    raise Failure(f"cannot reach {self.api_url} for GET {endpoint}: {exc.reason}") from exc
            time.sleep(2**attempt)

    def pull_heads(self, number: int) -> tuple[str, str]:
        pull = self.get(f"repos/{self.repo}/pulls/{number}", dict)
        return pull["base"]["sha"], pull["head"]["sha"]

    def pull_files(self, number: int, max_files: int) -> list[str]:
        paths: list[str] = []
        for page in itertools.count(1):
            batch = self.get(f"repos/{self.repo}/pulls/{number}/files?per_page=100&page={page}", list)
            for item in batch:
                paths.append(item["filename"])
                # `git diff --no-renames` reports a rename as its old path plus
                # its new one, so the git source sees both. The API reports it as
                # one entry carrying `previous_filename`; keeping only `filename`
                # would hide a record moved out of its directory from every
                # privileged job while the unprivileged check still saw it.
                if "previous_filename" in item:
                    paths.append(item["previous_filename"])
            if len(batch) < 100 or len(paths) > max_files:
                break
        if len(paths) > max_files:
            raise Failure(
                f"the request touches at least {len(paths)} files, over the --max-files cap "
                f"({max_files}); split the request, or raise the cap deliberately."
            )
        return paths

    def tree(self, sha: str, path: str, recursive: bool = False) -> list[dict]:
        """The entries of one tree, named ``path`` in messages; a truncated listing fails."""
        endpoint = f"repos/{self.repo}/git/trees/{urllib.parse.quote(sha, safe='')}"
        data = self.get(endpoint + ("?recursive=1" if recursive else ""), dict)
        if data["truncated"]:
            raise Failure(
                f"the tree of {path} came back truncated (GET {endpoint}); nothing is reported from a partial "
                "snapshot. Narrow the changes directory or archive older changes."
            )
        return data["tree"]

    def blob_text(self, sha: str, path: str) -> str:
        endpoint = f"repos/{self.repo}/git/blobs/{urllib.parse.quote(sha, safe='')}"
        data = self.get(endpoint, dict)
        if data["encoding"] != "base64":
            raise Failure(f"GET {endpoint} for {path} returned encoding {data['encoding']!r}, not base64.")
        try:
            return base64.b64decode(data["content"]).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise Failure(f"{path} is not UTF-8 text ({exc}); nothing is reported from a partial snapshot.") from exc


def listing_at(api: Api, commit: str, path: str) -> list[dict] | None:
    """The entries of a directory at a commit, or None when the directory does not exist there.

    Each read after the first is of a tree the previous listing named, so a
    failed read is an error and never an absent directory.
    """
    entries = api.tree(commit, f"the commit {commit}")
    walked: list[str] = []
    for segment in path.split("/"):
        walked.append(segment)
        entry = next((e for e in entries if e["path"] == segment and e["type"] == "tree"), None)
        if entry is None:
            return None
        entries = api.tree(entry["sha"], "/".join(walked))
    return entries


def change_dirs(api: Api, commit: str, changes_dir: str) -> dict[str, str]:
    """Tree SHAs of the directories under the changes directory and its archive, by path."""
    shas: dict[str, str] = {}
    entries = listing_at(api, commit, changes_dir)
    if entries is None:  # a project that has no changes directory at this commit
        return shas
    for entry in entries:
        if entry["type"] == "tree":
            shas[f"{changes_dir}/{entry['path']}"] = entry["sha"]
    archive = f"{changes_dir}/archive"
    if archive in shas:
        for entry in api.tree(shas[archive], archive):
            if entry["type"] == "tree":
                shas[f"{archive}/{entry['path']}"] = entry["sha"]
    return shas


def is_change_doc(relative: str) -> bool:
    """Whether a path inside a change directory is a document the commands read."""
    return relative in CHANGE_DOCS or (relative.startswith("specs/") and relative.endswith("/spec.md"))


def build_snapshot(api: Api, number: int, changes_dir: str, caps: dict) -> dict:
    """Read the request's related changes into a document the snapshot source replays.

    Only the documents the commands read are fetched, and a snapshot that
    would be partial — a cap reached, a truncated tree, an undecodable
    document, a directory name that is not a plain change name — fails
    instead of being reported on: a label or a status derived from half the
    request is worse than no answer.
    """
    base_sha, head_sha = api.pull_heads(number)
    changed = api.pull_files(number, caps["max_files"])
    base = change_dirs(api, base_sha, changes_dir)
    head = change_dirs(api, head_sha, changes_dir)

    paths: list[str] = []
    files: dict[str, str] = {}
    total = 0
    for name in change_names(changed, changes_dir):
        if not SAFE_NAME.match(name):
            raise Failure(
                f"the request touches {changes_dir}/{name!r}, which is not a plain change name; "
                "nothing is reported from a partial snapshot."
            )
        path = head_dir(list(head), changes_dir, name)
        if path is None:  # removed at the head: nothing to read
            continue
        for entry in api.tree(head[path], path, recursive=True):
            if entry["type"] != "blob":
                continue
            full = f"{path}/{entry['path']}"
            paths.append(full)
            if not is_change_doc(entry["path"]):
                continue
            if entry["size"] > caps["max_file_bytes"]:
                raise Failure(
                    f"{full} is {entry['size']} bytes, over the --max-file-bytes cap ({caps['max_file_bytes']}); "
                    "nothing is reported from a partial snapshot."
                )
            if total + entry["size"] > caps["max_total_bytes"]:
                raise Failure(
                    f"the documents reached the --max-total-bytes cap ({caps['max_total_bytes']}) at {full}; "
                    "nothing is reported from a partial snapshot."
                )
            files[full] = api.blob_text(entry["sha"], full)
            total += entry["size"]

    return {
        "schema": SNAPSHOT_SCHEMA,
        "repo": api.repo,
        "pull_request": number,
        "changes_dir": changes_dir,
        "changed_paths": changed,
        "base": {"sha": base_sha, "dirs": list(base)},
        "head": {"sha": head_sha, "dirs": list(head), "paths": sorted(paths), "files": files},
        "bytes": total,
    }


# --- related changes ------------------------------------------------------


def change_names(paths: list[str], changes_dir: str) -> list[str]:
    prefix = f"{changes_dir}/"
    names: set[str] = set()
    for path in paths:
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix) :].split("/")
        if parts[0] == "archive" and len(parts) >= 3:
            names.add(ARCHIVE_DATE.sub("", parts[1]))
        elif parts[0] != "archive" and len(parts) >= 2:
            names.add(parts[0])
    return sorted(names)


def head_dir(dirs: list[str], changes_dir: str, name: str) -> str | None:
    """The directory a change has in a listing: active first, else its latest archive entry."""
    if f"{changes_dir}/{name}" in dirs:
        return f"{changes_dir}/{name}"
    archived = [d for d in dirs if d.startswith(f"{changes_dir}/archive/") and archived_name(d) == name]
    return max(archived, default=None)


def archived_name(path: str) -> str:
    return ARCHIVE_DATE.sub("", path.rsplit("/", 1)[-1])


def state_at(source: Source, side: str, changes_dir: str, name: str) -> tuple[str, str | None]:
    active = f"{changes_dir}/{name}"
    if source.exists_dir(side, active):
        return "active", active
    matches = [e for e in source.entries(side, f"{changes_dir}/archive") if ARCHIVE_DATE.sub("", e) == name]
    if matches:
        return "archived", f"{changes_dir}/archive/{max(matches)}"
    return "removed", None


def task_counts(text: str | None) -> dict:
    """Progress from a task list; a change without one has not started."""
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


def related_changes(source: Source, changes_dir: str) -> list[dict]:
    changes = []
    for name in change_names(source.changed_paths(), changes_dir):
        state, path = state_at(source, "head", changes_dir, name)
        base_state, _ = state_at(source, "base", changes_dir, name)
        marker = source.head_text(f"{path}/.openspec.yaml") if path else None
        changes.append(
            {
                "name": name,
                "state": state,
                "path": path,
                "at_base": base_state,
                "skip_specs": bool(marker and SKIP_SPECS.search(marker)),
                "tasks": task_counts(source.head_text(f"{path}/tasks.md") if path else None),
            }
        )
    return changes


def select(changes: list[dict], wanted: list[str]) -> list[dict]:
    """The named changes, or all of them when none is named; a name the request does not touch is a bad argument."""
    by_name = {c["name"]: c for c in changes}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        related = ", ".join(code_span(name) for name in by_name) or "none"
        raise Usage(
            f"not a change this request touches: {', '.join(code_span(name) for name in unknown)}. "
            f"Related changes: {related}."
        )
    return [by_name[name] for name in wanted] if wanted else changes


# --- rendering ------------------------------------------------------------


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


def status_markdown(changes: list[dict]) -> str:
    if not changes:
        return "No related OpenSpec change: the diff touches nothing under the changes directory.\n"
    lines = ["| Change | State | Tasks | Progress |", "|---|---|---|---|"]
    for c in changes:
        t = c["tasks"]
        name = code_span(c["name"], in_table=True)
        lines.append(f"| {name} | {c['state']} | {t['done']} done, {t['open']} open | {t['progress']} |")
    for c in changes:
        if c["tasks"]["open_tasks"]:
            lines.append("")
            lines.append(f"Open tasks of {code_span(c['name'])} (first ten):")
            for task in c["tasks"]["open_tasks"][:10]:
                lines.append(f"- {code_span(task[:117] + '...' if len(task) > 120 else task)}")
    return "\n".join(lines) + "\n"


def fence_for(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def doc_paths(source: Source, path: str, doc: str) -> list[str]:
    if doc == "specs":
        return [p for p in source.head_files(f"{path}/specs") if p.endswith("/spec.md")]
    return [f"{path}/{doc}.md"]


def show_markdown(source: Source, changes: list[dict], doc: str, url_prefix: str, max_chars: int) -> str:
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
            emit(f"### {code_span(change['name'])} — removed at the head; nothing to show.\n\n")
            continue
        for d in docs:
            paths = doc_paths(source, change["path"], d)
            if not paths:
                emit(f"**{code_span(change['path'] + '/specs/')}** — _no delta spec yet_\n\n")
                continue
            for p in paths:
                view = f" ([view]({link_to(url_prefix, p)}))" if url_prefix else ""
                header = f"**{code_span(p)}**{view}\n\n"
                if overflow:
                    emit(f"**{code_span(p)}** — omitted for size{view}\n\n")
                    continue
                text = source.head_text(p)
                if text is None:
                    emit(f"**{code_span(p)}** — _no {p.rsplit('/', 1)[-1]} yet_\n\n")
                    continue
                fence = fence_for(text)
                block = f"{header}{fence}markdown\n{text.rstrip()}\n{fence}\n\n"
                budget = max_chars - used
                if len(block) > budget:
                    overflow = True
                    keep = text[: max(0, budget - len(header) - len(fence) * 2 - 80)]
                    keep = keep[: keep.rfind("\n")] if "\n" in keep else keep
                    tail = f": {link_to(url_prefix, p)}" if url_prefix else ""
                    emit(f"{header}{fence}markdown\n{keep}\n{fence}\n… truncated — full file{tail}\n\n")
                    continue
                emit(block)
    return "".join(out) or "No related OpenSpec change: the diff touches nothing under the changes directory.\n"


# --- commands -------------------------------------------------------------


def open_source(args: argparse.Namespace, root: Path, changes_dir: str) -> Source | None:
    """The head source the arguments name, or None when they name none."""
    if args.snapshot and (args.base or args.head):
        raise Usage("--snapshot reads the head on its own; do not pass --base/--head with it.")
    if args.snapshot:
        return SnapshotSource(Path(args.snapshot), changes_dir)
    if args.base and args.head:
        return GitSource(root, args.base, args.head)
    if args.base or args.head:
        raise Usage("--base and --head go together; pass both, or pass --snapshot instead.")
    return None


def need_source(args: argparse.Namespace, root: Path, changes_dir: str) -> Source:
    source = open_source(args, root, changes_dir)
    if source is None:
        raise Usage(f"{args.command} needs a head source: --base with --head, or --snapshot from the snapshot command.")
    return source


def run_openspec(root: Path, executable: str, *args: str) -> None:
    env = {**os.environ, "OPENSPEC_NO_UPDATE_CHECK": "1"}
    try:
        # The CLI's progress output is relayed to stderr so that `--json` on stdout stays parseable.
        result = subprocess.run([executable, *args], cwd=root, env=env, stdout=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise Failure(
            f"`{executable}` is not installed; install the OpenSpec CLI pinned in the justfile (`just setup`)."
        ) from exc
    sys.stderr.write(result.stdout)
    if result.returncode != 0:
        raise Failure(f"`{executable} {' '.join(args)}` exited {result.returncode}.")


def cmd_snapshot(args: argparse.Namespace, root: Path, changes_dir: str) -> int:
    auth = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not auth:
        raise Failure("snapshot needs a GitHub token in GH_TOKEN or GITHUB_TOKEN.")
    api = Api(args.repo, args.api_url, auth, args.max_calls)
    caps = {
        "max_files": args.max_files,
        "max_file_bytes": args.max_file_bytes,
        "max_total_bytes": args.max_total_bytes,
    }
    doc = build_snapshot(api, args.pr, changes_dir, caps)
    text = json.dumps(doc, indent=2)
    if not args.out:
        print(text)
        return 0
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(
        f"snapshot of {api.repo}#{args.pr} at {doc['head']['sha'][:7]}: "
        f"{len(doc['head']['files'])} file(s), {doc['bytes']} byte(s), {api.calls} API call(s) → {args.out}",
        file=sys.stderr,
    )
    return 0


def cmd_status(args: argparse.Namespace, root: Path, changes_dir: str) -> int:
    source = need_source(args, root, changes_dir)
    changes = select(related_changes(source, changes_dir), args.change)
    if args.json:
        print(json.dumps({"base": source.base_label, "head": source.head_label, "changes": changes}, indent=2))
    else:
        sys.stdout.write(status_markdown(changes))
    return 0


def cmd_show(args: argparse.Namespace, root: Path, changes_dir: str) -> int:
    source = need_source(args, root, changes_dir)
    changes = select(related_changes(source, changes_dir), [args.change] if args.change else [])
    sys.stdout.write(show_markdown(source, changes, args.doc, args.url_prefix, args.max_chars))
    return 0


def cmd_check(args: argparse.Namespace, root: Path, changes_dir: str) -> int:
    source = open_source(args, root, changes_dir)
    if args.all and source is not None:
        raise Usage("check --all reads the working tree; do not pass a head source with it.")
    if not args.all and source is None:
        raise Usage("check needs a head source (--base with --head, or --snapshot), or --all.")
    if not args.no_validate:
        run_openspec(root, args.openspec, "validate", "--all", "--strict", "--no-interactive")
    split = args.shape == "split"
    findings: list[str] = []
    if args.all:
        live = root / changes_dir
        unarchived = (
            sorted(p.name for p in live.iterdir() if p.is_dir() and p.name != "archive") if live.is_dir() else []
        )
        for name in unarchived:
            # Under the combined shape the integration branch never holds an
            # unarchived change. Under split it holds every approved record
            # whose implementation has not landed yet, by design.
            if split:
                print(f"warning: unarchived change {name} on the integration branch (expected under split).")
            else:
                findings.append(f"unarchived change {name}: the integration branch holds only archived changes.")
    else:
        for c in related_changes(source, changes_dir):
            if c["state"] == "active":
                message = (
                    f"unarchived change {c['name']}: {c['tasks']['open']} open task(s) — archive it in this "
                    "pull request once the deliberation on the finished implementation closes; "
                    "the request stays red until then."
                )
                # The one request that may merge with an unarchived record is the
                # split shape's specification request: it carries the record and
                # no implementation, and the implementation requests that follow
                # archive it. A request that implemented something is not that
                # request, whatever its shape.
                if args.draft:
                    print(f"warning: {message}")
                elif split and c["tasks"]["done"] == 0:
                    print(
                        f"warning: unarchived change {c['name']} carries no implemented task; "
                        "allowed for a specification request under the split shape."
                    )
                else:
                    findings.append(message)
            elif c["state"] == "archived" and c["tasks"]["open"]:
                print(f"warning: archived change {c['name']} still has {c['tasks']['open']} open task(s).")
    for finding in findings:
        print(finding)
    return 1 if findings else 0


def cmd_archive(args: argparse.Namespace, root: Path, changes_dir: str) -> int:
    """Archive every complete related change, or none of them.

    This is the one command that writes, through the OpenSpec CLI
    (``openspec archive <name> --yes``, plus ``--skip-specs`` for a change
    whose ``.openspec.yaml`` sets ``skip_specs: true``; flags verified
    against OpenSpec 1.12.0 on 2026-09-17). It must only run where the
    head is already trusted, which is why it refuses any source but git.
    The CLI's own output goes to stderr, so ``--json`` on stdout stays
    machine-readable.
    """
    source = need_source(args, root, changes_dir)
    if not isinstance(source, GitSource):
        raise Usage("archive edits the working tree, so it needs --base and --head, not a snapshot.")
    if git(root, "rev-parse", "HEAD").strip() != source.shas["head"]:
        raise Usage("archive edits the working tree, so --head must be the checked-out HEAD; check it out first.")
    # The CLI rewrites the main specs as well as the change records, and the
    # documented follow-up stages the whole tree the tool owns.
    owned = str(Path(changes_dir).parent) if "/" in changes_dir else changes_dir
    if git(root, "status", "--porcelain", "--", owned).strip():
        raise Failure(f"{owned} has uncommitted changes; commit or stash them before archiving.")
    plan = {"archived": [], "skipped": [], "refused": [], "validated": False}
    todo: list[dict] = []
    for c in related_changes(source, changes_dir):
        if c["state"] != "active":
            plan["skipped"].append({"name": c["name"], "reason": c["state"]})
        elif c["tasks"]["open"]:
            plan["refused"].append(
                {"name": c["name"], "open": c["tasks"]["open"], "tasks": c["tasks"]["open_tasks"][:5]}
            )
        elif c["tasks"]["done"] == 0:
            plan["refused"].append({"name": c["name"], "open": 0, "tasks": ["no ticked task in tasks.md"]})
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


def cmd_labels(args: argparse.Namespace, root: Path, changes_dir: str) -> int:
    if args.taxonomy:
        print(json.dumps({"managed": list(MANAGED_LABELS)}, indent=2))
        return 0
    desired = desired_labels(related_changes(need_source(args, root, changes_dir), changes_dir))
    current = [label.strip() for label in args.current.split(",") if label.strip()]
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


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive number")
    return number


def repo_name(value: str) -> str:
    if not SAFE_REPO.match(value):
        raise argparse.ArgumentTypeError(f"{value!r} is not an OWNER/NAME pair")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spec_changes.py",
        description="Resolve, show, check, archive, and label the OpenSpec changes a pull request touches.",
        epilog=(
            "Examples: python3 scripts/spec_changes.py status --base origin/main --head HEAD; "
            "python3 scripts/spec_changes.py snapshot --repo owner/name --pr 12 --out snapshot.json; "
            "python3 scripts/spec_changes.py labels --snapshot snapshot.json --current 'spec/unarchived'"
        ),
    )
    parser.add_argument("--root", default=".", help="git work tree holding the changes directory (default: .)")
    parser.add_argument("--changes-dir", default="openspec/changes", help="changes directory relative to --root")
    # Not `required`: argparse would then report a missing command before an unknown option.
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def reads(p: argparse.ArgumentParser) -> None:
        group = p.add_argument_group("head source (one of)")
        group.add_argument("--base", help="base ref, read with git plumbing (with --head)")
        group.add_argument("--head", help="head ref, read with git plumbing (with --base)")
        group.add_argument(
            "--snapshot",
            help="snapshot file from the snapshot command; the only source a privileged workflow may use",
        )

    p = sub.add_parser("snapshot", help="build a head snapshot from the GitHub REST API (no checkout, no fetch)")
    p.add_argument("--repo", required=True, type=repo_name, help="OWNER/NAME of the repository the request targets")
    p.add_argument("--pr", required=True, type=positive, help="pull request number")
    p.add_argument("--out", help="write the snapshot here (default: stdout)")
    p.add_argument("--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    p.add_argument("--max-files", type=positive, default=MAX_FILES, help=f"cap on touched files (default {MAX_FILES})")
    p.add_argument("--max-file-bytes", type=positive, default=MAX_FILE_BYTES, help="per-file byte cap")
    p.add_argument("--max-total-bytes", type=positive, default=MAX_TOTAL_BYTES, help="total byte cap")
    p.add_argument("--max-calls", type=positive, default=MAX_CALLS, help=f"cap on API requests (default {MAX_CALLS})")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("status", help="print a progress table (markdown) for the related changes")
    reads(p)
    p.add_argument("--change", action="append", default=[], help="limit to this change (repeatable)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("show", help="print a change's documents inside fenced blocks")
    reads(p)
    p.add_argument("--change", help="one related change (default: every related change)")
    p.add_argument("--doc", choices=(*DOCS, "all"), default="all")
    p.add_argument("--url-prefix", default="", help="link prefix, e.g. https://github.com/o/r/blob/<sha>")
    p.add_argument("--max-chars", type=positive, default=60000, help="truncate the output at this size (default 60000)")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("check", help="strict validation plus the unarchived-change rule")
    reads(p)
    p.add_argument("--all", action="store_true", help="fail on any change outside archive/ (integration branch)")
    p.add_argument("--draft", action="store_true", help="report unarchived related changes as warnings")
    p.add_argument(
        "--shape",
        choices=("combined", "split"),
        default="combined",
        help=(
            "the change request shape the project's contract records (default: combined). "
            "Under split, a related change with no implemented task is allowed unarchived: "
            "that is the specification request, which the implementation requests archive later."
        ),
    )
    p.add_argument("--no-validate", action="store_true", help="skip the OpenSpec validator")
    p.add_argument("--openspec", default="openspec", help="OpenSpec CLI executable (default: openspec)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("archive", help="archive every complete related change through the OpenSpec CLI")
    reads(p)
    p.add_argument("--dry-run", action="store_true", help="print the plan; change nothing")
    p.add_argument("--openspec", default="openspec", help="OpenSpec CLI executable (default: openspec)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_archive)

    p = sub.add_parser("labels", help="compute the archive-axis and progress-axis labels")
    reads(p)
    p.add_argument("--current", default="", help="comma-separated labels currently on the request")
    p.add_argument("--taxonomy", action="store_true", help="print the managed label names")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_labels)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required: snapshot, status, show, check, archive, or labels")
    root = Path(args.root).resolve()
    changes_dir = args.changes_dir.strip("/")
    try:
        return args.func(args, root, changes_dir)
    except Usage as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Failure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
