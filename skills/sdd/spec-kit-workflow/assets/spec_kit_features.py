#!/usr/bin/env python3
"""Operate on the Spec-Kit features a pull or merge request touches.

The project's own management script for its Spec-Kit request automation:
the check job, the `/spec` comment commands, and the progress-label job
call it at scripts/spec_kit_features.py. It reads and reports; nothing
here edits the working tree.

A request's *touched features* are the numbered feature directories under
the kit's specs directory (default ``specs/``, entries named
``<NNN>-<name>``) whose files the request touches. Layout verified against
Spec-Kit's templates and feature script on 2026-09-17: ``spec.md`` and
``plan.md`` are required, ``tasks.md`` lists tasks as ``- [ ] T001 ...``
checkboxes. The kit ships no validator and no archive operation:
completion is every task ticked.

The head is read through one of two sources, never through a checkout:
``--base``/``--head`` with git plumbing (``GitSource``), or ``--snapshot
FILE`` built from the GitHub REST API by the ``snapshot`` command
(``SnapshotSource``). ``check --all`` reads the working tree instead.

Every interface is checked where its data enters — git's exit status, the
API's status and JSON shape, the snapshot document's keys — and fails
naming it; nothing is reported from a partial read.

Exit codes: 0 success; 1 a failure or a finding; 2 bad arguments, an
unknown feature name included. An unexpected error ends with its
traceback (exit 1).
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

OPEN_TASK = re.compile(r"^[ \t]*-[ \t]*\[ \][ \t]*(.*)$", re.MULTILINE)
DONE_TASK = re.compile(r"^\s*-\s*\[[xX]\]\s", re.MULTILINE)
FEATURE_DIR = re.compile(r"^\d{3,}-[A-Za-z0-9._-]+$")
DOCS = ("spec", "plan", "tasks")
REQUIRED = ("spec.md", "plan.md")
PROGRESS_LABELS = ("spec/not-started", "spec/in-progress", "spec/done")

SNAPSHOT_SCHEMA = "spec-kit-snapshot/1"
# Every key the snapshot reader indexes, with its JSON type.
SNAPSHOT_KEYS = {
    ("changed_paths",): list,
    ("base", "sha"): str,
    ("head", "sha"): str,
    ("head", "dirs"): list,
    ("head", "paths"): list,
    ("head", "files"): dict,
}
SAFE_REPO = re.compile(r"^[A-Za-z0-9._-]{1,100}/[A-Za-z0-9._-]{1,100}$")
MAX_FILES = 3000
# Room `show` keeps for its closing line that counts the documents cut for size.
OMITTED_RESERVE = 80
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
    """A failure or a finding (exit 1)."""


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
    """A read-only view of the head."""

    base_label = "base"
    head_label = "head"

    def changed_paths(self) -> list[str]:
        raise NotImplementedError

    def entries(self, path: str) -> list[str]:
        """Names directly under a directory at the head; empty when the directory is absent."""
        raise NotImplementedError

    def read(self, path: str) -> str | None:
        """A feature document's text at the head, or None when the head has no such document."""
        raise NotImplementedError


class GitSource(Source):
    """Reads the two commits out of a local git object store.

    Use it where the head is already trusted or already present: a
    developer's clone, or an unprivileged pull-request check that checks
    the head out the ordinary way.
    """

    def __init__(self, root: Path, base: str, head: str) -> None:
        self.root = root
        self.base = resolve(root, base, "--base")
        self.head = resolve(root, head, "--head")
        self.base_label = base
        self.head_label = head

    def ls(self, path: str) -> list[tuple[str, str]]:
        """``(type, path)`` records of ``git ls-tree``, which lists nothing, and exits 0, for an absent path."""
        out = git(self.root, "ls-tree", "-z", self.head, "--", path)
        records = []
        for record in filter(None, out.split("\0")):
            meta, name = record.split("\t", 1)
            records.append((meta.split()[1], name))
        return records

    def changed_paths(self) -> list[str]:
        result = subprocess.run(
            ["git", "-C", str(self.root), "diff", "-z", "--name-only", "--no-renames", f"{self.base}...{self.head}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Failure(
                f"`git diff {self.base_label}...{self.head_label}` failed ({result.stderr.strip()}); the merge "
                "base must be reachable, so fetch with full history (fetch-depth 0) rather than a shallow clone."
            )
        return [path for path in result.stdout.split("\0") if path]

    def entries(self, path: str) -> list[str]:
        return [child.rsplit("/", 1)[-1] for _, child in self.ls(f"{path}/")]

    def read(self, path: str) -> str | None:
        if self.ls(path) != [("blob", path)]:
            return None
        return git(self.root, "show", f"{self.head}:{path}")


class SnapshotSource(Source):
    """Reads a snapshot document built from the GitHub REST API.

    Use it in every privileged workflow (``pull_request_target``,
    ``issue_comment``, ``workflow_run``): the head's bytes arrive as data
    to parse, so no object authored by the request ever reaches the
    runner's git store, and nothing it carries is ever executed.
    """

    def __init__(self, path: Path, specs_dir: str) -> None:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise Failure(f"cannot read the snapshot {path}: {exc}") from exc
        problem = snapshot_problem(doc, specs_dir)
        if problem:
            raise Failure(f"snapshot {path}: {problem}; rebuild it with the snapshot command and the same --specs-dir.")
        self.changed = doc["changed_paths"]
        self.dirs = doc["head"]["dirs"]
        self.paths = doc["head"]["paths"]
        self.texts = doc["head"]["files"]
        self.base_label = doc["base"]["sha"]
        self.head_label = doc["head"]["sha"]

    def changed_paths(self) -> list[str]:
        return self.changed

    def entries(self, path: str) -> list[str]:
        prefix = f"{path}/"
        names: list[str] = []
        for item in self.dirs + self.paths:
            if item.startswith(prefix):
                name = item[len(prefix) :].split("/")[0]
                if name not in names:
                    names.append(name)
        return names

    def read(self, path: str) -> str | None:
        # The snapshot holds every feature document the head has, so a missing key is a missing document.
        return self.texts.get(path)


def snapshot_problem(doc: object, specs_dir: str) -> str | None:
    """What makes a loaded snapshot document unusable for this specs directory, or None."""
    if not isinstance(doc, dict) or doc.get("schema") != SNAPSHOT_SCHEMA:
        return f"it is not a {SNAPSHOT_SCHEMA} document"
    if doc.get("specs_dir") != specs_dir:
        return f"it was built for the specs directory {doc.get('specs_dir')!r}, not {specs_dir!r}"
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

    def get(self, endpoint: str, kind: type, keys: tuple[str, ...] = ()) -> dict | list:
        """One GET whose body must be JSON of ``kind``; anything else fails naming the endpoint.

        ``keys`` are the fields the caller reads, of the object or of every
        item of the list; a missing one fails here, so callers read them
        directly.
        """
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
        what = "an entry" if isinstance(data, list) else "an object"
        for item in data if isinstance(data, list) else [data]:
            missing = [key for key in keys if not isinstance(item, dict) or key not in item]
            if missing:
                raise Failure(
                    f"GET {endpoint} returned {what} without `{'`, `'.join(missing)}`: {str(item)[:200]}; "
                    "nothing is reported from a partial snapshot."
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
                "User-Agent": "spec-kit-features",
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

    def pull_heads(self, number: int) -> tuple[str, str, int]:
        """The base and head SHAs and the count of changed files the request reports."""
        endpoint = f"repos/{self.repo}/pulls/{number}"
        pull = self.get(endpoint, dict, ("base", "head", "changed_files"))
        for side in ("base", "head"):
            if not isinstance(pull[side], dict) or not isinstance(pull[side].get("sha"), str):
                raise Failure(
                    f"GET {endpoint} returned a `{side}` without a `sha`: {str(pull[side])[:200]}; "
                    "nothing is reported from a partial snapshot."
                )
        if not isinstance(pull["changed_files"], int):
            raise Failure(
                f"GET {endpoint} returned `changed_files` {pull['changed_files']!r}, not a count; "
                "nothing is reported from a partial snapshot."
            )
        return pull["base"]["sha"], pull["head"]["sha"], pull["changed_files"]

    def pull_files(self, number: int, max_files: int, changed_files: int) -> list[str]:
        """The request's paths; a listing shorter than the request's ``changed_files`` fails.

        The endpoint lists at most 3000 files however it is paged, and a
        larger request comes back cut short without an error: only the count
        the pull request reports shows the cut.
        """
        endpoint = f"repos/{self.repo}/pulls/{number}/files"
        paths: list[str] = []
        listed = 0
        for page in itertools.count(1):
            batch = self.get(f"{endpoint}?per_page=100&page={page}", list, ("filename",))
            listed += len(batch)
            for item in batch:
                paths.append(item["filename"])
                # `git diff --no-renames` reports a rename as its old path plus
                # its new one, so the git source sees both. The API reports it as
                # one entry carrying `previous_filename`; keeping only `filename`
                # would hide a feature moved out of its directory from every
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
        if listed < changed_files:
            raise Failure(
                f"GET {endpoint} listed {listed} of the request's {changed_files} changed files (the endpoint "
                "lists at most 3000); nothing is reported from a partial snapshot. Split the request."
            )
        return paths

    def tree(self, sha: str, path: str, recursive: bool = False) -> list[dict]:
        """The entries of one tree, named ``path`` in messages; a truncated listing fails."""
        endpoint = f"repos/{self.repo}/git/trees/{urllib.parse.quote(sha, safe='')}"
        data = self.get(endpoint + ("?recursive=1" if recursive else ""), dict, ("truncated", "tree"))
        bad = next(
            (
                e
                for e in data["tree"]
                if not isinstance(e, dict)
                or not {"path", "type", "sha"} <= e.keys()
                or (e["type"] == "blob" and "size" not in e)
            ),
            None,
        )
        if bad is not None:
            raise Failure(
                f"GET {endpoint} listed an entry of {path} without its path, type, sha, or a blob's size: "
                f"{str(bad)[:200]}; nothing is reported from a partial snapshot."
            )
        if data["truncated"]:
            raise Failure(
                f"the tree of {path} came back truncated (GET {endpoint}); nothing is reported from a partial "
                "snapshot. Narrow the specs directory."
            )
        return data["tree"]

    def blob_text(self, sha: str, path: str) -> str:
        endpoint = f"repos/{self.repo}/git/blobs/{urllib.parse.quote(sha, safe='')}"
        data = self.get(endpoint, dict, ("encoding", "content"))
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


def build_snapshot(api: Api, number: int, specs_dir: str, caps: dict) -> dict:
    """Read the request's touched features into a document the snapshot source replays.

    Only the documents the commands read are fetched, and a snapshot that
    would be partial — a cap reached, a truncated tree, an undecodable
    document — fails instead of being reported on: a label derived from
    half the request is worse than no answer.
    """
    base_sha, head_sha, changed_files = api.pull_heads(number)
    changed = api.pull_files(number, caps["max_files"], changed_files)
    entries = listing_at(api, head_sha, specs_dir)
    # A project with no specs directory at the head has no feature to read.
    shas = {f"{specs_dir}/{e['path']}": e["sha"] for e in entries or [] if e["type"] == "tree"}

    paths: list[str] = []
    files: dict[str, str] = {}
    total = 0
    for name in feature_names(changed, specs_dir):
        path = f"{specs_dir}/{name}"
        if path not in shas:  # removed at the head: nothing to read
            continue
        for entry in api.tree(shas[path], path, recursive=True):
            if entry["type"] != "blob":
                continue
            full = f"{path}/{entry['path']}"
            paths.append(full)
            # Only the documents the commands read are fetched; the rest are listed by path.
            if entry["path"] not in {f"{d}.md" for d in DOCS}:
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
        "specs_dir": specs_dir,
        "changed_paths": changed,
        "base": {"sha": base_sha},
        "head": {"sha": head_sha, "dirs": list(shas), "paths": sorted(paths), "files": files},
        "bytes": total,
    }


# --- touched features -----------------------------------------------------


def feature_names(paths: list[str], specs_dir: str) -> list[str]:
    prefix = f"{specs_dir}/"
    names: set[str] = set()
    for path in paths:
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix) :].split("/")
        if len(parts) >= 2 and FEATURE_DIR.match(parts[0]):
            names.add(parts[0])
    return sorted(names)


def task_counts(text: str | None) -> dict:
    """Progress from a task list; a feature without one has not started."""
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
    """The named features, or all of them when none is named; a name the request does not touch is a bad argument."""
    by_name = {f["name"]: f for f in features}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        touched = ", ".join(code_span(name) for name in by_name) or "none"
        raise Usage(
            f"not a feature this request touches: {', '.join(code_span(name) for name in unknown)}. "
            f"Touched features: {touched}."
        )
    return [by_name[name] for name in wanted] if wanted else features


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
    """The documents as Markdown, at most ``max_chars`` long.

    The document that crosses the budget is cut and linked; every document
    after it is counted in one closing line instead of listed, so the output
    stays within ``max_chars`` however many documents follow.
    """
    docs = DOCS if doc == "all" else (doc,)
    out: list[str] = []
    used = 0
    overflow = False
    omitted = 0

    def emit(block: str) -> None:
        nonlocal used
        out.append(block)
        used += len(block)

    for f in features:
        if f["state"] == "removed":
            if not overflow:
                emit(f"### {code_span(f['name'])} — removed at the head; nothing to show.\n\n")
            continue
        for d in docs:
            p = f"{f['path']}/{d}.md"
            text = source.read(p)
            if text is None:
                if not overflow:
                    emit(f"**{code_span(p)}** — _no {d}.md yet_\n\n")
                continue
            if overflow:
                omitted += 1
                continue
            link = link_to(url_prefix, p) if url_prefix else ""
            view = f" ([view]({link}))" if url_prefix else ""
            header = f"**{code_span(p)}**{view}\n\n"
            fence = fence_for(text)
            block = f"{header}{fence}markdown\n{text.rstrip()}\n{fence}\n\n"
            budget = max_chars - used - OMITTED_RESERVE
            if len(block) > budget:
                overflow = True
                opening = f"{header}{fence}markdown\n"
                tail = f": {link}" if url_prefix else ""
                closing = f"\n{fence}\n… truncated — full file{tail}\n\n"
                keep = text[: max(0, budget - len(opening) - len(closing))]
                keep = keep[: keep.rfind("\n")] if "\n" in keep else keep
                emit(f"{opening}{keep}{closing}")
                continue
            emit(block)
    if omitted:
        emit(f"_{omitted} more document{'' if omitted == 1 else 's'} omitted for size._\n")
    return "".join(out) or "No touched Spec-Kit feature: the diff touches nothing under the specs directory.\n"


# --- commands -------------------------------------------------------------


def open_source(args: argparse.Namespace, root: Path, specs_dir: str) -> Source | None:
    """The head source the arguments name, or None when they name none."""
    if args.snapshot and (args.base or args.head):
        raise Usage("--snapshot reads the head on its own; do not pass --base/--head with it.")
    if args.snapshot:
        return SnapshotSource(Path(args.snapshot), specs_dir)
    if args.base and args.head:
        return GitSource(root, args.base, args.head)
    if args.base or args.head:
        raise Usage("--base and --head go together; pass both, or pass --snapshot instead.")
    return None


def need_source(args: argparse.Namespace, root: Path, specs_dir: str) -> Source:
    source = open_source(args, root, specs_dir)
    if source is None:
        raise Usage(f"{args.command} needs a head source: --base with --head, or --snapshot from the snapshot command.")
    return source


def cmd_snapshot(args: argparse.Namespace, root: Path, specs_dir: str) -> int:
    auth = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not auth:
        raise Failure("snapshot needs a GitHub token in GH_TOKEN or GITHUB_TOKEN.")
    api = Api(args.repo, args.api_url, auth, args.max_calls)
    caps = {
        "max_files": args.max_files,
        "max_file_bytes": args.max_file_bytes,
        "max_total_bytes": args.max_total_bytes,
    }
    doc = build_snapshot(api, args.pr, specs_dir, caps)
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


def cmd_status(args: argparse.Namespace, root: Path, specs_dir: str) -> int:
    source = need_source(args, root, specs_dir)
    features = select(touched_features(source, specs_dir), args.feature)
    if args.json:
        print(json.dumps({"base": source.base_label, "head": source.head_label, "features": features}, indent=2))
    else:
        sys.stdout.write(status_markdown(features))
    return 0


def cmd_show(args: argparse.Namespace, root: Path, specs_dir: str) -> int:
    source = need_source(args, root, specs_dir)
    features = select(touched_features(source, specs_dir), [args.feature] if args.feature else [])
    sys.stdout.write(show_markdown(source, features, args.doc, args.url_prefix, args.max_chars))
    return 0


def cmd_check(args: argparse.Namespace, root: Path, specs_dir: str) -> int:
    source = open_source(args, root, specs_dir)
    if args.all and source is not None:
        raise Usage("check --all reads the working tree; do not pass a head source with it.")
    if not args.all and source is None:
        raise Usage("check needs a head source (--base with --head, or --snapshot), or --all.")
    findings: list[str] = []
    if args.all:
        live = root / specs_dir
        features = (
            sorted(p for p in live.iterdir() if p.is_dir() and FEATURE_DIR.match(p.name)) if live.is_dir() else []
        )
        for entry in features:
            for required in REQUIRED:
                if not (entry / required).is_file():
                    findings.append(f"feature {entry.name}: missing {required}.")
    else:
        for f in touched_features(source, specs_dir):
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


def cmd_labels(args: argparse.Namespace, root: Path, specs_dir: str) -> int:
    if args.taxonomy:
        print(json.dumps({"managed": list(PROGRESS_LABELS)}, indent=2))
        return 0
    desired = desired_labels(touched_features(need_source(args, root, specs_dir), specs_dir))
    current = [label.strip() for label in args.current.split(",") if label.strip()]
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
    p.add_argument("--max-chars", type=positive, default=60000, help="truncate the output at this size (default 60000)")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required: snapshot, status, show, check, or labels")
    root = Path(args.root).resolve()
    specs_dir = args.specs_dir.strip("/")
    try:
        return args.func(args, root, specs_dir)
    except Usage as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Failure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
