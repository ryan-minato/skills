#!/usr/bin/env python3
"""Operate on the OpenSpec changes a pull or merge request touches.

A request's *related changes* are the directories under the changes
directory (default ``openspec/changes/``) whose files the request touches;
a directory under ``archive/`` counts under its change name with the
leading ``YYYY-MM-DD-`` stripped. Each change is ``active`` (its directory
exists at the head), ``archived`` (an archive directory for it exists at
the head), or ``removed``.

The head is read through one of two sources, never through a checkout:

* ``--base``/``--head`` reads the two commits with git plumbing. Use it
  where the head is already trusted or already present: a developer's
  clone, or an unprivileged pull-request check that checks the head out
  the ordinary way.
* ``--snapshot FILE`` reads a document built by the ``snapshot`` command,
  which pulls the head's file list and contents from the GitHub REST API.
  Use it in every privileged workflow (``pull_request_target``,
  ``issue_comment``, ``workflow_run``), so no object authored by the
  request ever reaches the runner's git store. The snapshot's bytes are
  parsed and never executed, change names are checked against a strict
  pattern before they reach a URL, only the documents the commands read
  are fetched, and the file, byte, and API-call caps below bound what one
  request can make the workflow read. A snapshot that would be partial (a
  cap reached, a truncated tree) fails instead of being reported on.
  Request-authored names, paths, and task text reach the rendered
  markdown only inside code spans, so they cannot add links or mentions.

``check`` and ``archive`` are the exceptions: both act on the working
tree, so both keep the git source. ``archive`` edits it through the
OpenSpec CLI (``openspec archive <name> --yes``, plus ``--skip-specs``
for a change whose ``.openspec.yaml`` sets ``skip_specs: true``; flags
verified against OpenSpec 1.12.0 on 2026-09-17) and must only run where
the head is already trusted. The CLI's own output goes to stderr, so
``--json`` output on stdout stays machine-readable.

Exit codes: 0 success; 1 a failure, a finding, or a refusal; 2 bad
arguments, an unresolvable ref, or a tree that is not a git repository.
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

OPEN_TASK = re.compile(r"^\s*-\s*\[ \]\s+(.*)$", re.MULTILINE)
DONE_TASK = re.compile(r"^\s*-\s*\[[xX]\]\s", re.MULTILINE)
SKIP_SPECS = re.compile(r"^\s*skip_specs\s*:\s*true\s*$", re.MULTILINE)
ARCHIVE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}-")
DOCS = ("proposal", "design", "tasks", "specs")

TRIGGER_LABEL = "spec/archive"
ARCHIVED_LABELS = ("spec/unarchived", "spec/archived")
PROGRESS_LABELS = ("spec/not-started", "spec/in-progress", "spec/done")
MANAGED_LABELS = ARCHIVED_LABELS + PROGRESS_LABELS

SNAPSHOT_SCHEMA = "spec-snapshot/1"
# A change name reaches an API path, so it is checked before it is used.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
SAFE_REPO = re.compile(r"^[A-Za-z0-9._-]{1,100}/[A-Za-z0-9._-]{1,100}$")
# The documents inside a change directory the commands read; nothing else is fetched.
CHANGE_DOCS = ("proposal.md", "design.md", "tasks.md", ".openspec.yaml")
MAX_FILES = 3000
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 8_000_000
# The platform token's REST budget is shared by every workflow of the repository.
MAX_CALLS = 200


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
            f"{what} {ref!r} does not resolve to a commit in {root}; fetch it first, or read the head "
            "through `snapshot` instead of fetching it into a privileged workflow."
        )
    return result.stdout.strip()


# --- sources --------------------------------------------------------------


class Source:
    """A read-only view of the base and the head, by side name."""

    base_label = "base"
    head_label = "head"

    def changed_paths(self) -> list[str]:
        raise NotImplementedError

    def exists_dir(self, side: str, path: str) -> bool:
        raise NotImplementedError

    def entries(self, side: str, path: str) -> list[str]:
        raise NotImplementedError

    def files(self, side: str, path: str) -> list[str]:
        raise NotImplementedError

    def read(self, side: str, path: str) -> str | None:
        raise NotImplementedError


class GitSource(Source):
    """Reads the two commits out of a local git object store."""

    def __init__(self, root: Path, base: str, head: str) -> None:
        self.root = root
        self.refs = {"base": base, "head": head}
        self.base_label = base
        self.head_label = head

    def _ref(self, side: str) -> str:
        return self.refs[side]

    def changed_paths(self) -> list[str]:
        base, head = self.refs["base"], self.refs["head"]
        result = subprocess.run(
            ["git", "-C", str(self.root), "diff", "--name-only", "--no-renames", f"{base}...{head}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise Usage(
                f"`git diff {base}...{head}` failed ({result.stderr.strip()}); the merge base must be reachable — "
                "fetch with full history (fetch-depth 0) rather than a shallow clone."
            )
        return [line for line in result.stdout.splitlines() if line]

    def exists_dir(self, side: str, path: str) -> bool:
        result = subprocess.run(
            ["git", "-C", str(self.root), "cat-file", "-e", f"{self._ref(side)}:{path}"], capture_output=True
        )
        return result.returncode == 0

    def entries(self, side: str, path: str) -> list[str]:
        out = git(self.root, "ls-tree", "--name-only", self._ref(side), f"{path.rstrip('/')}/", check=False)
        return [line.rsplit("/", 1)[-1] for line in out.splitlines() if line]

    def files(self, side: str, path: str) -> list[str]:
        out = git(self.root, "ls-tree", "-r", "--name-only", self._ref(side), f"{path.rstrip('/')}/", check=False)
        return [line for line in out.splitlines() if line]

    def read(self, side: str, path: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(self.root), "show", f"{self._ref(side)}:{path}"], capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else None


class SnapshotSource(Source):
    """Reads a snapshot document built from the GitHub REST API."""

    def __init__(self, doc: dict) -> None:
        if doc.get("schema") != SNAPSHOT_SCHEMA:
            raise Usage(f"snapshot schema {doc.get('schema')!r} is not {SNAPSHOT_SCHEMA!r}; rebuild it.")
        self.doc = doc
        self.sides = {"base": doc.get("base") or {}, "head": doc.get("head") or {}}
        self.base_label = self.sides["base"].get("sha", "base")
        self.head_label = self.sides["head"].get("sha", "head")

    def _side(self, side: str) -> dict:
        return self.sides[side]

    def changed_paths(self) -> list[str]:
        return list(self.doc.get("changed_paths") or [])

    def exists_dir(self, side: str, path: str) -> bool:
        s = self._side(side)
        path = path.rstrip("/")
        if path in (s.get("dirs") or []):
            return True
        return any(p.startswith(path + "/") for p in (s.get("paths") or []))

    def entries(self, side: str, path: str) -> list[str]:
        s = self._side(side)
        prefix = path.rstrip("/") + "/"
        names: list[str] = []
        for item in list(s.get("dirs") or []) + list(s.get("paths") or []):
            if item.startswith(prefix):
                name = item[len(prefix) :].split("/")[0]
                if name and name not in names:
                    names.append(name)
        return names

    def files(self, side: str, path: str) -> list[str]:
        prefix = path.rstrip("/") + "/"
        return [p for p in (self._side(side).get("paths") or []) if p.startswith(prefix)]

    def read(self, side: str, path: str) -> str | None:
        return (self._side(side).get("files") or {}).get(path)


# --- github rest api ------------------------------------------------------


class Api:
    """The few REST reads the snapshot needs, over the standard library."""

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
                "User-Agent": "spec-changes",
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
            paths.extend(item["filename"] for item in batch if "filename" in item)
            if len(batch) < 100 or len(paths) >= max_files:
                break
            page += 1
        if len(paths) >= max_files:
            raise Failure(
                f"the request touches at least {len(paths)} files, at or over the --max-files cap "
                f"({max_files}); split the request, or raise the cap deliberately."
            )
        return paths

    def tree(self, sha: str, recursive: bool = False) -> dict | None:
        query = "?recursive=1" if recursive else ""
        return self.get(f"repos/{self.repo}/git/trees/{urllib.parse.quote(sha, safe='')}{query}", allow_404=True)

    def blob_text(self, sha: str) -> str | None:
        data = self.get(f"repos/{self.repo}/git/blobs/{urllib.parse.quote(sha, safe='')}", allow_404=True)
        if not data or data.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(data["content"]).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None


def tree_sha_at(api: Api, commit: str, segments: list[str]) -> str | None:
    """Walk a commit's tree down a path, one tree read per segment."""
    sha = commit
    for segment in segments:
        data = api.tree(sha)
        if not data:
            return None
        entry = next((e for e in data.get("tree", []) if e.get("path") == segment and e.get("type") == "tree"), None)
        if entry is None:
            return None
        sha = entry["sha"]
    return sha


def side_listing(api: Api, commit: str, changes_dir: str) -> dict:
    """Directory names under the changes directory and its archive."""
    dirs: list[str] = []
    shas: dict[str, str] = {}
    root = tree_sha_at(api, commit, changes_dir.split("/"))
    if root is None:
        return {"sha": commit, "dirs": dirs, "shas": shas}
    listing = api.tree(root) or {"tree": []}
    for entry in listing.get("tree", []):
        if entry.get("type") != "tree":
            continue
        name = entry["path"]
        dirs.append(f"{changes_dir}/{name}")
        shas[f"{changes_dir}/{name}"] = entry["sha"]
    archive_sha = shas.get(f"{changes_dir}/archive")
    if archive_sha:
        archive = api.tree(archive_sha) or {"tree": []}
        for entry in archive.get("tree", []):
            if entry.get("type") != "tree":
                continue
            path = f"{changes_dir}/archive/{entry['path']}"
            dirs.append(path)
            shas[path] = entry["sha"]
    return {"sha": commit, "dirs": dirs, "shas": shas}


def is_change_doc(relative: str) -> bool:
    """Whether a path inside a change directory is a document the commands read."""
    return relative in CHANGE_DOCS or (relative.startswith("specs/") and relative.endswith("/spec.md"))


def build_snapshot(api: Api, number: int, changes_dir: str, caps: dict) -> dict:
    pull = api.pull(number)
    base_sha = pull["base"]["sha"]
    head_sha = pull["head"]["sha"]
    changed = api.pull_files(number, caps["max_files"])

    base = side_listing(api, base_sha, changes_dir)
    head = side_listing(api, head_sha, changes_dir)

    names = change_names(changed, changes_dir)
    skipped: list[dict] = []
    wanted: list[str] = []
    for name in names:
        if SAFE_NAME.match(name):
            wanted.append(name)
        else:
            skipped.append({"name": name, "reason": "the directory name is not a plain change name"})

    paths: list[str] = []
    files: dict[str, str] = {}
    total = 0
    for name in wanted:
        path = None
        if f"{changes_dir}/{name}" in head["dirs"]:
            path = f"{changes_dir}/{name}"
        else:
            matches = [
                d
                for d in head["dirs"]
                if d.startswith(f"{changes_dir}/archive/") and ARCHIVE_DATE.sub("", d.rsplit("/", 1)[-1]) == name
            ]
            path = sorted(matches)[-1] if matches else None
        if path is None:
            continue
        listing = api.tree(head["shas"][path], recursive=True) or {"tree": []}
        if listing.get("truncated"):
            raise Failure(f"the tree of {path} came back truncated; nothing is reported from a partial snapshot.")
        for entry in listing.get("tree", []):
            if entry.get("type") != "blob":
                continue
            full = f"{path}/{entry['path']}"
            paths.append(full)
            if not is_change_doc(entry["path"]):
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
                skipped.append({"path": full, "reason": "not decodable as UTF-8 text"})
                continue
            files[full] = text
            total += size

    return {
        "schema": SNAPSHOT_SCHEMA,
        "repo": api.repo,
        "pull_request": number,
        "changes_dir": changes_dir,
        "changed_paths": changed,
        "base": {"sha": base_sha, "dirs": base["dirs"]},
        "head": {"sha": head_sha, "dirs": head["dirs"], "paths": sorted(paths), "files": files},
        "skipped": skipped,
        "bytes": total,
    }


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


def archive_dir(source: Source, side: str, changes_dir: str, name: str) -> str | None:
    matches = [e for e in source.entries(side, f"{changes_dir}/archive") if ARCHIVE_DATE.sub("", e) == name]
    return f"{changes_dir}/archive/{sorted(matches)[-1]}" if matches else None


def state_at(source: Source, side: str, changes_dir: str, name: str) -> tuple[str, str | None]:
    active = f"{changes_dir}/{name}"
    if source.exists_dir(side, active):
        return "active", active
    archived = archive_dir(source, side, changes_dir, name)
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


def related_changes(source: Source, changes_dir: str) -> list[dict]:
    changes = []
    for name in change_names(source.changed_paths(), changes_dir):
        state, path = state_at(source, "head", changes_dir, name)
        base_state, _ = state_at(source, "base", changes_dir, name)
        tasks = task_counts(source.read("head", f"{path}/tasks.md")) if path else task_counts(None)
        marker = source.read("head", f"{path}/.openspec.yaml") if path else None
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


def doc_paths(source: Source, change: dict, doc: str) -> list[str]:
    path = change["path"]
    if path is None:
        return []
    if doc == "specs":
        return [p for p in source.files("head", f"{path}/specs") if p.endswith("/spec.md")]
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
            paths = doc_paths(source, change, d)
            if not paths:
                emit(f"**{code_span(change['name'] + '/' + d + '.md')}** — _no {d}.md yet_\n\n")
                continue
            for p in paths:
                link = link_to(url_prefix, p) if url_prefix else ""
                view = f" ([view]({link}))" if url_prefix else ""
                header = f"**{code_span(p)}**{view}\n\n"
                if overflow:
                    emit(f"**{code_span(p)}** — omitted for size{view}\n\n")
                    continue
                text = source.read("head", p)
                if text is None:
                    emit(f"**{code_span(p)}** — not in the snapshot (not decodable as text){view}\n\n")
                    continue
                fence = fence_for(text)
                block = f"{header}{fence}markdown\n{text.rstrip()}\n{fence}\n\n"
                budget = max_chars - used
                if len(block) > budget:
                    overflow = True
                    keep = text[: max(0, budget - len(header) - len(fence) * 2 - 80)]
                    keep = keep[: keep.rfind("\n")] if "\n" in keep else keep
                    tail = f": {link}" if url_prefix else ""
                    emit(f"{header}{fence}markdown\n{keep}\n{fence}\n… truncated — full file{tail}\n\n")
                    continue
                emit(block)
    return "".join(out) or "No related OpenSpec change: the diff touches nothing under the changes directory.\n"


# --- commands -------------------------------------------------------------


def run_openspec(root: Path, executable: str, *args: str) -> None:
    env = {**os.environ, "OPENSPEC_NO_UPDATE_CHECK": "1"}
    try:
        # The CLI's progress output is relayed to stderr so that `--json` on stdout stays parseable.
        result = subprocess.run([executable, *args], cwd=root, env=env, stdout=subprocess.PIPE, text=True)
    except OSError as exc:
        raise Failure(f"cannot run `{executable}`: {exc}; install the pinned OpenSpec CLI first.") from exc
    if result.stdout:
        sys.stderr.write(result.stdout)
    if result.returncode != 0:
        raise Failure(f"`{executable} {' '.join(args)}` exited {result.returncode}.")


def cmd_snapshot(args, root, changes_dir) -> int:
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
    doc = build_snapshot(api, args.pr, changes_dir, caps)
    text = json.dumps(doc, indent=2)
    if args.out:
        try:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            raise Usage(f"cannot write the snapshot to {args.out}: {exc}") from exc
        print(
            f"snapshot of {api.repo}#{args.pr} at {doc['head']['sha'][:7]}: "
            f"{len(doc['head']['files'])} file(s), {doc['bytes']} byte(s), {api.calls} API call(s) "
            f"→ {args.out}",
            file=sys.stderr,
        )
    else:
        print(text)
    for item in doc["skipped"]:
        print(f"skipped {item.get('path') or item.get('name')}: {item['reason']}", file=sys.stderr)
    return 0


def cmd_related(args, root, changes_dir) -> int:
    source = args.source
    changes = related_changes(source, changes_dir)
    if args.json:
        print(json.dumps({"base": source.base_label, "head": source.head_label, "changes": changes}, indent=2))
    else:
        for c in changes:
            print(f"{c['name']}\t{c['state']}\t{c['tasks']['done']}/{c['tasks']['open']}")
    return 0


def cmd_status(args, root, changes_dir) -> int:
    source = args.source
    changes = select(related_changes(source, changes_dir), args.change)
    if args.json:
        print(json.dumps({"base": source.base_label, "head": source.head_label, "changes": changes}, indent=2))
    else:
        sys.stdout.write(status_markdown(changes))
    return 0


def cmd_show(args, root, changes_dir) -> int:
    source = args.source
    changes = select(related_changes(source, changes_dir), [args.change] if args.change else [])
    sys.stdout.write(show_markdown(source, changes, args.doc, args.url_prefix, args.max_chars))
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
        for c in related_changes(args.source, changes_dir):
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
    changes = related_changes(args.source, changes_dir)
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
    desired = desired_labels(related_changes(args.source, changes_dir))
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
            "python3 scripts/spec_changes.py snapshot --repo owner/name --pr 12 --out snapshot.json; "
            "python3 scripts/spec_changes.py labels --snapshot snapshot.json --current 'spec/unarchived'"
        ),
    )
    parser.add_argument("--root", default=".", help="git work tree holding the changes directory (default: .)")
    parser.add_argument("--changes-dir", default="openspec/changes", help="changes directory relative to --root")
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

    p = sub.add_parser("related", help="list the related changes with their state and task counts")
    reads(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_related)

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
    p.add_argument("--max-chars", type=int, default=60000, help="truncate the output at this size (default 60000)")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("check", help="strict validation plus the unarchived-change rule")
    reads(p)
    p.add_argument("--all", action="store_true", help="fail on any change outside archive/ (integration branch)")
    p.add_argument("--draft", action="store_true", help="report unarchived related changes as warnings")
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
    p.add_argument("--taxonomy", action="store_true", help="print the trigger and managed label names")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_labels)
    return parser


def open_source(args, root: Path) -> Source | None:
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
        return SnapshotSource(doc)
    if base and head:
        if not (root / ".git").exists() and git(root, "rev-parse", "--git-dir", check=False).strip() == "":
            raise Usage(f"{root} is not a git work tree.")
        if args.command != "archive":
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
    changes_dir = args.changes_dir.strip("/")
    try:
        if args.command == "snapshot":
            return args.func(args, root, changes_dir)
        args.source = open_source(args, root)
        needs_no_source = (args.command == "check" and args.all) or (args.command == "labels" and args.taxonomy)
        if args.source is None and not needs_no_source:
            raise Usage(
                f"{args.command} needs a head source: --base with --head, or --snapshot from the snapshot command."
            )
        if args.command == "archive" and not isinstance(args.source, GitSource):
            raise Usage("archive edits the working tree, so it needs --base and --head, not a snapshot.")
        if args.command == "check" and args.all and args.source is not None:
            raise Usage("check --all reads the working tree; do not pass a head source with it.")
        return args.func(args, root, changes_dir)
    except Usage as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Failure as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
