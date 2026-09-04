#!/usr/bin/env python3
"""Check a pull request against the PR template and the commit convention.

The template (.github/PULL_REQUEST_TEMPLATE.md) is the single source of the
required sections: its `## ` headings are parsed here, never restated. Run
from a checkout of the BASE commit so a pull request cannot weaken the
template it is checked against.

Checks, in order:
  1. every template heading appears verbatim in the body;
  2. the related-work section carries `Closes #N` (or Fixes/Resolves) or
     `N/A — <reason>`;
  3. the checklist line containing the security keyword ("secrets") exists;
  4. commit subjects follow Conventional Commits — over BASE..HEAD for a
     branch in this repository, over the pull request title for a fork;
  5. for a ready (non-draft) pull request: the validation section is not
     empty, every checklist box is ticked, and a `Spec:` line names an
     OpenSpec change (`openspec/changes/<slug>`) or `none — <reason>`.

Usage:
    python3 scripts/check_pr_policy.py --event "$GITHUB_EVENT_PATH"
    python3 scripts/check_pr_policy.py --pr 123        # local dry run via gh

Findings go to stdout, one per line, each ending with the file to fix.
Exit codes: 0 = compliant, 1 = findings, 2 = bad input, unreadable
template, or git failure (a shallow checkout cannot list the range).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TEMPLATE = Path(".github/PULL_REQUEST_TEMPLATE.md")
SECURITY_KEYWORD = "secrets"
COMMIT_TYPES = "feat|fix|docs|refactor|chore|test|ci|perf|build|style|revert"
SUBJECT_RE = re.compile(rf"^({COMMIT_TYPES})(\([a-z0-9][a-z0-9, ._-]*\))?!?: [a-z0-9].+$")
EXEMPT_SUBJECT_RE = re.compile(r"^(Merge |Revert |fixup!|squash!)")
CLOSING_RE = re.compile(r"\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\s+#\d+\b", re.IGNORECASE)
NO_ISSUE_RE = re.compile(r"^N/A\s+[—-]\s+\S", re.IGNORECASE | re.MULTILINE)
# A change record is named as a clickable link to its directory on the branch
# (`[openspec/changes/<slug>](https://github.com/<owner>/<repo>/tree/<ref>/openspec/changes/<slug>)`),
# as a bare URL, or — accepted for older pull requests — as a bare path.
SPEC_PATH = r"openspec/changes/(archive/\d{4}-\d{2}-\d{2}-)?[a-z0-9][a-z0-9-]*/?"
SPEC_URL = rf"https://github\.com/[\w.-]+/[\w.-]+/(tree|blob)/[^\s)]+?/{SPEC_PATH}"
SPEC_RE = re.compile(
    rf"^Spec:\s*(\[{SPEC_PATH}\]\({SPEC_URL}\)|{SPEC_URL}|{SPEC_PATH}|none\s+[—-]\s+\S.*)\s*$",
    re.MULTILINE,
)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_RE = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
CHECKBOX_RE = re.compile(r"^\s*- \[( |x|X)\]\s*(.*)$")
BOT_AUTHORS = {"dependabot[bot]"}


def fail(code: int, message: str) -> None:
    print(f"check_pr_policy: error: {message}", file=sys.stderr)
    sys.exit(code)


def strip_comments(text: str) -> str:
    return COMMENT_RE.sub("", text or "")


def load_event(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail(2, f"cannot read event payload {path}: {exc}")
    pr = payload.get("pull_request")
    if not isinstance(pr, dict):
        fail(2, f"{path} is not a pull_request event payload")
    head = pr.get("head") or {}
    base = pr.get("base") or {}
    return {
        "number": pr.get("number"),
        "title": pr.get("title") or "",
        "body": pr.get("body") or "",
        "draft": bool(pr.get("draft")),
        "fork": bool((head.get("repo") or {}).get("fork")),
        "base_sha": base.get("sha"),
        "head_sha": head.get("sha"),
        "author": (pr.get("user") or {}).get("login") or "",
    }


def load_pr_via_gh(number: int) -> dict:
    fields = "number,title,body,isDraft,isCrossRepository,baseRefOid,headRefOid,author"
    try:
        out = subprocess.run(
            ["gh", "pr", "view", str(number), "--json", fields],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(2, f"gh pr view failed: {exc}")
    data = json.loads(out)
    return {
        "number": data["number"],
        "title": data.get("title") or "",
        "body": data.get("body") or "",
        "draft": bool(data.get("isDraft")),
        "fork": bool(data.get("isCrossRepository")),
        "base_sha": data.get("baseRefOid"),
        "head_sha": data.get("headRefOid"),
        "author": (data.get("author") or {}).get("login") or "",
    }


def template_headings(path: Path) -> tuple[list[str], dict[str, str]]:
    try:
        text = strip_comments(path.read_text(encoding="utf-8"))
    except OSError as exc:
        fail(2, f"cannot read {path}: {exc}")
    headings = HEADING_RE.findall(text)
    if not headings:
        fail(2, f"{path} has no '## ' headings to enforce; fix the template")
    roles: dict[str, str] = {}
    for heading in headings:
        lower = heading.lower()
        for role in ("related", "validation", "checklist"):
            if role in lower and role not in roles:
                roles[role] = heading
    missing = [role for role in ("related", "validation", "checklist") if role not in roles]
    if missing:
        fail(2, f"{path} lacks a heading for: {', '.join(missing)}; fix the template")
    return headings, roles


def body_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = None
    for line in body.splitlines():
        match = HEADING_RE.match(line)
        if match:
            current = match.group(1)
            sections[current] = ""
        elif current is not None:
            sections[current] += line + "\n"
    return sections


def commit_subjects(base: str, head: str) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "log", "--format=%s", f"{base}..{head}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(2, f"git log {base}..{head} failed (shallow checkout or missing fetch?): {exc}")
    return [line for line in out.splitlines() if line.strip()]


def check_subject(subject: str, label: str, findings: list[str]) -> None:
    if EXEMPT_SUBJECT_RE.match(subject):
        return
    if not SUBJECT_RE.match(subject):
        findings.append(
            f"{label} {subject!r} is not a Conventional Commit subject "
            f"(`type(scope): subject`, types {COMMIT_TYPES}). Fix: rewrite it; see AGENTS.md Commits."
        )


def check(pr: dict, template: Path) -> list[str]:
    findings: list[str] = []
    headings, roles = template_headings(template)
    body = strip_comments(pr["body"])
    ready = not pr["draft"]

    if pr["author"] not in BOT_AUTHORS:
        sections = body_sections(body)
        for heading in headings:
            if heading not in sections:
                findings.append(f"PR body is missing the section '## {heading}'. Fix: restore it from {template}.")
        related = sections.get(roles["related"], "")
        if not (CLOSING_RE.search(related) or NO_ISSUE_RE.search(related)):
            findings.append(
                f"'## {roles['related']}' needs `Closes #N` (or Fixes/Resolves) or `N/A — <reason>`. "
                "Fix: the PR description."
            )
        checklist_lines = [CHECKBOX_RE.match(line) for line in sections.get(roles["checklist"], "").splitlines()]
        checklist = [m for m in checklist_lines if m]
        security = [m for m in checklist if SECURITY_KEYWORD in m.group(2).lower()]
        if not security:
            findings.append(
                f"The checklist line containing '{SECURITY_KEYWORD}' is missing. "
                f"Fix: restore it from {template} and tick it."
            )
        if ready:
            validation = sections.get(roles["validation"], "")
            if not re.sub(r"^\s*-\s*$", "", validation, flags=re.MULTILINE).strip():
                findings.append(
                    f"'## {roles['validation']}' is empty on a ready pull request. "
                    "Fix: record what you ran in the PR description."
                )
            unticked = [m.group(2) for m in checklist if m.group(1) == " "]
            for item in unticked:
                findings.append(
                    f"Checklist item not ticked on a ready pull request: {item!r}. Fix: complete it and tick it."
                )
            if not SPEC_RE.search(body):
                findings.append(
                    "A ready pull request needs a `Spec: openspec/changes/<slug>` line or `Spec: none — <reason>`. "
                    "Fix: the PR description."
                )

    if pr["fork"]:
        check_subject(pr["title"], "Pull request title (fork PRs are squash-merged)", findings)
    else:
        if not (pr["base_sha"] and pr["head_sha"]):
            fail(2, "base and head SHAs are required for the commit-range check")
        for subject in commit_subjects(pr["base_sha"], pr["head_sha"]):
            check_subject(subject, "Commit subject", findings)
    return findings


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event", help="path to the pull_request event payload (GITHUB_EVENT_PATH)")
    source.add_argument("--pr", type=int, help="pull request number to fetch with gh (local dry run)")
    parser.add_argument("--template", default=str(TEMPLATE), help=f"PR template path (default {TEMPLATE})")
    args = parser.parse_args(argv)

    pr = load_event(args.event) if args.event else load_pr_via_gh(args.pr)
    findings = check(pr, Path(args.template))
    for finding in findings:
        print(finding)
    if findings:
        print(f"{len(findings)} finding(s) for pull request #{pr['number']}.", file=sys.stderr)
        return 1
    print(f"Pull request #{pr['number']} satisfies the template and commit policy.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
