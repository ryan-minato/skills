#!/usr/bin/env python3
"""Collect read-only evidence of the branching model a git repository uses.

Inspects the repository in --repo-dir (default: the current directory) and
reports the branch inventory grouped by naming shape, long-lived branch
candidates with their divergence from the default branch, tag shapes and
whether recent tags sit off the default line, the merge-commit ratio on the
default branch, the branch names referenced by CI configuration, and a ranked
list of candidate models with the reason for each.

The ranking is evidence, not a verdict: confirm it with the user before
recording a model. Read-only; makes no network calls and writes nothing.

Usage:
    python3 scripts/detect_branching.py [--repo-dir PATH] [--remote origin]
                                        [--stale-days 120] [--json] [--full]

Exit codes: 0 = report printed; 1 = not a git repository or git failed;
2 = bad arguments or git is not installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

ENVIRONMENT_NAMES = {
    "acceptance",
    "canary",
    "demo",
    "int",
    "live",
    "preprod",
    "pre-prod",
    "pre-production",
    "preproduction",
    "prod",
    "production",
    "qa",
    "sandbox",
    "stage",
    "staging",
    "uat",
}
INTEGRATION_NAMES = {"develop", "development", "integration", "next"}
TOPIC_PREFIXES = (
    "feature/",
    "feat/",
    "fix/",
    "bugfix/",
    "chore/",
    "docs/",
    "refactor/",
    "test/",
    "spike/",
    "wip/",
)
RELEASE_PREFIXES = ("release/", "releases/", "rel/", "hotfix/", "support/")
VERSION_BRANCH_RE = re.compile(r"^v?\d+([._-]\d+)*([._-](x|stable|maintenance|lts))?$", re.IGNORECASE)
STABLE_SUFFIX_RE = re.compile(r"[-_](stable|maintenance|lts)$", re.IGNORECASE)
SEMVER_TAG_RE = re.compile(r"^v?\d+\.\d+\.\d+([-+].*)?$")
CI_FILES = (
    ".gitlab-ci.yml",
    ".gitlab-ci.yaml",
    ".circleci/config.yml",
    "azure-pipelines.yml",
    "bitbucket-pipelines.yml",
    ".drone.yml",
    "Jenkinsfile",
)
CI_GLOB_DIRS = (".github/workflows",)
CI_READ_LIMIT = 512 * 1024
LIST_CAP = 20
SECONDS_PER_DAY = 86400


class GitError(Exception):
    """A git invocation failed in a way the caller should report and stop on."""


def git(repo_dir: str, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", repo_dir, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print(
            "git is not installed or not on PATH. Install git, then rerun.",
            file=sys.stderr,
        )
        sys.exit(2)
    if proc.returncode != 0:
        raise GitError(proc.stderr.strip() or " ".join(args))
    return proc.stdout


def git_ok(repo_dir: str, *args: str) -> str | None:
    try:
        return git(repo_dir, *args)
    except GitError:
        return None


def collect_branches(repo_dir: str, remote: str) -> dict[str, int]:
    """Map branch name to the newest commit date across local and remote refs."""
    branches: dict[str, int] = {}
    for ref_space in ("refs/heads", f"refs/remotes/{remote}"):
        out = git_ok(
            repo_dir,
            "for-each-ref",
            "--format=%(refname)\t%(refname:short)\t%(committerdate:unix)",
            ref_space,
        )
        if out is None:
            continue
        prefix = f"{remote}/"
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            full, name, stamp = parts
            # git shortens refs/remotes/<remote>/HEAD to the remote name itself.
            if full.endswith("/HEAD"):
                continue
            name = name.removeprefix(prefix)
            if not name:
                continue
            try:
                when = int(stamp)
            except ValueError:
                continue
            branches[name] = max(branches.get(name, 0), when)
    return branches


def resolve_default(repo_dir: str, remote: str, branches: dict[str, int]) -> str:
    head = git_ok(repo_dir, "symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD")
    if head:
        name = head.strip()
        prefix = f"{remote}/"
        name = name.removeprefix(prefix)
        if name:
            return name
    for candidate in ("main", "master", "trunk"):
        if candidate in branches:
            return candidate
    current = git_ok(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
    if current and current.strip() not in ("", "HEAD"):
        return current.strip()
    return next(iter(branches), "")


def classify(name: str) -> str:
    lowered = name.lower()
    leaf = lowered.rsplit("/", 1)[-1]
    if lowered in INTEGRATION_NAMES:
        return "integration"
    if leaf in ENVIRONMENT_NAMES or lowered in ENVIRONMENT_NAMES:
        return "environment"
    if lowered.startswith(RELEASE_PREFIXES):
        return "release"
    if VERSION_BRANCH_RE.match(leaf) or STABLE_SUFFIX_RE.search(leaf):
        return "release"
    if lowered.startswith(TOPIC_PREFIXES):
        return "topic"
    return "other"


def divergence(repo_dir: str, default: str, name: str) -> dict[str, int] | None:
    out = git_ok(repo_dir, "rev-list", "--left-right", "--count", f"{default}...{name}")
    if not out:
        return None
    parts = out.split()
    if len(parts) != 2:
        return None
    return {"behind_default": int(parts[0]), "ahead_of_default": int(parts[1])}


def tag_report(repo_dir: str, default: str) -> dict:
    out = git_ok(
        repo_dir,
        "for-each-ref",
        "--sort=-creatordate",
        "--format=%(refname:short)\t%(creatordate:unix)",
        "refs/tags",
    )
    names: list[str] = []
    if out:
        for line in out.splitlines():
            name = line.partition("\t")[0]
            if name:
                names.append(name)
    recent = names[:10]
    off_default = []
    for tag in recent:
        probe = subprocess.run(
            ["git", "-C", repo_dir, "merge-base", "--is-ancestor", tag, default],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 1:
            off_default.append(tag)
    return {
        "total": len(names),
        "recent": recent,
        "semver_share": round(sum(1 for t in recent if SEMVER_TAG_RE.match(t)) / len(recent), 2) if recent else 0.0,
        "off_default_line": off_default,
    }


def merge_ratio(repo_dir: str, default: str, sample: int) -> dict:
    # %H keeps root commits countable: %p alone renders them as a blank line.
    out = git_ok(repo_dir, "log", f"-n{sample}", "--pretty=%H %p", default)
    if out is None:
        return {"commits_sampled": 0, "merge_commits": 0, "merge_ratio": 0.0}
    lines = [line for line in out.splitlines() if line.strip()]
    merges = sum(1 for line in lines if len(line.split()) > 2)
    return {
        "commits_sampled": len(lines),
        "merge_commits": merges,
        "merge_ratio": round(merges / len(lines), 2) if lines else 0.0,
    }


def ci_references(repo_dir: str, candidates: list[str]) -> dict[str, list[str]]:
    paths = list(CI_FILES)
    for directory in CI_GLOB_DIRS:
        full = os.path.join(repo_dir, directory)
        if os.path.isdir(full):
            for entry in sorted(os.listdir(full)):
                if entry.endswith((".yml", ".yaml")):
                    paths.append(os.path.join(directory, entry))
    found: dict[str, list[str]] = {}
    for rel in paths:
        full = os.path.join(repo_dir, rel)
        if not os.path.isfile(full):
            continue
        try:
            with open(full, encoding="utf-8", errors="replace") as handle:
                text = handle.read(CI_READ_LIMIT)
        except OSError:
            continue
        hits = [name for name in candidates if re.search(rf"(?<![\w/-]){re.escape(name)}(?![\w/-])", text)]
        if hits:
            found[rel] = hits
    return found


def rank_models(evidence: dict) -> list[dict]:
    groups = evidence["groups"]
    active = {name for name, info in evidence["long_lived"].items() if not info["stale"]}
    integration = [n for n in groups.get("integration", []) if n in active]
    environments = [n for n in groups.get("environment", []) if n in active]
    releases = [n for n in groups.get("release", []) if n in active]
    ranked = []
    if integration and releases:
        ranked.append(
            {
                "model": "git flow",
                "reason": (
                    f"active integration branch {integration[0]!r} alongside "
                    f"{len(releases)} active release or hotfix branch(es)"
                ),
            }
        )
    if len(releases) >= 1 and evidence["tags"]["off_default_line"]:
        ranked.append(
            {
                "model": "GitLab Flow, release branches",
                "reason": (
                    f"{len(releases)} active release branch(es) and recent tags "
                    f"off the default line: "
                    f"{', '.join(evidence['tags']['off_default_line'][:3])}"
                ),
            }
        )
    elif len(releases) >= 2:
        ranked.append(
            {
                "model": "GitLab Flow, release branches",
                "reason": f"{len(releases)} active release branches: {', '.join(releases[:3])}",
            }
        )
    if len(environments) >= 2:
        ranked.append(
            {
                "model": "GitLab Flow, environment branches",
                "reason": f"{len(environments)} active environment branches: {', '.join(environments[:4])}",
            }
        )
    elif len(environments) == 1:
        ranked.append(
            {
                "model": "GitLab Flow, production branch",
                "reason": f"one active environment branch: {environments[0]}",
            }
        )
    if not ranked:
        ranked.append(
            {
                "model": "GitHub Flow",
                "reason": (
                    "no active long-lived branch besides the default branch; "
                    f"{len(groups.get('topic', []))} topic branch(es) present"
                ),
            }
        )
    return ranked


def build_report(args: argparse.Namespace) -> dict:
    repo_dir = args.repo_dir
    if not git_ok(repo_dir, "rev-parse", "--is-inside-work-tree"):
        raise GitError(f"{repo_dir!r} is not a git repository. Pass --repo-dir pointing at a git checkout.")
    branches = collect_branches(repo_dir, args.remote)
    if not branches:
        raise GitError("the repository has no branches. Commit something, or point --repo-dir at a populated checkout.")
    default = resolve_default(repo_dir, args.remote, branches)
    now = int(time.time())

    groups: dict[str, list[str]] = {}
    for name in sorted(branches):
        if name == default:
            continue
        groups.setdefault(classify(name), []).append(name)

    # A repository with thousands of topic branches would otherwise bury the
    # signal and risk the harness truncating stdout. --full lifts the cap.
    omitted: dict[str, int] = {}
    if not args.full:
        for kind, names in groups.items():
            if len(names) > LIST_CAP:
                omitted[kind] = len(names) - LIST_CAP
                groups[kind] = names[:LIST_CAP]

    long_lived: dict[str, dict] = {}
    watch = [name for kind in ("integration", "environment", "release", "other") for name in groups.get(kind, [])][:30]
    for name in watch:
        age_days = (now - branches[name]) // SECONDS_PER_DAY
        entry = {
            "kind": classify(name),
            "days_since_last_commit": age_days,
            "stale": age_days > args.stale_days,
        }
        div = divergence(repo_dir, default, name)
        if div:
            entry.update(div)
        long_lived[name] = entry

    evidence = {
        "repo_dir": os.path.abspath(repo_dir),
        "default_branch": default,
        "branch_count": len(branches),
        "groups": groups,
        "groups_omitted": omitted,
        "long_lived": long_lived,
        "tags": tag_report(repo_dir, default),
        "default_branch_history": merge_ratio(repo_dir, default, args.sample),
        "ci_branch_references": ci_references(repo_dir, sorted(set([*list(long_lived), default]))),
    }
    evidence["candidate_models"] = rank_models(evidence)
    return evidence


def render(report: dict) -> str:
    lines = [
        f"repository: {report['repo_dir']}",
        f"default branch: {report['default_branch']}",
        f"branches: {report['branch_count']}",
        "",
        "branch groups",
    ]
    for kind in ("integration", "environment", "release", "other", "topic"):
        names = report["groups"].get(kind, [])
        if not names:
            continue
        extra = report["groups_omitted"].get(kind, 0)
        suffix = f" (+{extra} more, use --full)" if extra else ""
        lines.append(f"  {kind}: {', '.join(names)}{suffix}")
    if not any(report["groups"].values()):
        lines.append("  none besides the default branch")

    lines += ["", "long-lived branch candidates"]
    if report["long_lived"]:
        for name, info in report["long_lived"].items():
            state = "stale" if info["stale"] else "active"
            div = ""
            if "ahead_of_default" in info:
                div = f", {info['ahead_of_default']} ahead / {info['behind_default']} behind default"
            lines.append(f"  {name} [{info['kind']}] {state}, last commit {info['days_since_last_commit']}d ago{div}")
    else:
        lines.append("  none")

    tags = report["tags"]
    lines += [
        "",
        f"tags: {tags['total']} total, semver share of recent: {tags['semver_share']}",
    ]
    if tags["recent"]:
        lines.append(f"  recent: {', '.join(tags['recent'])}")
    if tags["off_default_line"]:
        lines.append("  off the default line (parallel maintenance): " + ", ".join(tags["off_default_line"]))

    history = report["default_branch_history"]
    lines += [
        "",
        (
            f"default branch history: {history['merge_commits']} merge commits "
            f"in {history['commits_sampled']} sampled "
            f"(ratio {history['merge_ratio']})"
        ),
    ]

    lines += ["", "CI configuration referencing these branches"]
    if report["ci_branch_references"]:
        for path, names in report["ci_branch_references"].items():
            lines.append(f"  {path}: {', '.join(names)}")
    else:
        lines.append("  none found")

    lines += ["", "candidate models, most supported first"]
    for item in report["candidate_models"]:
        lines.append(f"  {item['model']} — {item['reason']}")
    lines += [
        "",
        ("This ranking is evidence, not a verdict. Confirm the model with the user before recording it."),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect read-only evidence of a repository's branching model.",
        epilog="Example: python3 scripts/detect_branching.py --repo-dir . --json",
    )
    parser.add_argument("--repo-dir", default=".", help="git checkout to inspect (default: .)")
    parser.add_argument(
        "--remote",
        default="origin",
        help="remote whose branches count (default: origin)",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=120,
        help="days without a commit after which a branch is an artifact (default: 120)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="commits sampled on the default branch for the merge ratio (default: 200)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--full",
        action="store_true",
        help="list every branch instead of capping groups",
    )
    args = parser.parse_args()
    if args.stale_days < 0:
        parser.error("--stale-days must be zero or greater")
    if args.sample < 1:
        parser.error("--sample must be at least 1")

    try:
        report = build_report(args)
    except GitError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
