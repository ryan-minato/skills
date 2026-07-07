#!/usr/bin/env python3
"""Compute the next semver tag for a release.

Reads the latest version from --latest, or from the git tags of the
current directory's repository when --latest is omitted, then applies the
requested bump and prints the next tag to stdout (nothing else).

Usage:
    python3 scripts/next_version.py --bump patch
    python3 scripts/next_version.py --bump minor --pre rc
    python3 scripts/next_version.py --bump major --latest v2.9.3 --prefix v

Rules:
- Tags are matched as PREFIX + MAJOR.MINOR.PATCH with an optional
  -IDENT.N prerelease suffix; other tags are ignored.
- --prefix defaults to the latest tag's own prefix ("v" or none).
- --bump major|minor|patch resets the lower parts to zero.
- --pre IDENT appends -IDENT.1 to the bumped version. When the latest tag
  is already a prerelease with the same identifier, the series continues
  toward its base version instead: the counter is incremented and --bump
  is ignored (pass --latest with the last final release to start a new
  series at a different version).
- A prerelease latest (e.g. v1.3.0-rc.2) finalizes to its base version
  when bumped with the part that produced it (patch keeps 1.3.0).

Exit codes: 0 = tag printed; 1 = no parseable version tag found (pass
--latest explicitly); 2 = bad arguments or git unavailable.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

SEMVER_RE = re.compile(
    r"^(?P<prefix>[A-Za-z]*)"
    r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<ident>[0-9A-Za-z]+)\.(?P<counter>\d+))?$"
)


def parse(tag: str):
    match = SEMVER_RE.match(tag.strip())
    if not match:
        return None
    return {
        "prefix": match.group("prefix"),
        "release": (
            int(match.group("major")),
            int(match.group("minor")),
            int(match.group("patch")),
        ),
        "ident": match.group("ident"),
        "counter": int(match.group("counter")) if match.group("counter") else None,
    }


def latest_from_git() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "tag", "--list"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        print("git is not installed or not on PATH", file=sys.stderr)
        sys.exit(2)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(2)
    best_tag, best_key = None, None
    for tag in proc.stdout.splitlines():
        parsed = parse(tag)
        if parsed is None:
            continue
        # A final release outranks its own prereleases.
        key = (
            parsed["release"],
            parsed["counter"] is None,
            parsed["counter"] or 0,
        )
        if best_key is None or key > best_key:
            best_tag, best_key = tag.strip(), key
    return best_tag


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="next_version.py",
        description="Print the next semver tag (see module docstring)",
    )
    parser.add_argument(
        "--bump", required=True, choices=["major", "minor", "patch"]
    )
    parser.add_argument("--latest", help="current latest tag; else read git tags")
    parser.add_argument("--prefix", help="tag prefix for the output, e.g. v")
    parser.add_argument("--pre", help="prerelease identifier, e.g. rc")
    args = parser.parse_args()

    latest = args.latest or latest_from_git()
    if latest is None:
        print(
            "no parseable version tag found; pass --latest vX.Y.Z",
            file=sys.stderr,
        )
        sys.exit(1)
    current = parse(latest)
    if current is None:
        print(f"cannot parse {latest!r} as PREFIXX.Y.Z[-ident.N]", file=sys.stderr)
        sys.exit(1 if args.latest is None else 2)

    major, minor, patch = current["release"]
    if args.bump == "major":
        nxt = (major + 1, 0, 0)
    elif args.bump == "minor":
        nxt = (major, minor + 1, 0)
    else:
        nxt = (major, minor, patch + 1)

    # Finalizing a prerelease: keep its base instead of bumping past it.
    if current["ident"] is not None and args.pre is None:
        nxt = current["release"]

    counter = 1
    if args.pre and current["ident"] == args.pre and current["counter"] is not None:
        # Continue the running prerelease series toward its base version.
        nxt = current["release"]
        counter = current["counter"] + 1

    prefix = args.prefix if args.prefix is not None else current["prefix"]
    version = f"{prefix}{nxt[0]}.{nxt[1]}.{nxt[2]}"
    if args.pre:
        version = f"{version}-{args.pre}.{counter}"

    print(version)


if __name__ == "__main__":
    main()
