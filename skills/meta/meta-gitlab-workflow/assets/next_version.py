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
- When the latest tag is a prerelease and --pre is absent, the bump
  finalizes it: the base version is printed and --bump is ignored (pass
  --latest with the last final release to bump past the base).
- Among prereleases of the same base version, the latest is picked by
  identifier (alphabetically — matching the common alpha < beta < rc)
  and then by counter; a final release always outranks its prereleases.

Exit codes: 0 = tag printed; 1 = no version tag in the repository (pass
--latest) or git failed; 2 = bad arguments.
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


def parse(tag: str) -> dict | None:
    match = SEMVER_RE.match(tag.strip())
    if not match:
        return None
    return {
        "tag": tag.strip(),
        "prefix": match["prefix"],
        "release": (int(match["major"]), int(match["minor"]), int(match["patch"])),
        "ident": match["ident"],
        "counter": int(match["counter"]) if match["counter"] else None,
    }


def version_tag(value: str) -> dict:
    """argparse type for --latest: a tag this script can parse."""
    parsed = parse(value)
    if parsed is None:
        raise argparse.ArgumentTypeError(f"{value!r} is not PREFIX + X.Y.Z[-IDENT.N]")
    return parsed


def latest_from_git() -> dict | None:
    """The highest version tag of the current repository, or None when it has none."""
    try:
        result = subprocess.run(["git", "tag", "--list"], capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("next_version: error: `git` is not installed; pass --latest instead.")
    if result.returncode != 0:
        sys.exit(f"next_version: error: `git tag --list` exited {result.returncode}: {result.stderr.strip()}")
    tags = [parsed for parsed in map(parse, result.stdout.splitlines()) if parsed]
    # A final release outranks its own prereleases; among prereleases,
    # compare the identifier alphabetically, then the counter.
    return max(
        tags,
        key=lambda t: (t["release"], t["counter"] is None, t["ident"] or "", t["counter"] or 0),
        default=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="next_version.py", description="Print the next semver tag (see the module docstring)."
    )
    parser.add_argument("--bump", required=True, choices=["major", "minor", "patch"])
    parser.add_argument("--latest", type=version_tag, help="current latest tag; else read the git tags")
    parser.add_argument("--prefix", help="tag prefix for the output, e.g. v")
    parser.add_argument("--pre", help="prerelease identifier, e.g. rc")
    args = parser.parse_args()

    current = args.latest or latest_from_git()
    if current is None:
        sys.exit("next_version: error: the repository has no version tag; pass --latest vX.Y.Z.")

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
    if args.pre and current["ident"] == args.pre:
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
