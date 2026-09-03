#!/usr/bin/env python3
"""Find and (on request) delete installed disposable skills.

Default run is a dry run: it lists every skill one level below each root
whose resolved frontmatter description starts with the marker, plus every
entry it had to skip. Nothing is deleted without --delete and the confirmation
token printed by a fresh dry run. The script never prompts; human confirmation
belongs in the conversation between the two runs.

The marker is not embedded here: it is read from this skill's own SKILL.md
description (the prefix through the first "):"), so the script always
matches the marker generation it was shipped with — the one sentence every
disposable builder of the meta and scaffold catalogs opens with. --marker
exists only to recover when this skill's own SKILL.md was damaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

SELF_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = SELF_DIR.parent
MARKER_END = "):"
# A real marker is a ~60-character sentence prefix; anything much shorter
# is a dangerously broad startswith() net. Deliberately NOT validating a
# "):"" suffix: --marker exists to recover across marker generations, and a
# future generation may not share the current shape.
MIN_MARKER_LEN = 16


class FrontmatterError(ValueError):
    pass


def parse_frontmatter(text: str) -> dict[str, str]:
    """Parse a SKILL.md frontmatter block into string fields.

    Tries PyYAML when available; otherwise a subset parser covering plain
    scalars, single/double-quoted scalars, and >, >-, |, |- block scalars
    for top-level keys. Anything else raises FrontmatterError so the
    caller can skip-and-report instead of guessing.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise FrontmatterError("no frontmatter block")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise FrontmatterError("unterminated frontmatter block") from None
    block = "\n".join(lines[1:end])

    try:
        import yaml  # type: ignore
    except ImportError:
        pass
    else:
        try:
            data = yaml.safe_load(block)
        except yaml.YAMLError as exc:
            raise FrontmatterError(f"YAML parse failed: {exc}") from None
        if not isinstance(data, dict):
            raise FrontmatterError("frontmatter is not a mapping")
        return {str(k): v for k, v in data.items() if isinstance(v, str)}

    return _parse_subset(block.splitlines())


def _parse_subset(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t":
            raise FrontmatterError(f"unexpected indented line: {line!r}")
        key, sep, rest = line.partition(":")
        if not sep or not key.strip():
            raise FrontmatterError(f"not a key-value line: {line!r}")
        key = key.strip()
        rest = rest.strip()
        if rest in (">", ">-", ">+", "|", "|-", "|+"):
            folded = rest[0] == ">"
            body: list[str] = []
            while i < len(lines) and (not lines[i].strip() or lines[i][0] in " \t"):
                body.append(lines[i].strip())
                i += 1
            while body and not body[-1]:
                body.pop()
            fields[key] = (" " if folded else "\n").join(body)
        elif len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in "'\"":
            fields[key] = rest[1:-1]
        elif rest.startswith(("[", "{")):
            raise FrontmatterError(f"unsupported or malformed collection value for {key!r}: {rest!r}")
        else:
            fields[key] = rest
    return fields


def marker_from_self() -> str:
    skill_md = SELF_DIR / "SKILL.md"
    try:
        fields = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, FrontmatterError) as exc:
        sys.exit(
            f"error: cannot read the marker from {skill_md}: {exc}\n"
            "Pass --marker with the exact marker text to recover."
        )
    description = fields.get("description", "")
    end = description.find(MARKER_END)
    if end < 0:
        sys.exit(
            f"error: no marker prefix in the description of {skill_md}.\n"
            "Pass --marker with the exact marker text to recover."
        )
    return description[: end + len(MARKER_END)]


def in_plugin_cache(path: Path) -> bool:
    """True when the path sits inside a plugin manager's cache.

    Plugin-managed installs (e.g. a plugin marketplace) keep their payload
    under <config>/plugins/cache/...; deleting files there leaves the
    manager restoring or mis-tracking the plugin. Those installs are
    removed by uninstalling through the manager, never by this script.
    """
    parts = path.parts
    return any(parts[i] == "plugins" and parts[i + 1] == "cache" for i in range(len(parts) - 1))


def scan(roots: list[Path], marker: str):
    matches: list[tuple[Path, str, str]] = []
    skipped: list[tuple[Path, str]] = []
    for root in roots:
        if not root.is_dir():
            skipped.append((root, "root does not exist or is not a directory"))
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_symlink():
                skipped.append((entry, "symlink — resolve and scan its real root"))
                continue
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                fields = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, FrontmatterError) as exc:
                skipped.append((skill_md, f"unparsable frontmatter: {exc}"))
                continue
            description = fields.get("description")
            if not isinstance(description, str):
                skipped.append((skill_md, "no string description field"))
                continue
            if description.startswith(marker):
                first_line = description.splitlines()[0] if description else ""
                matches.append((entry, fields.get("name", entry.name), first_line))
    return matches, skipped


def inventory_token(
    roots: list[Path],
    marker: str,
    matches: list[tuple[Path, str, str]],
    skipped: list[tuple[Path, str]],
) -> str:
    """Bind deletion approval to the complete displayed scan result."""
    payload = {
        "roots": [str(path) for path in roots],
        "marker": marker,
        "matches": [
            {"path": str(path), "name": name, "description": description} for path, name, description in matches
        ],
        "skipped": [{"path": str(path), "reason": reason} for path, reason in skipped],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=None,
        help=f"a skill root to scan; repeatable (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="delete matched skill directories after --confirm TOKEN; this skill's own directory is deleted last",
    )
    parser.add_argument(
        "--confirm",
        metavar="TOKEN",
        help="exact inventory token printed by the fresh dry run",
    )
    parser.add_argument(
        "--marker",
        default=None,
        help="override the marker text; recovery use only, when this skill's own SKILL.md cannot be read",
    )
    args = parser.parse_args()

    roots = list(dict.fromkeys(r.resolve() for r in (args.root or [DEFAULT_ROOT])))
    marker = args.marker if args.marker is not None else marker_from_self()
    if len(marker.strip()) < MIN_MARKER_LEN:
        sys.exit(
            f"error: refusing marker {marker!r} — shorter than "
            f"{MIN_MARKER_LEN} characters, which would match far more than "
            "the disposable skills. Pass the full marker text."
        )

    cache_paths = [p for p in [*roots, SELF_DIR] if in_plugin_cache(p)]
    if cache_paths:
        note = (
            "plugin-manager cache detected "
            f"({', '.join(str(p) for p in dict.fromkeys(cache_paths))}): "
            "plugin-managed installs are removed by uninstalling through "
            "the plugin manager, not by deleting files."
        )
        if args.delete:
            sys.exit(f"error: {note}")
        print(f"warning: {note}")

    matches, skipped = scan(roots, marker)
    token = inventory_token(roots, marker, matches, skipped)  # pragma: allowlist secret

    mode = "DELETING" if args.delete else "DRY RUN (nothing deleted)"
    print(f"{mode} — scanned root(s): {', '.join(str(r) for r in roots)}")
    print(f"matched {len(matches)} skill(s):")
    for path, name, first_line in matches:
        print(f"  {path}\n    name: {name}\n    description: {first_line}")
    if skipped:
        print(f"skipped {len(skipped)} entr(y/ies) — verify these by hand:")
        for path, reason in skipped:
            print(f"  {path}: {reason}")
    print(f"confirmation token: {token}")  # pragma: allowlist secret

    if not args.delete:
        return

    if args.confirm is None:
        parser.error("--delete requires --confirm TOKEN from a fresh dry run")
    if args.confirm != token:
        parser.error(
            "confirmation token does not match the current scan; run a new dry "
            "run, review its exact listing, and confirm that token"
        )

    failures = 0
    deleted = 0
    self_matches = [match for match in matches if match[0].resolve() == SELF_DIR]
    other_matches = [match for match in matches if match not in self_matches]
    for path, _name, _line in other_matches:
        try:
            shutil.rmtree(path)
            print(f"deleted {path}")
            deleted += 1
        except OSError as exc:
            failures += 1
            print(f"FAILED to delete {path}: {exc}")

    if failures and self_matches:
        print(f"retained {SELF_DIR}: another deletion failed, so the recovery tool remains available")
    else:
        for path, _name, _line in self_matches:
            try:
                shutil.rmtree(path)
                print(f"deleted {path}")
                deleted += 1
            except OSError as exc:
                failures += 1
                print(f"FAILED to delete {path}: {exc}")
    print(f"done: {deleted} deleted, {failures} failed, {len(skipped)} skipped.")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
