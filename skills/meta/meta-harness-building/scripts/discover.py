#!/usr/bin/env python3
"""Discover disposable builders from the live ryan-minato/skills repository.

Examples:
  python3 scripts/discover.py
  python3 scripts/discover.py --catalog meta --full
  python3 scripts/discover.py --skill meta/meta-git-branching --full
  python3 scripts/discover.py --full --output /tmp/meta-skills.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ARCHIVE_URL = "https://codeload.github.com/ryan-minato/skills/tar.gz/refs/heads/main"
MANIFEST_PATH = ".claude-plugin/marketplace.json"
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_REPOSITORY_FILE_BYTES = 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 30
SUMMARY_LIMIT = 100
MARKER_END = "):"
MIN_MARKER_LENGTH = 16
CATALOG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class DiscoveryError(RuntimeError):
    """A source or inventory error that should be reported without a traceback."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str


@dataclass(frozen=True)
class Catalog:
    name: str
    description: str
    skills: tuple[Skill, ...]


@dataclass(frozen=True)
class CatalogSpec:
    name: str
    description: str
    skill_names: tuple[str, ...]


def _string_field(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiscoveryError(
            f"marketplace field `{field}` must be a non-empty string. "
            "Fix the live marketplace manifest."
        )
    return value


def parse_manifest(text: str) -> tuple[CatalogSpec, ...]:
    """Parse and validate the repository-owned marketplace inventory."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DiscoveryError(
            f"the marketplace manifest is not valid JSON ({exc}). "
            "Fix the live manifest and retry."
        ) from None
    if not isinstance(data, dict) or not isinstance(data.get("plugins"), list):
        raise DiscoveryError(
            "the marketplace manifest must contain a `plugins` array. "
            "Fix the live manifest and retry."
        )

    catalogs: list[CatalogSpec] = []
    seen_catalogs: set[str] = set()
    seen_skill_ids: set[str] = set()
    for index, raw in enumerate(data["plugins"]):
        if not isinstance(raw, dict):
            raise DiscoveryError(
                f"marketplace plugin #{index + 1} must be an object. "
                "Fix the live manifest and retry."
            )
        name = _string_field(raw.get("name"), f"plugins[{index}].name")
        if not CATALOG_RE.fullmatch(name):
            raise DiscoveryError(
                f"marketplace catalog `{name}` is not a canonical catalog name. "
                "Use lowercase letters, digits, and single hyphens."
            )
        if name in seen_catalogs:
            raise DiscoveryError(
                f"marketplace catalog `{name}` appears more than once. "
                "Remove the duplicate entry."
            )
        seen_catalogs.add(name)

        description = _string_field(
            raw.get("description"), f"plugins[{index}].description"
        )
        expected_source = "./"
        source = _string_field(raw.get("source"), f"plugins[{index}].source")
        if source != expected_source:
            raise DiscoveryError(
                f"catalog `{name}` has source `{source}`, expected "
                f"`{expected_source}`. Fix the marketplace manifest."
            )

        raw_skills = raw.get("skills")
        if not isinstance(raw_skills, list):
            raise DiscoveryError(
                f"catalog `{name}` must have an explicit `skills` array. "
                "Fix the marketplace manifest."
            )
        skill_names: list[str] = []
        for skill_index, raw_skill in enumerate(raw_skills):
            skill_ref = _string_field(
                raw_skill, f"plugins[{index}].skills[{skill_index}]"
            )
            expected_prefix = f"./skills/{name}/"
            if not skill_ref.startswith(expected_prefix):
                raise DiscoveryError(
                    f"catalog `{name}` skill path `{skill_ref}` must start with "
                    f"`{expected_prefix}` and name one direct skill directory."
                )
            skill_name = skill_ref[len(expected_prefix) :]
            if "/" in skill_name or not SKILL_RE.fullmatch(skill_name):
                raise DiscoveryError(
                    f"catalog `{name}` skill path `{skill_ref}` must name one "
                    "direct skill directory; path traversal and nested "
                    "paths are forbidden."
                )
            skill_id = f"{name}/{skill_name}"
            if skill_id in seen_skill_ids:
                raise DiscoveryError(
                    f"skill `{skill_id}` appears more than once in the marketplace. "
                    "Remove the duplicate entry."
                )
            seen_skill_ids.add(skill_id)
            skill_names.append(skill_name)
        catalogs.append(CatalogSpec(name, description, tuple(skill_names)))

    return tuple(catalogs)


def parse_frontmatter(text: str, source: str) -> dict[str, str]:
    """Parse the top-level string fields used by published SKILL.md files."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise DiscoveryError(
            f"`{source}` has no YAML frontmatter. Fix the live skill and retry."
        )
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise DiscoveryError(
            f"`{source}` has unterminated YAML frontmatter. "
            "Fix the live skill and retry."
        ) from None

    fields: dict[str, str] = {}
    frontmatter = lines[1:end]
    index = 0
    while index < len(frontmatter):
        line = frontmatter[index]
        index += 1
        if not line.strip() or line.lstrip().startswith("#") or line[0].isspace():
            continue
        key, separator, raw_value = line.partition(":")
        if not separator:
            raise DiscoveryError(
                f"`{source}` has an invalid top-level frontmatter line "
                f"`{line}`. Fix the live skill and retry."
            )
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in {"name", "description"}:
            continue
        if key in fields:
            raise DiscoveryError(
                f"`{source}` repeats frontmatter field `{key}`. Keep exactly one value."
            )

        if raw_value in {">", ">-", ">+", "|", "|-", "|+"}:
            body: list[str] = []
            while index < len(frontmatter):
                next_line = frontmatter[index]
                if next_line and not next_line[0].isspace():
                    break
                body.append(next_line)
                index += 1
            fields[key] = _parse_block_scalar(body, folded=raw_value[0] == ">")
        elif len(raw_value) >= 2 and raw_value[0] == raw_value[-1] == '"':
            try:
                decoded = json.loads(raw_value)
            except json.JSONDecodeError as exc:
                raise DiscoveryError(
                    f"`{source}` has an invalid quoted `{key}` value ({exc}). "
                    "Fix the live skill and retry."
                ) from None
            if not isinstance(decoded, str):
                raise DiscoveryError(f"`{source}` field `{key}` must be a string.")
            fields[key] = decoded
        elif len(raw_value) >= 2 and raw_value[0] == raw_value[-1] == "'":
            fields[key] = raw_value[1:-1].replace("''", "'")
        elif raw_value:
            fields[key] = raw_value
        else:
            raise DiscoveryError(
                f"`{source}` field `{key}` must be a string. "
                "Fix the live skill and retry."
            )

    for required in ("name", "description"):
        if not fields.get(required):
            raise DiscoveryError(
                f"`{source}` has no non-empty string `{required}` field. "
                "Fix the live skill and retry."
            )
    return fields


def _parse_block_scalar(lines: list[str], folded: bool) -> str:
    nonempty = [line for line in lines if line.strip()]
    indentation = (
        min(len(line) - len(line.lstrip()) for line in nonempty) if nonempty else 0
    )
    values = [line[indentation:].rstrip() if line.strip() else "" for line in lines]
    while values and not values[-1]:
        values.pop()
    if not folded:
        return "\n".join(values)

    paragraphs: list[str] = []
    current: list[str] = []
    for value in values:
        if value:
            current.append(value)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n".join(paragraphs)


def _safe_local_reader(repo_root: Path) -> Callable[[str], str]:
    try:
        root = repo_root.resolve(strict=True)
    except OSError as exc:
        raise DiscoveryError(
            f"cannot resolve local repository `{repo_root}` ({exc}). "
            "Pass an existing repository root."
        ) from None
    if not root.is_dir():
        raise DiscoveryError(
            f"local repository `{root}` is not a directory. Pass the repository root."
        )

    def read(relative: str) -> str:
        try:
            candidate = (root / relative).resolve(strict=True)
        except OSError as exc:
            raise DiscoveryError(
                f"cannot read repository file `{relative}` ({exc}). "
                "Restore the file named by the marketplace manifest."
            ) from None
        try:
            candidate.relative_to(root)
        except ValueError:
            raise DiscoveryError(
                f"repository file `{relative}` resolves outside `{root}`. "
                "Remove the escaping symlink or path."
            ) from None
        if not candidate.is_file():
            raise DiscoveryError(
                f"repository path `{relative}` is not a file. "
                "Restore the file named by the marketplace manifest."
            )
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            raise DiscoveryError(
                f"cannot read repository file `{relative}` ({exc})."
            ) from None
        return _decode_repository_file(data, relative)

    return read


def _download_archive() -> bytes:
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "meta-skill-discovery"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=NETWORK_TIMEOUT_SECONDS
        ) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_ARCHIVE_BYTES:
                raise DiscoveryError(
                    f"the live repository archive is larger than "
                    f"{MAX_ARCHIVE_BYTES} bytes. Refusing the unexpected payload."
                )
            data = response.read(MAX_ARCHIVE_BYTES + 1)
    except DiscoveryError:
        raise
    except urllib.error.HTTPError as exc:
        raise DiscoveryError(
            f"GitHub returned HTTP {exc.code} for the live repository archive. "
            "Check repository availability and retry; no cached inventory is used."
        ) from None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        raise DiscoveryError(
            f"cannot fetch the live repository archive ({reason}). "
            "Check network access and retry; no cached inventory is used."
        ) from None
    if len(data) > MAX_ARCHIVE_BYTES:
        raise DiscoveryError(
            f"the live repository archive exceeds {MAX_ARCHIVE_BYTES} bytes. "
            "Refusing the unexpected payload."
        )
    return data


def _archive_reader(data: bytes) -> tuple[Callable[[str], str], tarfile.TarFile]:
    try:
        # The caller owns and closes the returned archive after using the reader.
        archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")  # noqa: SIM115
    except tarfile.TarError as exc:
        raise DiscoveryError(
            f"the live repository response is not a valid tar.gz archive ({exc}). "
            "Retry after checking GitHub availability."
        ) from None

    manifest_suffix = f"/{MANIFEST_PATH}"
    manifest_members = [
        member
        for member in archive.getmembers()
        if member.isfile() and member.name.endswith(manifest_suffix)
    ]
    if len(manifest_members) != 1:
        archive.close()
        raise DiscoveryError(
            "the live repository archive must contain exactly one marketplace "
            "manifest. Check the repository layout."
        )
    manifest_name = manifest_members[0].name
    prefix = manifest_name[: -len(MANIFEST_PATH)]

    def read(relative: str) -> str:
        _validate_relative_path(relative)
        member_name = f"{prefix}{relative}"
        try:
            member = archive.getmember(member_name)
        except KeyError:
            raise DiscoveryError(
                f"the live archive is missing `{relative}`. "
                "Restore the file named by the marketplace manifest."
            ) from None
        if not member.isfile():
            raise DiscoveryError(
                f"live archive path `{relative}` is not a file. "
                "Restore the file named by the marketplace manifest."
            )
        if member.size > MAX_REPOSITORY_FILE_BYTES:
            raise DiscoveryError(
                f"live repository file `{relative}` exceeds "
                f"{MAX_REPOSITORY_FILE_BYTES} bytes. Refusing the unexpected file."
            )
        extracted = archive.extractfile(member)
        if extracted is None:
            raise DiscoveryError(f"cannot read live repository file `{relative}`.")
        return _decode_repository_file(extracted.read(), relative)

    return read, archive


def _validate_relative_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DiscoveryError(
            f"repository path `{relative}` is unsafe. "
            "Only normalized repository-relative paths are allowed."
        )


def _decode_repository_file(data: bytes, relative: str) -> str:
    if len(data) > MAX_REPOSITORY_FILE_BYTES:
        raise DiscoveryError(
            f"repository file `{relative}` exceeds {MAX_REPOSITORY_FILE_BYTES} "
            "bytes. Refusing the unexpected file."
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise DiscoveryError(
            f"repository file `{relative}` is not UTF-8 text. "
            "Fix the live source and retry."
        ) from None


def load_inventory(repo_root: Path | None = None) -> tuple[Catalog, ...]:
    """Load one internally consistent repository snapshot."""
    archive: tarfile.TarFile | None = None
    if repo_root is None:
        read, archive = _archive_reader(_download_archive())
    else:
        read = _safe_local_reader(repo_root)
    try:
        specs = parse_manifest(read(MANIFEST_PATH))
        catalogs: list[Catalog] = []
        marker: str | None = None
        for spec in specs:
            if spec.name != "meta":
                continue
            skills: list[Skill] = []
            for skill_name in spec.skill_names:
                relative = f"skills/{spec.name}/{skill_name}/SKILL.md"
                fields = parse_frontmatter(read(relative), relative)
                if fields["name"] != skill_name:
                    raise DiscoveryError(
                        f"`{relative}` declares name `{fields['name']}`, expected "
                        f"`{skill_name}` from its marketplace path."
                    )
                skill_marker = _description_marker(fields["description"], relative)
                if marker is None:
                    marker = skill_marker
                elif marker != skill_marker:
                    raise DiscoveryError(
                        f"`{relative}` uses a different disposal marker. "
                        "Repair the marker contract before listing skills."
                    )
                skills.append(
                    Skill(
                        name=skill_name,
                        description=fields["description"][len(skill_marker) :].strip(),
                    )
                )
            catalogs.append(Catalog(spec.name, spec.description, tuple(skills)))
        return tuple(catalogs)
    finally:
        if archive is not None:
            archive.close()


def _description_marker(description: str, relative: str) -> str:
    end = description.find(MARKER_END)
    if end < 0:
        raise DiscoveryError(
            f"`{relative}` description has no disposal marker terminator. "
            "Repair the marker contract before listing skills."
        )
    marker = description[: end + len(MARKER_END)]
    if (
        len(marker) < MIN_MARKER_LENGTH
        or len(description) == len(marker)
        or description[len(marker)] != " "
    ):
        raise DiscoveryError(
            f"`{relative}` has an invalid disposal marker prefix. "
            "Repair the marker contract before listing skills."
        )
    return marker


def select_inventory(
    catalogs: tuple[Catalog, ...],
    catalog_name: str | None,
    skill_id: str | None,
) -> tuple[Catalog, ...]:
    by_name = {catalog.name: catalog for catalog in catalogs}
    if catalog_name is not None:
        catalog = by_name.get(catalog_name)
        if catalog is None:
            valid = ", ".join(by_name) or "(none)"
            raise DiscoveryError(
                f"unknown catalog `{catalog_name}`. Valid catalogs: {valid}."
            )
        return (catalog,)
    if skill_id is not None:
        catalog_name, skill_name = skill_id.split("/", 1)
        catalog = by_name.get(catalog_name)
        if catalog is None:
            valid = ", ".join(by_name) or "(none)"
            raise DiscoveryError(
                f"unknown catalog `{catalog_name}` in `{skill_id}`. "
                f"Valid catalogs: {valid}."
            )
        skill = next((item for item in catalog.skills if item.name == skill_name), None)
        if skill is None:
            valid = ", ".join(item.name for item in catalog.skills) or "(none)"
            raise DiscoveryError(
                f"unknown skill `{skill_id}`. Valid skills in `{catalog_name}`: "
                f"{valid}."
            )
        return (Catalog(catalog.name, catalog.description, (skill,)),)
    return catalogs


def build_output(
    catalogs: tuple[Catalog, ...],
    source: str,
    full: bool,
) -> dict[str, object]:
    result_catalogs: list[dict[str, object]] = []
    for catalog in catalogs:
        skills: list[dict[str, str]] = []
        for skill in catalog.skills:
            if full:
                skills.append({"name": skill.name, "description": skill.description})
            else:
                skills.append(
                    {"name": skill.name, "summary": _summarize(skill.description)}
                )
        result_catalogs.append(
            {
                "name": catalog.name,
                "description": catalog.description,
                "skill_count": len(skills),
                "skills": skills,
            }
        )
    return {
        "source": source,
        "catalog_count": len(result_catalogs),
        "skill_count": sum(int(catalog["skill_count"]) for catalog in result_catalogs),
        "catalogs": result_catalogs,
    }


def _summarize(description: str) -> str:
    summary = description.split(" Use when ", 1)[0]
    if len(summary) <= SUMMARY_LIMIT:
        return summary
    shortened = summary[: SUMMARY_LIMIT - 1].rsplit(" ", 1)[0].rstrip()
    if not shortened:
        shortened = summary[: SUMMARY_LIMIT - 1]
    return f"{shortened}…"


def write_output(path: Path, payload: str) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise DiscoveryError(
            f"output directory `{parent}` does not exist. "
            "Create it or choose an existing directory."
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise DiscoveryError(
            f"cannot write output `{path}` ({exc}). Choose a writable output path."
        ) from None


def run_self_test() -> None:
    marker = "Disposable test inventory marker):"
    folded_skill = f"""---
name: meta-one
description: >-
  {marker} Builds one useful thing.
  Use when one thing is needed.
metadata:
  meta-skills.dependencies: "other/meta-two"
---

# One
"""
    plain_skill = f"""---
name: meta-two
description: "{marker} Builds another useful thing. Use when two is needed."
---

# Two
"""
    manifest = {
        "plugins": [
            {
                "name": "meta",
                "source": "./",
                "description": "Disposable harness builders.",
                "skills": [
                    "./skills/meta/meta-one",
                    "./skills/meta/meta-two",
                ],
            }
        ]
    }
    cases = 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        def materialize(raw_manifest: object = manifest) -> None:
            (root / ".claude-plugin").mkdir(parents=True, exist_ok=True)
            (root / ".claude-plugin" / "marketplace.json").write_text(
                json.dumps(raw_manifest), encoding="utf-8"
            )
            for relative, content in (
                ("skills/meta/meta-one/SKILL.md", folded_skill),
                ("skills/meta/meta-two/SKILL.md", plain_skill),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

        materialize()
        inventory = load_inventory(root)
        assert len(inventory) == 1
        assert inventory[0].skills[0].description.startswith("Builds one useful")
        assert "\n" not in inventory[0].skills[0].description
        cases += 1

        selected = select_inventory(inventory, "meta", None)
        assert len(selected) == 1 and selected[0].name == "meta"
        cases += 1

        try:
            select_inventory(inventory, "missing", None)
        except DiscoveryError as exc:
            assert "Valid catalogs" in str(exc)
        else:
            raise AssertionError("unknown catalog was accepted")
        cases += 1

        (root / "skills/meta/meta-two/SKILL.md").unlink()
        try:
            load_inventory(root)
        except DiscoveryError as exc:
            assert "meta-two/SKILL.md" in str(exc)
        else:
            raise AssertionError("missing skill file was accepted")
        cases += 1

        materialize()
        escaping = json.loads(json.dumps(manifest))
        escaping["plugins"][0]["skills"] = ["./skills/meta/../outside"]
        materialize(escaping)
        try:
            load_inventory(root)
        except DiscoveryError as exc:
            assert "path traversal" in str(exc)
        else:
            raise AssertionError("escaping marketplace path was accepted")
        cases += 1

        (root / ".claude-plugin" / "marketplace.json").write_text(
            "{not json", encoding="utf-8"
        )
        try:
            load_inventory(root)
        except DiscoveryError as exc:
            assert "not valid JSON" in str(exc)
        else:
            raise AssertionError("invalid manifest JSON was accepted")
        cases += 1

        materialize()
        mismatched = plain_skill.replace(marker, "Different test marker):")
        (root / "skills/meta/meta-two/SKILL.md").write_text(
            mismatched, encoding="utf-8"
        )
        try:
            load_inventory(root)
        except DiscoveryError as exc:
            assert "different disposal marker" in str(exc)
        else:
            raise AssertionError("mismatched marker was accepted")
        cases += 1

    original_urlopen = urllib.request.urlopen

    def offline(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.URLError("offline self-test")

    urllib.request.urlopen = offline
    try:
        try:
            _download_archive()
        except DiscoveryError as exc:
            assert "no cached inventory is used" in str(exc)
        else:
            raise AssertionError("network failure did not stop discovery")
    finally:
        urllib.request.urlopen = original_urlopen
    cases += 1

    print(json.dumps({"self_test": "pass", "cases": cases}, separators=(",", ":")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Example: python3 scripts/discover.py --catalog meta --full\n"
            "The default source is the live main branch of "
            "ryan-minato/skills."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--catalog", help="return only this exact catalog")
    selector.add_argument(
        "--skill",
        help="return one exact dependency target as catalog/skill",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="include full marker-free descriptions instead of summaries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically write full JSON here; stdout receives a short receipt",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="read this local repository root instead of the live remote archive",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run built-in parser and inventory fixtures, then exit",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        if any(
            value
            for value in (
                args.catalog,
                args.skill,
                args.full,
                args.output,
                args.repo_root,
            )
        ):
            parser.error("--self-test cannot be combined with discovery options")
        try:
            run_self_test()
        except (AssertionError, DiscoveryError, OSError) as exc:
            print(f"error: self-test failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.catalog and not CATALOG_RE.fullmatch(args.catalog):
        parser.error("--catalog must use lowercase letters, digits, and single hyphens")
    if args.skill:
        catalog_name, separator, skill_name = args.skill.partition("/")
        if (
            separator != "/"
            or "/" in skill_name
            or not CATALOG_RE.fullmatch(catalog_name)
            or not SKILL_RE.fullmatch(skill_name)
        ):
            parser.error("--skill must be a canonical `catalog/skill` identifier")

    try:
        inventory = load_inventory(args.repo_root)
        selected = select_inventory(inventory, args.catalog, args.skill)
        source = (
            str(args.repo_root.resolve()) if args.repo_root is not None else ARCHIVE_URL
        )
        output = build_output(selected, source, full=args.full or bool(args.output))
        payload = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        if args.output:
            write_output(args.output, payload)
            receipt = {
                "output": str(args.output),
                "catalog_count": output["catalog_count"],
                "skill_count": output["skill_count"],
            }
            print(json.dumps(receipt, separators=(",", ":")))
        else:
            print(payload)
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
