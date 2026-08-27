#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024
SCHEMA_VERSION = 1


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(data_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in data_dir.rglob("*") if item.is_file()):
        stat = path.stat()
        files.append(
            {
                "path": path.relative_to(data_dir).as_posix(),
                "size": stat.st_size,
                "sha256": hash_file(path),
            }
        )
    return {"schema": SCHEMA_VERSION, "files": files}


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read manifest {path}: {error}. "
            "Create it with the snapshot command."
        ) from error
    if manifest.get("schema") != SCHEMA_VERSION or not isinstance(
        manifest.get("files"), list
    ):
        raise ValueError(
            f"Manifest {path} has an unsupported shape. "
            "Create a new manifest at a new provenance path."
        )
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, list[str]]:
    expected_files = {item["path"]: item for item in expected["files"]}
    actual_files = {item["path"]: item for item in actual["files"]}
    expected_paths = set(expected_files)
    actual_paths = set(actual_files)
    changed = sorted(
        path
        for path in expected_paths & actual_paths
        if expected_files[path]["size"] != actual_files[path]["size"]
        or expected_files[path]["sha256"] != actual_files[path]["sha256"]
    )
    return {
        "added": sorted(actual_paths - expected_paths),
        "removed": sorted(expected_paths - actual_paths),
        "changed": changed,
    }


def resolve_inputs(data_dir_text: str, manifest_text: str) -> tuple[Path, Path]:
    data_dir = Path(data_dir_text).resolve()
    manifest = Path(manifest_text).resolve()
    if not data_dir.is_dir():
        raise ValueError(
            f"Data directory does not exist: {data_dir}. "
            "Download the configured source version first."
        )
    if manifest.is_relative_to(data_dir):
        raise ValueError(
            f"Manifest must live outside immutable data: {manifest}. "
            "Place it in the product provenance location."
        )
    return data_dir, manifest


def snapshot(data_dir: Path, manifest_path: Path) -> int:
    current = build_manifest(data_dir)
    if manifest_path.exists():
        existing = load_manifest(manifest_path)
        differences = compare(existing, current)
        if any(differences.values()):
            print(
                json.dumps(
                    {
                        "status": "error",
                        "message": (
                            "Existing manifest does not match the source tree. "
                            "Do not overwrite the baseline; publish a new source "
                            "version and use a new provenance path."
                        ),
                        "differences": differences,
                    },
                    indent=2,
                )
            )
            return 1
        print(
            json.dumps(
                {
                    "status": "ok",
                    "action": "snapshot",
                    "files": len(current["files"]),
                    "manifest": str(manifest_path),
                    "unchanged": True,
                },
                indent=2,
            )
        )
        return 0

    write_manifest(manifest_path, current)
    print(
        json.dumps(
            {
                "status": "ok",
                "action": "snapshot",
                "files": len(current["files"]),
                "manifest": str(manifest_path),
                "unchanged": False,
            },
            indent=2,
        )
    )
    return 0


def verify(data_dir: Path, manifest_path: Path) -> int:
    expected = load_manifest(manifest_path)
    actual = build_manifest(data_dir)
    differences = compare(expected, actual)
    if any(differences.values()):
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": (
                        "Original local data changed. Restore the recorded source "
                        "version; pipeline workflows may not mutate data/."
                    ),
                    "differences": differences,
                },
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "action": "verify",
                "files": len(actual["files"]),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Snapshot or verify immutable local source data.",
        epilog=(
            "Example: uv run python -m PACKAGE.data_guard snapshot "
            "data output/_provenance/input-manifest.json"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("snapshot", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("data_dir", help="Local original-data directory")
        subparser.add_argument(
            "manifest", help="Manifest path outside the data directory"
        )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        data_dir, manifest = resolve_inputs(args.data_dir, args.manifest)
        if args.command == "snapshot":
            return snapshot(data_dir, manifest)
        return verify(data_dir, manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps({"status": "error", "message": str(error)}, indent=2),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
