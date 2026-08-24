#!/usr/bin/env python3

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import tomllib

PLACEHOLDER_RE = re.compile(r"__[A-Z][A-Z0-9_]*__")
MARKER = "Disposable meta-" + "skill (delete after the harness is built):"
BACKENDS = {"local", "s3", "huggingface"}
REQUIRED_RECIPES = {
    "setup",
    "download-data",
    "pipeline",
    "test",
    "check",
    "report",
    "safe-to-commit",
    "safe-to-push",
}
REQUIRED_PATHS = (
    "AGENTS.md",
    "ARCHITECTURE.md",
    "pyproject.toml",
    ".env.example",
    "config/project.toml",
    "notebooks",
    "tests",
    "report/report.md",
    "src",
    ".agents/knowledge/PROJECT.md",
    ".agents/knowledge/DATA.md",
)
LOCKFILE_NAMES = {
    "uv.lock",
    "poetry.lock",
    "pdm.lock",
    "Pipfile.lock",
    "requirements.lock",
    "requirements.txt",
}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".txt"}
SKIP_PARTS = {".git", ".venv", "data", "output", "model", "__pycache__"}


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str


def add(issues: list[Issue], code: str, path: str | Path, message: str) -> None:
    issues.append(Issue(code=code, path=str(path), message=message))


def read_text(path: Path, issues: list[Issue], code: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        add(issues, code, path, f"Cannot read file: {error}. Restore or recreate it.")
        return ""


def load_toml(path: Path, issues: list[Issue], code: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        add(issues, code, path, f"Cannot parse TOML: {error}. Fix the file.")
        return {}
    return data


def check_required(root: Path, issues: list[Issue]) -> None:
    for relative in REQUIRED_PATHS:
        path = root / relative
        if not path.exists():
            add(
                issues,
                "structure.required",
                relative,
                "Required scaffold path is missing. Create and rework its asset.",
            )


def iter_harness_text(root: Path) -> list[Path]:
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if (
            path.name in {"justfile", ".gitignore", ".editorconfig", ".env.example"}
            or path.suffix in TEXT_SUFFIXES
        ):
            paths.append(path)
    return paths


def check_text_integrity(root: Path, issues: list[Issue]) -> None:
    for path in iter_harness_text(root):
        text = read_text(path, issues, "text.read")
        match = PLACEHOLDER_RE.search(text)
        if match:
            add(
                issues,
                "text.placeholder",
                path.relative_to(root),
                f"Unresolved scaffold placeholder {match.group(0)!r}. Replace or delete it.",
            )
        if MARKER in text:
            add(
                issues,
                "text.marker",
                path.relative_to(root),
                "Generated project content carries the disposable-skill marker. Remove it.",
            )


def markdown_targets(text: str) -> list[str]:
    return re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text)


def check_links(root: Path, issues: list[Issue]) -> None:
    for path in root.rglob("*.md"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        for target in markdown_targets(read_text(path, issues, "link.read")):
            target = target.strip().strip("<>")
            parsed = urlparse(target)
            if parsed.scheme or target.startswith("#"):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).resolve().exists():
                add(
                    issues,
                    "link.missing",
                    path.relative_to(root),
                    f"Link target {target!r} does not resolve. Fix or remove it.",
                )


def check_pyproject(root: Path, issues: list[Issue]) -> None:
    path = root / "pyproject.toml"
    data = load_toml(path, issues, "pyproject.parse")
    if not data:
        return
    if "project" not in data and "poetry" not in data.get("tool", {}):
        add(
            issues,
            "pyproject.project",
            path.name,
            "Missing project metadata. Add [project] or the selected package "
            "manager's supported project table.",
        )
    if "build-system" not in data:
        add(
            issues,
            "pyproject.package",
            path.name,
            "Missing [build-system]. Initialize a package/src layout with the "
            "project's selected package manager.",
        )
    if not any((root / name).is_file() for name in LOCKFILE_NAMES):
        add(
            issues,
            "project.lockfile",
            root,
            "No recognized dependency lock or pinned requirements file exists. "
            "Create one with the project's selected dependency manager.",
        )


def check_settings_and_workflows(root: Path, issues: list[Issue]) -> None:
    package_dirs = (
        [
            path
            for path in (root / "src").iterdir()
            if path.is_dir() and (path / "__init__.py").is_file()
        ]
        if (root / "src").is_dir()
        else []
    )
    if len(package_dirs) != 1:
        add(
            issues,
            "python.src-layout",
            "src",
            "Expected exactly one importable package under src/. Create a "
            "package with the project's selected package manager.",
        )
    workflow_files = list((root / "src").glob("*/workflows/*.py"))
    if not workflow_files:
        add(
            issues,
            "python.workflows",
            "src",
            "No workflow entry module found. Add at least one thin production entry.",
        )


def source_identity_ok(source: dict[str, Any]) -> bool:
    backend = source.get("backend")
    if backend == "s3":
        return bool(
            source.get("version") or (source.get("etag") and source.get("checksum"))
        )
    if backend == "huggingface":
        return bool(source.get("version"))
    return bool(source.get("version") or source.get("checksum"))


def mutable_identity(value: Any) -> bool:
    return str(value).strip().lower() in {"head", "latest", "main", "master"}


def source_module_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()


def check_storage(root: Path, config: dict[str, Any], issues: list[Issue]) -> None:
    sources = config.get("sources")
    products = config.get("products")
    if not isinstance(sources, list) or not sources:
        add(
            issues,
            "config.sources",
            "config/project.toml",
            "Define at least one [[sources]] table.",
        )
        sources = []
    if not isinstance(products, list) or not products:
        add(
            issues,
            "config.products",
            "config/project.toml",
            "Define at least one [[products]] table.",
        )
        products = []

    local_sources = []
    local_products = []
    package_dirs = (
        [
            path
            for path in (root / "src").iterdir()
            if path.is_dir() and (path / "__init__.py").is_file()
        ]
        if (root / "src").is_dir()
        else []
    )
    package_dir = package_dirs[0] if len(package_dirs) == 1 else None
    for kind, entries in (("source", sources), ("product", products)):
        for index, entry in enumerate(entries):
            path = f"config/project.toml:{kind}[{index}]"
            if not isinstance(entry, dict):
                add(issues, "config.entry", path, f"Each {kind} must be a TOML table.")
                continue
            missing = [
                name for name in ("name", "backend", "uri") if not entry.get(name)
            ]
            if missing:
                add(
                    issues,
                    "config.fields",
                    path,
                    f"Missing required fields: {', '.join(missing)}.",
                )
                continue
            backend = entry["backend"]
            if backend not in BACKENDS:
                add(
                    issues,
                    "config.backend",
                    path,
                    f"Unknown backend {backend!r}; choose local, s3, or huggingface.",
                )
                continue
            if kind == "source":
                if not source_identity_ok(entry):
                    identity_message = (
                        "S3 source needs an object version ID, or both ETag and checksum."
                        if backend == "s3"
                        else "Source has no required immutable version or checksum. Record one."
                    )
                    add(
                        issues,
                        "config.identity",
                        path,
                        identity_message,
                    )
                if entry.get("version") and mutable_identity(entry["version"]):
                    add(
                        issues,
                        "config.mutable-identity",
                        path,
                        "Source version is a mutable name. Resolve it to an immutable version or commit.",
                    )
                if package_dir is not None:
                    module_name = source_module_name(str(entry["name"]))
                    for relative in (
                        Path("sources") / f"{module_name}.py",
                        Path("workflows") / f"download_{module_name}.py",
                    ):
                        if not (package_dir / relative).is_file():
                            add(
                                issues,
                                "python.source-workflow",
                                (package_dir / relative).relative_to(root),
                                "Each source needs acquisition logic and a matching thin download workflow.",
                            )
                if backend == "local":
                    local_sources.append(entry)
            elif backend == "local":
                local_products.append(entry)
            if kind == "product" and str(entry["uri"]).rstrip("/").startswith("data/"):
                add(
                    issues,
                    "config.product-in-data",
                    path,
                    "Product URI points into immutable data/. Move it to output/ or a remote product location.",
                )

    raw_path = root / "data" / "raw"
    if raw_path.exists():
        add(
            issues,
            "storage.raw-layer",
            raw_path.relative_to(root),
            "Do not add a raw layer. Local original inputs live at data/<source>/.",
        )
    if local_sources:
        if not (root / "data").is_dir():
            add(
                issues,
                "storage.data-dir",
                "data",
                "Local sources require data/<source>/ directories.",
            )
        for source in local_sources:
            expected = root / str(source["uri"])
            if not expected.is_dir():
                add(
                    issues,
                    "storage.local-source",
                    source["uri"],
                    "Configured local source directory is missing. Create data/<source>/.",
                )
            relative_parts = Path(str(source["uri"])).parts
            if (
                len(relative_parts) < 2
                or relative_parts[0] != "data"
                or relative_parts[1] == "raw"
            ):
                add(
                    issues,
                    "storage.local-source-uri",
                    source["uri"],
                    "Local source URI must be data/<source>/ with no raw layer.",
                )
        if not list((root / "src").glob("*/data_guard.py")):
            add(
                issues,
                "storage.data-guard",
                "src",
                "Local sources require the copied data_guard.py and pipeline verification.",
            )
        guard_text = read_text(root / "justfile", issues, "storage.guard-read")
        if package_dir is not None:
            for path in (package_dir / "workflows").glob("*.py"):
                guard_text += read_text(path, issues, "storage.guard-read")
        if not all(
            token in guard_text for token in ("data_guard", "snapshot", "verify")
        ):
            add(
                issues,
                "storage.data-guard-wiring",
                "justfile",
                "Pipeline does not show snapshot and verify calls to data_guard. Wire both around production steps.",
            )
    if local_products:
        for relative in ("output", "output/_provenance"):
            if not (root / relative).is_dir():
                add(
                    issues,
                    "storage.output-dir",
                    relative,
                    "Local products require output/ and output/_provenance/.",
                )


def check_model(root: Path, config: dict[str, Any], issues: list[Issue]) -> None:
    model = config.get("model")
    if model is None:
        return
    if not isinstance(model, dict):
        add(
            issues,
            "config.model",
            "config/project.toml:model",
            "Model configuration must be a TOML table.",
        )
        return
    revision = model.get("revision")
    if not revision:
        add(
            issues,
            "config.model-revision",
            "config/project.toml:model",
            "Model use requires an immutable revision.",
        )
    elif mutable_identity(revision):
        add(
            issues,
            "config.model-revision",
            "config/project.toml:model",
            "Model revision is mutable. Resolve it to an immutable commit.",
        )
    local_path = model.get("local_path")
    if local_path:
        parts = Path(str(local_path)).parts
        if not parts or parts[0] != "model":
            add(
                issues,
                "config.model-path",
                "config/project.toml:model",
                "Local model weights must live under model/.",
            )
        if not (root / "model").is_dir():
            add(
                issues,
                "storage.model-dir",
                "model",
                "Local model configuration requires a model/ directory.",
            )


def check_commands_and_hooks(root: Path, issues: list[Issue]) -> None:
    justfile_path = root / "justfile"
    if justfile_path.is_file():
        justfile = read_text(justfile_path, issues, "just.read")
        recipes = set(
            re.findall(r"^([a-zA-Z0-9_-]+)(?:\s+[^:]*)?:", justfile, re.MULTILINE)
        )
        for recipe in sorted(REQUIRED_RECIPES - recipes):
            add(
                issues,
                "just.recipe",
                "justfile",
                f"Missing required recipe {recipe!r}. Add the stable command interface.",
            )


def check_gitignore(root: Path, issues: list[Issue]) -> None:
    text = read_text(root / ".gitignore", issues, "gitignore.read")
    lines = {
        line.strip().lstrip("/")
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    }
    for pattern in (".env", "data/", "output/", "model/"):
        if pattern not in lines:
            add(
                issues,
                "gitignore.pattern",
                ".gitignore",
                f"Missing ignored path {pattern!r}. Keep local sensitive or generated artifacts out of Git.",
            )


def check_agent_entry(root: Path, issues: list[Issue]) -> None:
    text = read_text(root / "AGENTS.md", issues, "agents.read")
    for target in (
        "ARCHITECTURE.md",
        ".agents/knowledge/PROJECT.md",
        ".agents/knowledge/DATA.md",
    ):
        if target not in text:
            add(
                issues,
                "agents.route",
                "AGENTS.md",
                f"Knowledge route {target!r} is missing. Add it to the when-to-read table.",
            )


def validate(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    if not root.is_dir():
        return [
            Issue(
                code="root.missing",
                path=str(root),
                message="Project root does not exist. Pass --project-root to the scaffolded repository.",
            )
        ]
    check_required(root, issues)
    check_text_integrity(root, issues)
    check_links(root, issues)
    check_pyproject(root, issues)
    check_settings_and_workflows(root, issues)
    config = load_toml(root / "config/project.toml", issues, "config.parse")
    if config:
        check_storage(root, config, issues)
        check_model(root, config, issues)
    check_commands_and_hooks(root, issues)
    check_gitignore(root, issues)
    check_agent_entry(root, issues)
    return issues


def render(root: Path, issues: list[Issue], output_format: str) -> None:
    payload = {
        "project_root": str(root),
        "status": "ok" if not issues else "error",
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }
    if output_format == "json":
        print(json.dumps(payload, indent=2))
        return
    if not issues:
        print(f"validate_scaffold: OK ({root})")
        return
    print(f"validate_scaffold: {len(issues)} issue(s) in {root}")
    for issue in issues:
        print(f"- [{issue.code}] {issue.path}: {issue.message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a generated reproducible data-analysis scaffold.",
        epilog=(
            "Example: python3 scripts/validate_scaffold.py "
            "--project-root /path/to/project --format json"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Root of the generated target project",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = args.project_root.resolve()
    try:
        issues = validate(root)
    except OSError as error:
        print(
            f"validate_scaffold: cannot inspect {root}: {error}. "
            "Check permissions and retry.",
            file=sys.stderr,
        )
        return 1
    render(root, issues, args.format)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
