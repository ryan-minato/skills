"""Run manifest: collect and write the identity of a training or evaluation run.

Drop this module into the project and call it from the training entry point:

    from run_manifest import start_manifest, finish_manifest

    manifest = start_manifest(
        output_dir=Path("outputs") / run_id,
        run_id=run_id,
        resolved_config_path=Path("outputs") / run_id / "config.resolved.yaml",
        seed=cfg.seed,
        inputs=[{"name": "train", "kind": "dataset", "identity": "hf:org/data@<revision>"}],
        parent_run_id=None,
    )
    ...
    finish_manifest(manifest, status="complete")

The manifest is a plain JSON document. It records the executed commit and
whether the tree was dirty, the resolved configuration's hash, the image
digest (from IMAGE_DIGEST) or the lock file's hash, interpreter and host
facts, GPU facts when nvidia-smi is available, seeds, inputs, and lineage.
Every value is an identity or a hash; the module never records secrets,
environment variables, or data.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOCK_CANDIDATES = ("uv.lock", "poetry.lock", "requirements.txt", "environment.yml")


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_snapshot(repo_root: Path | None = None) -> dict[str, Any]:
    """The executed commit, the dirty flag, and the patch hash when dirty."""
    cwd = str(repo_root) if repo_root else None
    commit = _run(["git", "-C", cwd, "rev-parse", "HEAD"] if cwd else ["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"commit": None, "dirty": None, "patch_sha256": None, "remote": None}
    base = ["git", "-C", cwd] if cwd else ["git"]
    status = _run([*base, "status", "--porcelain"]) or ""
    dirty = bool(status)
    patch = _run([*base, "diff", "HEAD"]) if dirty else None
    return {
        "commit": commit,
        "dirty": dirty,
        "patch_sha256": _sha256_text(patch) if patch else None,
        "remote": _run([*base, "remote", "get-url", "origin"]),
    }


def environment_identity(repo_root: Path | None = None) -> dict[str, Any]:
    """Image digest when injected, else the lock file's hash; plus the interpreter."""
    root = repo_root or Path.cwd()
    lock = next((root / name for name in LOCK_CANDIDATES if (root / name).exists()), None)
    return {
        "image_digest": os.environ.get("IMAGE_DIGEST"),
        "lock_file": lock.name if lock else None,
        "lock_sha256": _sha256_file(lock) if lock else None,
        "python": platform.python_version(),
    }


def host_facts() -> dict[str, Any]:
    """Hostname, platform, and GPU facts when a query tool is available."""
    facts: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpus": [],
        "driver": None,
    }
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if out:
            rows = [line.split(", ") for line in out.splitlines() if line.strip()]
            facts["gpus"] = [row[0] for row in rows]
            facts["driver"] = rows[0][1] if rows and len(rows[0]) > 1 else None
    return facts


def start_manifest(
    output_dir: Path,
    run_id: str,
    resolved_config_path: Path | None,
    seed: int | None,
    inputs: list[dict[str, str]],
    parent_run_id: str | None = None,
    deterministic: bool | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Write the manifest with status running and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"resolved_path": None, "sha256": None}
    if resolved_config_path and resolved_config_path.exists():
        config = {"resolved_path": str(resolved_config_path), "sha256": _sha256_file(resolved_config_path)}
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": _now(),
        "ended_at": None,
        "source": source_snapshot(repo_root),
        "config": config,
        "environment": environment_identity(repo_root),
        "host": host_facts(),
        "inputs": inputs,
        "randomness": {"seed": seed, "deterministic": deterministic},
        "parent_run_id": parent_run_id,
        "argv": sys.argv,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def finish_manifest(path: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    """Finalize the manifest: status, end time, and any late facts (e.g. the tracker run URL)."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["ended_at"] = _now()
    if extra:
        manifest.update(extra)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
