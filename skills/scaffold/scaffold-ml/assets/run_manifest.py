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

The manifest is a plain JSON document. It records the executed commit,
whether the tree was dirty and the hash of the uncommitted changes
(tracked diff plus untracked files), the resolved configuration's hash,
the image digest (from IMAGE_DIGEST) or a real lock file's hash,
interpreter and host facts, GPU and runtime facts when available, seeds,
inputs, and lineage. A `degraded` list names every identity the record
lacks (a dirty tree, no resolved configuration, no environment identity,
no runtime facts), and the same reasons are printed to stderr, so an
incomplete record never looks complete. The start-time record is kept as
`manifest.running.json` when the run finishes. Every value is an identity
or a hash; the module never records secrets, environment variables,
command lines, or data.

Runtime facts cover PyTorch, JAX, and TensorFlow when importable and
NVIDIA devices when `nvidia-smi` is present; extend `runtime_facts` and
`host_facts` for other stacks (ROCm, TPU) in the project's copy.
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

LOCK_FILES = ("uv.lock", "poetry.lock", "pdm.lock", "Pipfile.lock", "conda-lock.yml", "requirements.lock")
PINNED_REQUIREMENTS = "requirements.txt"  # accepted only when every requirement is pinned with ==


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
    """The executed commit, the dirty flag, and a hash of everything uncommitted when dirty."""
    cwd = str(repo_root) if repo_root else None
    base = ["git", "-C", cwd] if cwd else ["git"]
    commit = _run([*base, "rev-parse", "HEAD"])
    if commit is None:
        return {"commit": None, "dirty": None, "patch_sha256": None, "untracked_files": None, "remote": None}
    status = _run([*base, "status", "--porcelain"]) or ""
    dirty = bool(status)
    patch_sha = None
    untracked = []
    if dirty:
        # `git diff HEAD` covers tracked changes only; untracked files are part
        # of the executed source too, so their contents enter the hash.
        root = Path(_run([*base, "rev-parse", "--show-toplevel"]) or (cwd or "."))
        untracked = sorted((_run([*base, "ls-files", "--others", "--exclude-standard"]) or "").splitlines())
        parts = [_run([*base, "diff", "HEAD"]) or ""]
        for rel in untracked:
            path = root / rel
            if path.is_file():
                parts.append(f"{rel}\n{_sha256_file(path)}")
        patch_sha = _sha256_text("\n".join(parts))
    return {
        "commit": commit,
        "dirty": dirty,
        "patch_sha256": patch_sha,
        "untracked_files": len(untracked),
        "remote": _strip_userinfo(_run([*base, "remote", "get-url", "origin"])),
    }


def _strip_userinfo(url: str | None) -> str | None:
    """Drop `user:token@` from a remote URL so an embedded credential never reaches the manifest."""
    if not url or "@" not in url:
        return url
    if "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://{rest.rsplit('@', 1)[1]}"
    return url  # scp-style git@host:path carries a user name, not a secret


def _fully_pinned(requirements: Path) -> bool:
    """True when every requirement line pins an exact version (`==`) or a direct reference (`@`)."""
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "--")):
            continue
        if "==" not in line and " @ " not in line:
            return False
    return True


def environment_identity(repo_root: Path | None = None) -> dict[str, Any]:
    """Image digest when injected, else a real lock file's hash; plus the interpreter.

    An unpinned requirements file is not a lock: it is reported under
    `unpinned_manifest` and contributes no identity.
    """
    root = repo_root or Path.cwd()
    lock = next((root / name for name in LOCK_FILES if (root / name).is_file()), None)
    unpinned = None
    if lock is None and (root / PINNED_REQUIREMENTS).is_file():
        if _fully_pinned(root / PINNED_REQUIREMENTS):
            lock = root / PINNED_REQUIREMENTS
        else:
            unpinned = PINNED_REQUIREMENTS
    return {
        "image_digest": os.environ.get("IMAGE_DIGEST"),
        "lock_file": lock.name if lock else None,
        "lock_sha256": _sha256_file(lock) if lock else None,
        "unpinned_manifest": unpinned,
        "python": platform.python_version(),
    }


def runtime_facts() -> dict[str, Any]:
    """Framework and accelerator runtime versions for every importable framework."""
    facts: dict[str, Any] = {}
    try:
        import torch  # optional: only when the project uses it

        facts["torch"] = torch.__version__
        facts["cuda"] = getattr(torch.version, "cuda", None)
        facts["hip"] = getattr(torch.version, "hip", None)
        facts["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version()) if torch.cuda.is_available() else None
    except ImportError:
        pass
    try:
        import jax  # optional

        facts["jax"] = jax.__version__
        facts["jax_backend"] = jax.default_backend()
    except ImportError:
        pass
    try:
        import tensorflow as tf  # optional

        facts["tensorflow"] = tf.__version__
    except ImportError:
        pass
    return facts


def host_facts() -> dict[str, Any]:
    """Hostname, platform, GPU facts when a query tool is available, and runtime versions."""
    facts: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpus": [],
        "driver": None,
        "runtime": runtime_facts(),
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
    if resolved_config_path and resolved_config_path.is_file():
        config = {"resolved_path": str(resolved_config_path), "sha256": _sha256_file(resolved_config_path)}
    source = source_snapshot(repo_root)
    environment = environment_identity(repo_root)
    host = host_facts()
    degraded = []
    if source["commit"] is None:
        degraded.append("no_git_commit")
    elif source["dirty"]:
        degraded.append("dirty_tree")
    if config["sha256"] is None:
        degraded.append("no_resolved_config")
    if environment["image_digest"] is None and environment["lock_sha256"] is None:
        degraded.append("no_environment_identity")
    if not host["runtime"]:
        degraded.append("no_runtime_facts")
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": _now(),
        "ended_at": None,
        "degraded": degraded,
        "source": source,
        "config": config,
        "environment": environment,
        "host": host,
        "inputs": inputs,
        "randomness": {"seed": seed, "deterministic": deterministic},
        "parent_run_id": parent_run_id,
    }
    if degraded:
        print(f"run_manifest: run {run_id} is degraded: {', '.join(degraded)}", file=sys.stderr)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def finish_manifest(path: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    """Finalize the manifest: status, end time, and any late facts (e.g. the tracker run URL).

    The start-time record is preserved beside it as `manifest.running.json`.
    """
    running = path.with_name("manifest.running.json")
    if not running.exists():
        running.write_bytes(path.read_bytes())
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["ended_at"] = _now()
    if extra:
        manifest.update(extra)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
