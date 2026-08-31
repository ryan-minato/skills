# Containerized Environments

Read when the user asks for a containerized dev environment or for
training to run on a server or in a container. Never scaffold either
unprompted.

## Generic GPU container decisions

Whether a container is needed at all, CUDA/ROCm base-image selection with
live tag discovery, and GPU access wiring (docker run, Compose, dev
containers, shared memory) belong to the `meta-gpu-container` skill — make
those decisions there first. If it is not installed, install it from
https://github.com/ryan-minato/skills.git:

    npx skills add ryan-minato/skills --skill meta-gpu-container

The rest of this reference is what is specific to this scaffold.

## Base image and the lockfile

The default is `nvidia/cuda` (a `runtime` variant unless something compiles
CUDA code) with the whole environment installed by `uv sync --frozen`: only
the CUDA layer comes from the image, and the environment equals the
committed lockfile — this scaffold's whole reproducibility story. A
preinstalled-torch image (NGC `nvcr.io/nvidia/pytorch`, `pytorch/pytorch`)
conflicts with the lockfile; if one is chosen, the image's torch is
authoritative — drop torch from `pyproject.toml` and record that rule in
AGENTS.md. Tags are always enumerated live from the registry, never assumed.

## Volumes and data

Mount `data/`, `outputs/`, and the Hugging Face cache directory as
volumes. Images contain the environment only — never data, checkpoints,
or credentials. Record the container-path ↔ config-path mapping in
AGENTS.md. DataLoader workers exhaust Docker's default shared memory;
the assets raise it.

## Assets

- Dev environment → copy `assets/devcontainer.json` to
  `.devcontainer/devcontainer.json`.
- Training image and runner → copy `assets/Dockerfile` and
  `assets/compose.yaml` to the project root, and add `docker-build` /
  `docker-train` recipes to the justfile in the same change.
