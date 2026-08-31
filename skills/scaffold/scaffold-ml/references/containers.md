# Containerized Environments

Read when the user asks for a containerized dev environment or for
training to run on a server or in a container. Never scaffold either
unprompted.

## Generic GPU container decisions

Whether a container is needed at all, CUDA/ROCm base-image selection with
live tag discovery, and GPU access wiring (docker run, Compose, dev
containers, shared memory) are covered by the `meta-gpu-container` skill —
make those decisions with it first, and its guidance supersedes the
summary below. When it is absent, recommend installing the whole `meta`
catalog (the workflow's container step carries the command), never that
skill alone; if the user declines, this summary is the fallback. The rest
of this reference is what is specific to this scaffold.

## Base image and the locked environment

The default is `nvidia/cuda` (a `runtime` variant unless something
compiles CUDA code) with the whole environment installed from this
scaffold's committed lock: only the CUDA layer comes from the image, and
the environment equals the committed file — this scaffold's whole
reproducibility story.

- Maintainable training project: `uv sync --frozen` from `uv.lock`.
- Quick experiment: `uv pip sync requirements.txt` from the compiled
  requirements.

A preinstalled-torch image (NGC `nvcr.io/nvidia/pytorch`,
`pytorch/pytorch`) conflicts with the locked environment; if one is
chosen, the image's torch is authoritative — drop torch from
`pyproject.toml` (training) or `requirements.in` (experiment) and record
that rule in AGENTS.md. On AMD hardware use `rocm/pytorch` instead, with
device passthrough (`--device /dev/kfd --device /dev/dri`) in place of
the NVIDIA reservation.

Tags are always enumerated live from the registry, never assumed: before
filling any `<tag>` placeholder, enumerate current tags from the
registry's own listing (the repository's Docker Hub tags page, the NGC
catalog), and confirm a specific tag with
`docker manifest inspect <image>:<tag>`.

## Volumes and data

Mount `data/`, `outputs/`, and the Hugging Face cache directory as
volumes. Images contain the environment only — never data, checkpoints,
or credentials. Record the container-path ↔ config-path mapping in
AGENTS.md.

## Shared memory

DataLoader workers exhaust Docker's default shared memory. The compose
assets carry an `shm_size` placeholder that must be filled with a real
value; the dev-container asset and plain `docker run` configure nothing —
add `--shm-size` (or `ipc: host`) there yourself.

## Assets

Each mode ships its own copies in its assets directory
(`assets/experiment/` or `assets/training/`).

- Dev environment → copy the mode's `devcontainer.json` to
  `.devcontainer/devcontainer.json`.
- Training image and runner → copy the mode's `Dockerfile` and
  `compose.yaml` to the project root, and add `docker-build` /
  `docker-train` recipes to the justfile in the same change.
