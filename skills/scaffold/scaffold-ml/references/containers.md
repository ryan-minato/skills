# Containerized Environments

Read when the user asks for a dev container, a Compose environment, or
a training image, after the GPU container builder's decisions. Never
scaffold either unprompted.

## Generic GPU container decisions

Whether a container is needed at all, CUDA/ROCm base-image selection
with live tag discovery, and GPU access wiring (docker run, Compose, dev
containers, shared memory) are covered by the `meta-gpu-container`
skill — make those decisions with it first, and its guidance supersedes
the summary below. When it is absent, the workflow's container step has
already routed the whole `meta` catalog through
`ryan-minato-skills-installing`; if the user declined, this summary is
the fallback. The rest of this reference is what is specific to this
scaffold.

## Three stages, one identity

The recipe (`assets/Dockerfile`) has three
stages:

```text
environment   bare nvidia/cuda (runtime variant) or rocm base + uv; the
              project's committed lock installed — nothing else
runtime       the environment plus the entry points mounted or copied at
              run time; the target for day-to-day training
sealed        the environment plus the source tree copied in; the target
              for a run that must be reproducible from the image alone
```

Only the CUDA or ROCm layer comes from the base image; the environment
equals the committed lock (`uv sync --frozen --no-dev` for a uv project,
`uv pip sync requirements.txt` for the requirements carrier), which is
this scaffold's whole reproducibility story. The environment is installed
at `/opt/venv` (`UV_PROJECT_ENVIRONMENT`), outside `/app`, so the Compose
source mount over `/app` cannot hide it. The build context is filtered by
`.dockerignore` (from `assets/dockerignore`): `.git`, data, outputs,
secrets, and caches never enter an image, not even the sealed one. A
sealed run therefore records no commit; its digest identifies source and
environment together, and the image's revision label (`GIT_COMMIT` build
argument, set by `just docker-build`) ties it to the commit.

The **image digest** is the run's environment identity — never the
Dockerfile (a recipe) and never a tag (mutable). Obtain it after the
push from the image's repository digests, or, when the image is never
pushed, from the build's image-id file (`just docker-digest` prints it),
and inject it into the run as `IMAGE_DIGEST` so the manifest records it.
A container does not pin the host: the manifest still records the GPU
model and count, the driver, and the runtime versions beside the digest.

Tags are enumerated live from the registry, never assumed: before
filling any `<tag>` placeholder, enumerate current tags from the
registry's own listing and confirm one with
`docker manifest inspect <image>:<tag>`.

## Preinstalled-stack images

A vendor image with the framework preinstalled (NGC's PyTorch image,
`pytorch/pytorch`) conflicts with the locked environment. When one is
chosen, the image's framework is authoritative: remove the framework
from `pyproject.toml` or `requirements.in`, and record in `AGENTS.md`
that the image's stack rules and that the digest is still the identity.
On AMD hardware use `rocm/pytorch` with device passthrough
(`--device /dev/kfd --device /dev/dri`) instead of the NVIDIA reservation.

## Volumes and data

Mount `data/`, `outputs/`, and the model-hub cache directory as volumes;
record the container-path ↔ configuration-path mapping in `AGENTS.md`.
Images contain the environment only — never data, checkpoints,
credentials, or the `.env` file; secrets ride in at run time.

## Shared memory

Data-loader workers exhaust Docker's default shared memory. The Compose
asset carries an `shm_size` placeholder and the dev-container asset an
`--shm-size` placeholder in `runArgs`; both must be filled with a real
value. Plain `docker run` configures nothing — add `--shm-size` (or
`ipc: host`) yourself.

## Assets

- Dev environment → `assets/devcontainer.json`
  to `.devcontainer/devcontainer.json` (uv and just features, the GPU
  flags the GPU container builder decided, shared memory, post-create
  setup for the chosen carrier, cache directory).
- Training image and runner → `assets/Dockerfile`,
  `assets/compose.yaml`, and `assets/dockerignore` (to `.dockerignore`)
  at the project root; the justfile's `docker-build` and `docker-digest`
  recipes land in the same change.
