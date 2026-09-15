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

For the requirements carrier the environment stage installs the compiled
file instead of the lock:

```dockerfile
ENV UV_TORCH_BACKEND=<backend matching the base image>
COPY requirements.txt .
RUN uv venv --python <version> /opt/venv && uv pip sync --python /opt/venv/bin/python requirements.txt
```

Only the CUDA or ROCm layer comes from the base image; the environment
equals the committed lock (`uv sync --frozen --no-dev` for a uv project,
`uv pip sync requirements.txt` for the requirements carrier), which is
this scaffold's whole reproducibility story. The environment is installed
at `/opt/venv` (`UV_PROJECT_ENVIRONMENT`), outside `/app`, so the Compose
source mount over `/app` cannot hide it. The build context is filtered by
`.dockerignore` (from `assets/dockerignore`): `.git`, data, outputs,
secrets, and caches never enter an image, not even the sealed one. A
sealed image is stamped with the commit it copies (`GIT_COMMIT` build
argument, set by `just docker-build`, which refuses a dirty tree for the
sealed target): the manifest reads it when no repository is present, and
the image's revision label carries it too. Two guards, two claims: the
stamp says the tracked source inside the image is that commit, which the
dirty-tree check protects (ignored paths are outside the snapshot by
definition — credentials, local caches, machine-local configuration);
the image's contents are that commit plus whatever `.dockerignore` does
not exclude, which only the digest identifies. A sealed image is not
`git archive HEAD`; when the project needs that stronger guarantee,
build the sealed target from `git archive HEAD` piped as the context
instead of widening the dirty-tree check.

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

## Task-runner recipes

Append to the justfile; `<project name>` is the one slot. The sealed
build refuses a dirty tree so the stamp is the code inside; the digest
recipe prints what a run exports as `IMAGE_DIGEST`.

```just
docker-build target="runtime":
    @if [ "{{target}}" = sealed ] && [ -n "$(git status --porcelain)" ]; then echo "docker-build: commit first; a sealed image stamps the commit it copies" >&2; exit 1; fi
    docker build --target {{target}} --build-arg GIT_COMMIT=$(git rev-parse HEAD) --iidfile .image-id -t <project name>:{{target}} .

docker-digest:
    @cat .image-id
```
