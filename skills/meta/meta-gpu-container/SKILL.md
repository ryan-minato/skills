---
name: meta-gpu-container
description: >-
  Disposable meta-skill (delete after the harness is built): establishes GPU
  container environments for a project — judges whether a container is
  warranted, selects and live-verifies a current CUDA or ROCm base image,
  wires GPU access into docker run, Compose, and dev containers, and deposits
  the resulting rules into the target harness. Use when a project of any
  domain must run GPU workloads in containers — "add a GPU container",
  "containerize this for the GPU server", "run this on the GPU in Docker";
  when a CUDA or ROCm base image or its current tag must be chosen or
  refreshed; or when a scaffold builder defers GPU container setup here. Not
  for CPU-only containers, application deployment images, or authoring dev
  container Features and Templates.
license: Apache-2.0
compatibility: Image discovery scripts require Python 3.11+ and network access.
---

# GPU Container Environments

Build only after inspecting the target hardware and accelerator stack, the
machines the project actually runs on, and the project's existing dependency
management. Containers are never scaffolded unprompted.

## Is a container needed at all?

When the project's environment sync works directly on the target machine, skip
containers. Legitimate reasons to add one: the server's environment is not
yours to control, CUDA/driver versions drift between machines, the team needs
identical dev environments, or the deployment platform demands an image. The
dev container and the workload image are two independent decisions — take
either without the other.

## Choose the base image

| Image | Positioning |
|---|---|
| `nvidia/cuda` (`base`/`runtime`/`devel` variants) + the project's own locked environment installed on top — the default | Only the CUDA layer comes from the image; the environment equals the project's committed lockfile, keeping one reproducibility story inside and outside the container |
| Vendor full-stack images (e.g. NGC `nvcr.io/nvidia/pytorch`) | Tuned complete stack (CUDA, cuDNN, NCCL) with the framework preinstalled; large |
| Framework project images (e.g. `pytorch/pytorch` `-runtime`/`-devel`) | Slim, framework preinstalled |
| `rocm/*` (e.g. `rocm/pytorch`) | AMD GPUs |

A preinstalled-stack image conflicts with the project's lockfile: either the
image's stack is authoritative — drop those packages from the project
dependencies and record that rule in the target AGENTS.md — or, the default,
build on `nvidia/cuda` and install everything from the lockfile. Use a
`runtime` variant unless something compiles GPU code in the image.

## Discover current tags

A current image tag is a live inventory fact: enumerate tags from the registry
at decision time, never assume or reconstruct one, and never deposit a static
tag catalog anywhere. Run the relevant script:

- [scripts/list_dockerhub_tags.py](scripts/list_dockerhub_tags.py) — tags of a
  Docker Hub repository.
- [scripts/list_ngc_images.py](scripts/list_ngc_images.py) — search the NVIDIA
  NGC catalog for images.
- [scripts/list_ngc_tags.py](scripts/list_ngc_tags.py) — tags of one NGC image.

Each prints one result per line and documents itself via `--help`. Read
[references/image-discovery.md](references/image-discovery.md) when a script
fails, the registry is not covered by them, or a result needs manual
confirmation. Record the verified image choice and its rationale in the target
project.

## Wire GPU access

- Host prerequisite for NVIDIA: the NVIDIA Container Toolkit. AMD needs the
  kernel driver plus device nodes only. Verify current setup in official
  first-party documentation.
- `docker run`: `--gpus all` (or `--gpus 'device=N'` to expose a subset);
  ROCm passes devices through with `--device /dev/kfd --device /dev/dri`.
- Compose: a device reservation under the service
  (`deploy.resources.reservations.devices` with `driver: nvidia` and
  `capabilities: [gpu]`); verify current syntax in the Compose documentation.
- Dev container: declare `"hostRequirements": {"gpu": "optional"}` and nothing
  else — implementations inject `--gpus all` when a GPU runtime is present and
  skip it otherwise, so one config serves GPU and CPU-only machines. Verify
  the current dev-container specification. Two known limits: detection keys on
  the *runtime*, so a machine with the NVIDIA runtime but no GPU can fail to
  start; and compose-based dev containers ignore the field. In both cases fall
  back to explicit `"runArgs": ["--gpus", "all"]`, accepting the config then
  only works on GPU machines.
- Workloads that move tensors between worker processes (e.g. PyTorch
  DataLoader workers) exhaust Docker's default shared memory: raise it
  (`--shm-size`, compose `shm_size`, or `ipc: host`) in any such container.

When a `devcontainer-setup` skill is installed, let it drive dev-container
authoring generally; this skill supplies the GPU-specific decisions on top.

## Volumes and data

Images contain the environment only — never data, model weights, outputs, or
credentials. Mount data directories, output directories, and large model or
dataset caches (e.g. the Hugging Face cache) as volumes, and pass secrets in
at runtime.

## Deposit durable output

Before this builder is disposed of, the target project must carry everything
needed to operate the containers without it: the chosen image and its
rationale, the image-authority rule when a preinstalled stack was selected,
the container-path ↔ config-path mapping for mounted volumes, and the build
and run commands, recorded in AGENTS.md or the project's task runner. Do not
copy this skill's disposable marker into any generated file.
