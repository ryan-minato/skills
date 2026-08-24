# Containerized Environments

Read when the user asks for a containerized dev environment or for
training to run on a server or in a container. Never scaffold either
unprompted.

## Is a container needed at all?

When `uv pip sync requirements.txt` works directly on the target machine,
skip containers. Legitimate reasons to add one: the server's environment
is not yours to control, CUDA/driver versions drift between machines, the
team needs identical dev environments, or the deployment platform demands
an image. The dev container and the training image are two independent
decisions — take either without the other.

## Base image

| Image | Positioning |
|---|---|
| `nvidia/cuda` (`base`/`runtime`/`devel` variants) + Python and torch from this project's compiled requirements — the default here | Only the CUDA layer comes from the image; the environment equals the committed `requirements.txt`, which is this scaffold's whole reproducibility story |
| `nvcr.io/nvidia/pytorch` (NGC) | NVIDIA's tuned full stack (CUDA, cuDNN, NCCL) with torch preinstalled; large |
| `pytorch/pytorch` (`-runtime`/`-devel`) | The PyTorch project's slim images, torch preinstalled |
| `rocm/pytorch` | AMD GPUs |

Preinstalled-torch images conflict with the compiled requirements: either
the image's torch is authoritative (drop torch from `requirements.in` and
record that rule in AGENTS.md) or — the default — build on `nvidia/cuda`
and install everything from the compiled file. Tags are always enumerated
live from the registry, never assumed.

## GPU access

- Host prerequisite for NVIDIA: the NVIDIA Container Toolkit. AMD needs the
  kernel driver plus device nodes only. Verify current setup in official
  first-party documentation.
- `docker run`: `--gpus all` (or `--gpus 'device=N'` to expose a subset);
  ROCm passes devices through with `--device /dev/kfd --device /dev/dri`.
- Compose: a device reservation under the service
  (`deploy.resources.reservations.devices` with `driver: nvidia` and
  `capabilities: [gpu]`); verify current syntax in the Compose documentation.
- Dev container: declare `"hostRequirements": {"gpu": "optional"}` and
  nothing else — implementations inject `--gpus all` when a GPU runtime
  is present and skip it otherwise, so one config serves GPU and
  CPU-only machines. Verify the current dev-container specification. Two known limits:
  detection keys on the *runtime*, so a machine with the NVIDIA runtime
  but no GPU can fail to start; and compose-based dev containers ignore
  the field. In both cases fall back to explicit
  `"runArgs": ["--gpus", "all"]`, accepting the config then only works
  on GPU machines.
- DataLoader workers exhaust Docker's default shared memory: raise it
  (`--shm-size`, compose `shm_size`, or `ipc: host`) in any training
  container.

## Volumes and data

Mount `data/`, `outputs/`, and the Hugging Face cache directory as
volumes. Images contain the environment only — never data, checkpoints,
or credentials. Record the container-path ↔ config-path mapping in
AGENTS.md.

## Assets

- Dev environment → copy `assets/devcontainer.json` to
  `.devcontainer/devcontainer.json`.
- Training image and runner → copy `assets/Dockerfile` and
  `assets/compose.yaml` to the project root, and add `docker-build` /
  `docker-train` recipes to the justfile in the same change.
