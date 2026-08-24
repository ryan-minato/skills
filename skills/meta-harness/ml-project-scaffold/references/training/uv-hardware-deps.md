# Hardware-Matched Dependencies with uv

Read when adding torch or any hardware-bound dependency, and again when
the project's hardware changes. Mechanics below are uv's; verify current
syntax and index URLs against uv's official first-party PyTorch guidance
before committing.

## Declare the indexes

One `[[tool.uv.index]]` block per backend the project targets, always
with `explicit = true` so only routed packages may use it — without
`explicit`, unrelated dependencies can resolve from the PyTorch index:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

The per-backend index URLs (cpu, CUDA generations, ROCm, XPU) and current
version numbers are listed in the uv guide — fetch, do not recall.

## Route the packages

`[tool.uv.sources]` pins torch-family packages (torch, torchvision, and
companions like triton) to the declared index:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]
torchvision = [{ index = "pytorch-cu130" }]
```

PyTorch publishes no CUDA builds for macOS — when dev machines differ
from the training box, gate the routing with environment markers
(`marker = "sys_platform == 'linux' or sys_platform == 'win32'"`; ROCm is
Linux-only). Examples for each combination are in the uv guide.

## Single- vs multi-hardware projects

- **One accelerator target (the default):** route to one index and stop —
  the simplest form that works, matching the expose-only-what-changes
  rule.
- **Genuinely multiple targets:** define mutually exclusive extras
  (`[project.optional-dependencies]` with e.g. `cpu` and `cu130`),
  declare them conflicting under `[tool.uv] conflicts`, route each extra
  to its index in `[tool.uv.sources]`, and install with
  `uv sync --extra cu130`. Adopt this only when the project actually
  trains on more than one accelerator family.

## Picking the backend

Probe the machine: `nvidia-smi` (CUDA driver version → supported CUDA
generation), `rocminfo` for AMD. Match the index to what the driver
supports, not to the newest number.

## Boundaries

- `--torch-backend` / `UV_TORCH_BACKEND` belong to uv's pip interface
  and do not apply to `uv sync` / `uv add` — in this project the answer
  is always indexes plus sources.
- Packages that compile against the local CUDA toolchain
  (flash-attention and similar) follow their own docs, not the index
  routing.
