# The Requirements Carrier: Pinned Files with uv

Read when the carrier is the requirements workflow, when creating or
updating the requirements files, or when a development machine differs
from the training box. This carrier suits a scripts-only repository
with no installable package; a uv project is the default otherwise.

## Four files, committed together

- `requirements.in` — hand-edited; the runtime dependencies only,
  unpinned or lightly bounded (`assets/requirements.in`).
- `requirements.dev.in` — hand-edited; first line `-r requirements.in`,
  then the development tools (ruff, pytest, pre-commit)
  (`assets/requirements.dev.in`).
- `requirements.txt` and `requirements.dev.txt` — compiled, fully
  pinned; never edited by hand. Compile both and commit all four in the
  same change:

  ```sh
  uv pip compile requirements.in -o requirements.txt --torch-backend <backend>
  uv pip compile requirements.dev.in -o requirements.dev.txt --torch-backend <backend>
  ```

- The training box syncs `requirements.txt`; a development machine syncs
  `requirements.dev.txt`. `uv pip sync` makes the environment exactly
  equal to the file — that equality is the reproducibility guarantee, and
  `requirements.txt` is the environment identity the manifest hashes. A
  requirements file with ranges is not a lock; the manifest records it
  as unpinned.
- Upgrades happen only by editing an `.in` file (or compiling with
  `--upgrade` / `--upgrade-package`) and recompiling.

## Accelerator wheels

- Write `torch` plainly in `requirements.in` — no index URLs in the
  file. The wheel variant (CUDA, ROCm, CPU, XPU) is selected at compile
  and sync time with `--torch-backend <value>` (or `UV_TORCH_BACKEND`),
  which uv's pip interface supports on `compile`, `sync`, and `install`.
  Verify current values in uv's official first-party PyTorch guidance.
- Record an explicit backend (for example `cu130`) in the justfile so
  every machine that must match uses the same one; `auto` — which probes
  the local driver — is for exploratory installs, not for the recorded
  recipes.

## When the development machine differs from the training box

A macOS laptop has no CUDA builds; the training host has. Both machines
install the same compiled versions: each runs the recipe with its own
backend (the justfile carries one variable per machine class), and
`AGENTS.md` records which machine uses which. Packages that compile
against the local CUDA toolchain (flash-attention and similar) follow
their own documentation, not the backend flag.
