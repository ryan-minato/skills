# Pinned Dependencies with uv

Read when creating or updating the requirements files.

## Shape

- `requirements.in` — hand-edited; the direct training dependencies only,
  unpinned or lightly bounded.
- `requirements.dev.in` — hand-edited; first line `-r requirements.in`,
  then dev tools (ruff, pytest, pre-commit).
- Compile both, and commit all four files in the same change:

  ```sh
  uv pip compile requirements.in -o requirements.txt
  uv pip compile requirements.dev.in -o requirements.dev.txt
  ```

- Install with `uv pip sync requirements.txt` on the training box and
  `uv pip sync requirements.dev.txt` on a dev machine — sync makes the
  environment exactly equal to the file, which is the reproducibility
  guarantee.
- Upgrades happen only by editing an `.in` file (or compiling with
  `--upgrade` / `--upgrade-package`) and recompiling — never by editing a
  compiled file.

## torch and other accelerator-bound packages

- Write `torch` plainly in `requirements.in` — no index URLs in the file.
  The wheel variant (CUDA, ROCm, CPU, XPU) is selected at compile and
  sync time with `--torch-backend <value>` (or the `UV_TORCH_BACKEND`
  environment variable), which the uv pip interface supports on
  `compile`, `sync`, and `install`. Verify current values and semantics in
  uv's official first-party PyTorch guidance.
- Reproducibility semantics: package versions pin in the compiled files;
  the wheel variant comes from the backend flag. Record an explicit
  backend (for example `cu130`) in the justfile so every machine that
  must match uses the same one; `auto` — which probes the local CUDA
  driver or AMD/Intel GPU — is for exploratory local installs, not for
  the recorded recipes.
- When the dev machine and the training box differ (a macOS laptop has
  no CUDA builds), both still install the same compiled versions — each
  machine runs the recipe with its own backend, and AGENTS.md records
  which machine uses which.
- Fallback when the flag cannot express the source (a private mirror, an
  index uv does not map): pass the explicit per-backend wheel index —
  listed in the same uv guide — at compile time.
- Packages that compile against the local CUDA toolchain (flash-attention
  and similar) follow their own docs, not the backend flag.
