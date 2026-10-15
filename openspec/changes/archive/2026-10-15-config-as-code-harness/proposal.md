## Why

The `config-as-code` change adds `.tf` starting-shape assets to `meta-github-workflow` and `meta-gitlab-workflow`. A broken asset would only surface when an agent delivers it into a target project. Running the OpenTofu formatter and validator over every `.tf` asset in this repository's checks catches syntax and schema errors before merge.

## What Changes

- **OpenTofu pin.** The `justfile` gains an `opentofu_version` variable as the single source of the pin. `just install-tools` installs that version as a standalone binary from the official OpenTofu release into the repository's ignored `.tools/bin`, with its checksum verified. The dev container and CI get the same binary through the same recipe, and recipes call it by that path, so no `PATH` change is needed.
- **New recipe `tf-validate`.** For every `skills/*/*/assets/**/*.tf` file, it runs `tofu fmt -check`. Then, in a scratch copy, it runs `tofu init -backend=false` followed by `tofu validate`. Nothing is written into the skill directories. A failure names the asset.
- **`just check` runs `tf-validate`**, so `checks / quality` runs it with no workflow change beyond what `just install-tools` already does.
- **`scripts/validate_harness.py`** rejects an OpenTofu version hard-coded in `checks.yml` or `devcontainer.json`, matching the existing OpenSpec rule, so the `justfile` pin stays the only copy.
- **Documents:**
  - the `AGENTS.md` Validation table gains a `tf-validate` row;
  - `ARCHITECTURE.md` `## Quality Gates` names the new recipe;
  - `.agents/knowledge/github-checks.md` adds OpenTofu to the tools of the `checks / quality` row;
  - `.agents/knowledge/harness-maintenance.md` gains a register row for `opentofu_version`.

## Skills touched

Repository change (`skip_specs: true`): no public skill domain.

## Installed behavior

Agents working in this repository can run `just tf-validate`, or `just check`, to prove a `.tf` asset formats and validates against its pinned providers. A `.tf` asset that fails either step now fails `checks / quality`.

## Impact

- `justfile`, `.gitignore` (`.tools/`), `scripts/validate_harness.py`, `AGENTS.md`, `ARCHITECTURE.md`, `.agents/knowledge/github-checks.md`, `.agents/knowledge/harness-maintenance.md`.
- `.github/workflows/checks.yml` is not changed: the recipes call the binary by its path.
- `.devcontainer/devcontainer.json` is not changed, because `just setup` already runs `install-tools`.

## Non-goals

- Running `plan` or `apply` against any live organization or group.
- Validating `.tf` files anywhere other than skill assets.
- A Terraform CLI in this repository. The assets stay compatible with both tools, and validation uses OpenTofu.

## Tracked work

No issue. This is the companion of `config-as-code`.
