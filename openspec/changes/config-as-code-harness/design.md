## Context

See proposal.md for motivation, and `config-as-code` for the assets this change validates.

**Current shape:**
- **Pins.** The `justfile` holds the OpenSpec pin as `openspec_version`, and `install-tools` installs it.
- **Duplicate-pin guard.** `scripts/validate_harness.py` fails when `checks.yml` or `devcontainer.json` hard-codes the OpenSpec version.
- **The quality job.** `checks / quality` runs `just install-tools && just check`.
- **`just check`.** It is `validate`, `lint`, `spec-validate`, and the pre-commit hooks.
- **`.tf` files.** The repository has no `.tf` file and no OpenTofu or Terraform tool today.

**Binding constraints:**
- **Recipes.** Checks run through `just` recipes only, and `just check` is what CI runs.
- **Pins.** A tool pin has one source and a register row in `harness-maintenance.md`.
- **Mirrors.** `AGENTS.md` and `ARCHITECTURE.md` follow changes to check commands.
- **Network.** `tofu init` downloads the providers named in each asset's `required_providers`. CI has network access, and so does the dev container on a normal setup.

## Placement

| What Changes bullet | Carrier | Proof |
|---|---|---|
| OpenTofu pin | `justfile` `opentofu_version` and an `install-opentofu` step of `install-tools`: the release zip for the host architecture and the release's SHA256SUMS, as the official standalone install documentation lays them out, unpacked into the ignored `.tools/bin` (`.gitignore`) | `just install-tools` then `tofu version` prints the pin; a repeated `install-tools` leaves the same binary |
| `tf-validate` recipe | `justfile` recipe looping over `skills/*/*/assets/**/*.tf`: `tofu fmt -check`, then `init -backend=false` and `validate` in a temp copy | passes on the `config-as-code` assets; a deliberately broken scratch copy (an unknown attribute, a misformatted line) fails, naming the asset; `git status --porcelain` stays empty after a run |
| `just check` runs it | `justfile` `check` dependencies | `just check` output shows the recipe; the `checks / quality` run on the draft pull request shows it |
| Version guard | `scripts/validate_harness.py`, beside the OpenSpec rule | a scratch edit hard-coding a tofu version in `checks.yml` makes `just validate` fail, naming the file; reverted |
| Documents | `AGENTS.md` Validation table, `ARCHITECTURE.md` `## Quality Gates`, `.agents/knowledge/github-checks.md` quality row, `.agents/knowledge/harness-maintenance.md` register row | read-through; `just validate` |

## Decisions

- **OpenTofu, not Terraform, for this repository's validation.** It is MPL-licensed, and one pin is easier to keep than two.
  - The assets stay compatible with both tools by using only syntax both support: `import` blocks without `for_each` meet both floors.
  - Rejected: validating with both CLIs, which means two pins and twice the CI time for little extra assurance.
- **Install through `just install-tools`, not a dev container feature or a setup action.**
  - The `justfile` stays the single pin.
  - OpenTofu publishes no official dev container feature.
  - CI already calls `install-tools`.
  - Rejected: `opentofu/setup-opentofu` in CI, because it would add a second copy of the pin and a third-party action.
- **Validate a temp copy.** `init` writes `.terraform/` and a lock file, and neither belongs in a skill directory.
- **Part of `just check`, not a separate job.** Local and CI results stay identical, and no required-check name changes.

## Risks / Trade-offs

- **[`init` needs network access, so an offline `just check` fails]** → The failure names the asset and says `init` could not reach the registry. CI is the authority.
- **[The registry or a provider release is unavailable, which turns CI red for reasons outside the change]** → Each asset pins its providers' version constraints. A transient failure is retried by re-running the job, and the recipe never skips silently.
- **[The standalone installer's interface changes]** → The install step is verified against OpenTofu's installation documentation when written, and the date is recorded in the recipe comment.

## Verification plan

Written before implementation; results go to the pull request's Validation section. Every row of the Placement table names its proof. In addition:
- `just check` passes locally;
- `checks / quality` passes on the draft pull request after the assets land.
