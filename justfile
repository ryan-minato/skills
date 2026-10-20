# Canonical check recipes for this repository.
# Agents and humans should run checks through these recipes (not ad-hoc
# commands) so results stay consistent across environments.

# OpenSpec is the one pinned tool: the generated openspec-* skills must equal
# this version's output byte for byte. Every other tool follows its latest
# release. scripts/validate_harness.py checks that no other file names it.
openspec_version := "1.12.0"

# Repository-local tool directory (ignored by git); recipes find tools here first.
tools_bin := justfile_directory() / ".tools" / "bin"
export PATH := tools_bin + ":" + env_var("PATH")

# List available recipes
default:
    @just --list

# One-time environment setup (run after cloning / container creation)
setup: install-tools
    pre-commit install
    git config commit.template .gitmessage

# Install the tools `just check` needs beyond the dev container features:
# OpenSpec at the pinned version, run by Deno (no Node), into .tools/bin
install-tools: install-opentofu
    OPENSPEC_NO_UPDATE_CHECK=1 deno install --global --force --quiet --root "{{justfile_directory()}}/.tools" --name openspec --allow-read --allow-write --allow-env --allow-sys=cpus,homedir,uid "npm:@fission-ai/openspec@{{openspec_version}}"

# Install the latest OpenTofu release into .tools/bin, checking it against the
# release's SHA256SUMS (download layout per opentofu.org standalone install docs, read 2026-10-14)
install-opentofu:
    #!/usr/bin/env bash
    set -euo pipefail
    auth=()
    if [ -n "${GITHUB_TOKEN:-}" ]; then auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}"); fi
    release="$(curl --proto '=https' --tlsv1.2 -fsSL "${auth[@]}" https://api.github.com/repos/opentofu/opentofu/releases/latest)"
    version="$(printf '%s\n' "$release" | sed -n 's/^ *"tag_name": *"v\([0-9][0-9.]*\)".*/\1/p')"
    [ -n "$version" ] || { echo "install-opentofu: the latest-release response has no tag_name; read it at https://api.github.com/repos/opentofu/opentofu/releases/latest" >&2; exit 1; }
    bin="{{tools_bin}}/tofu"
    if [ -x "$bin" ] && "$bin" version -json | grep -q "\"terraform_version\": *\"$version\""; then
        echo "OpenTofu $version already installed at $bin"
        exit 0
    fi
    [ "$(uname -s)" = Linux ] || { echo "install-opentofu: this recipe downloads the Linux release; on $(uname -s) install OpenTofu $version yourself" >&2; exit 1; }
    case "$(uname -m)" in
        x86_64) arch=amd64 ;;
        aarch64|arm64) arch=arm64 ;;
        *) echo "install-opentofu: unsupported architecture $(uname -m)" >&2; exit 1 ;;
    esac
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    base="https://github.com/opentofu/opentofu/releases/download/v${version}"
    zip="tofu_${version}_linux_${arch}.zip"
    curl --proto '=https' --tlsv1.2 -fsSL -o "$tmp/$zip" "$base/$zip"
    curl --proto '=https' --tlsv1.2 -fsSL -o "$tmp/SHA256SUMS" "$base/tofu_${version}_SHA256SUMS"
    (cd "$tmp" && grep " ${zip}\$" SHA256SUMS | sha256sum -c -)
    mkdir -p "{{tools_bin}}"
    unzip -o -q "$tmp/$zip" tofu -d "{{tools_bin}}"
    "$bin" version

# Validate skill layout, harness synchronization, and catalog consistency
validate:
    python3 scripts/validate_skills.py
    python3 scripts/validate_harness.py

# Lint specific skill directories (spec + quality checks)
check-skill +PATHS:
    ./scripts/check_skill.ts {{PATHS}}

# Regenerate marketplace.json skills[] from the catalogs on disk
gen-marketplace:
    python3 scripts/gen_marketplace.py

# Lint and check formatting: the TypeScript management code with Deno (the paths
# deno.json includes, never a skill), the Python scripts with ruff
lint:
    deno lint
    deno fmt --check
    deno check scripts/ .agents/skills/change-workflow/scripts/
    ruff check scripts skills/*/*/scripts skills/meta/*/assets skills/sdd/*/assets
    ruff format --check scripts skills/*/*/scripts skills/meta/*/assets skills/sdd/*/assets

# Run the management code's tests (the *_test.ts files deno.json includes)
test:
    deno test --allow-read --allow-write --allow-run --allow-env --allow-net --no-prompt

# Format-check and validate every .tf skill asset with the pinned OpenTofu, in temp copies
tf-validate:
    #!/usr/bin/env bash
    set -euo pipefail
    tofu="{{tools_bin}}/tofu"
    [ -x "$tofu" ] || { echo "tf-validate: OpenTofu missing at $tofu; run 'just install-tools'" >&2; exit 1; }
    mapfile -t assets < <(find skills -path '*/assets/*' -name '*.tf' | sort)
    if [ "${#assets[@]}" -eq 0 ]; then
        echo "tf-validate: no .tf assets"
        exit 0
    fi
    export TF_PLUGIN_CACHE_DIR="{{justfile_directory()}}/.tools/plugin-cache"
    mkdir -p "$TF_PLUGIN_CACHE_DIR"
    failed=0
    for asset in "${assets[@]}"; do
        if ! "$tofu" fmt -check -diff "$asset"; then
            echo "tf-validate: $asset is not formatted (tofu fmt)" >&2
            failed=1
            continue
        fi
        work="$(mktemp -d)"
        cp "$asset" "$work/main.tf"
        if ! ( cd "$work" && "$tofu" init -backend=false -input=false -no-color >/dev/null && "$tofu" validate -no-color ); then
            echo "tf-validate: $asset failed init or validate" >&2
            failed=1
        fi
        rm -rf "$work"
    done
    exit "$failed"

# Validate every OpenSpec spec and change (strict)
spec-validate:
    @command -v openspec >/dev/null || { echo "openspec CLI missing: run 'just install-tools' (installs @fission-ai/openspec@{{openspec_version}} through Deno)" >&2; exit 1; }
    OPENSPEC_NO_UPDATE_CHECK=1 openspec validate --all --strict --no-interactive

# Regenerate the OpenSpec-managed skills after bumping openspec_version; commit the result
spec-sync:
    OPENSPEC_NO_UPDATE_CHECK=1 openspec update --force

# Strict validation plus the unarchived-change rule for the changes this branch touches (what `checks / spec` runs on a pull request)
# This repository is combined-shape: every pull request ends with its record archived, so no request may merge unarchived.
spec-check BASE="origin/main" HEAD="HEAD" *ARGS:
    python3 scripts/spec_changes.py check --base {{BASE}} --head {{HEAD}} --shape combined {{ARGS}}

# Related-change tooling for a pull request: snapshot, status, show, check, archive, labels (see --help)
spec-changes *ARGS:
    python3 scripts/spec_changes.py {{ARGS}}

# Safety gate for staged changes (also runs as the first pre-commit hook)
commit-gate:
    python3 scripts/check_commit_safety.py

# Run every check (validators, lint, tests, spec validation, the remaining pre-commit hooks;
# the hooks the recipes above already ran are skipped)
check: validate lint test spec-validate tf-validate
    SKIP=commit-safety,validate-skills,validate-harness,ruff-check,ruff-format pre-commit run --all-files
