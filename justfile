# Canonical check recipes for this repository.
# Agents and humans should run checks through these recipes (not ad-hoc
# commands) so results stay consistent across environments.

# Pinned tool versions. These variables are the single source for the pins;
# scripts/validate_harness.py checks that generated files and CI agree.
openspec_version := "1.12.0"
opentofu_version := "1.13.0"

# Repository-local tool directory (ignored by git); recipes call tools from here.
tools_bin := justfile_directory() / ".tools" / "bin"

# List available recipes
default:
    @just --list

# One-time environment setup (run after cloning / container creation)
setup: install-tools
    pre-commit install
    git config commit.template .gitmessage

# Install the pinned tools `just check` needs beyond the dev container features
install-tools: install-opentofu
    npm install -g "@fission-ai/openspec@{{openspec_version}}"

# Install the pinned OpenTofu release into .tools/bin, checking it against the
# release's SHA256SUMS (download layout per opentofu.org standalone install docs, read 2026-10-14)
install-opentofu:
    #!/usr/bin/env bash
    set -euo pipefail
    version="{{opentofu_version}}"
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
    python3 scripts/check_skill.py {{PATHS}}

# Regenerate marketplace.json skills[] from the catalogs on disk
gen-marketplace:
    python3 scripts/gen_marketplace.py

# Lint and check formatting of repository and skill scripts
lint:
    ruff check scripts skills/*/*/scripts skills/meta/*/assets skills/sdd/*/assets
    ruff format --check scripts skills/*/*/scripts skills/meta/*/assets skills/sdd/*/assets

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
    @command -v openspec >/dev/null || { echo "openspec CLI missing: run 'just setup' (installs @fission-ai/openspec@{{openspec_version}})" >&2; exit 1; }
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

# Run every check (validators, lint, spec validation, pre-commit hooks)
check: validate lint spec-validate tf-validate
    SKIP=commit-safety pre-commit run --all-files
