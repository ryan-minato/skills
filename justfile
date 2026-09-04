# Canonical check recipes for this repository.
# Agents and humans should run checks through these recipes (not ad-hoc
# commands) so results stay consistent across environments.

# Pinned tool versions. These variables are the single source for the pins;
# scripts/validate_harness.py checks that generated files and CI agree.
openspec_version := "1.12.0"

# List available recipes
default:
    @just --list

# One-time environment setup (run after cloning / container creation)
setup: install-tools
    pre-commit install
    git config commit.template .gitmessage

# Install the pinned tools `just check` needs beyond the dev container features
install-tools:
    npm install -g "@fission-ai/openspec@{{openspec_version}}"

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
    ruff check scripts skills/*/*/scripts
    ruff format --check scripts skills/*/*/scripts

# Validate every OpenSpec spec and change (strict)
spec-validate:
    @command -v openspec >/dev/null || { echo "openspec CLI missing: run 'just setup' (installs @fission-ai/openspec@{{openspec_version}})" >&2; exit 1; }
    OPENSPEC_NO_UPDATE_CHECK=1 openspec validate --all --strict --no-interactive

# Regenerate the OpenSpec-managed skills after bumping openspec_version; commit the result
spec-sync:
    OPENSPEC_NO_UPDATE_CHECK=1 openspec update --force

# Archive every OpenSpec change whose tasks are all complete (what the spec-archive workflow runs after a merge)
spec-archive-completed *ARGS:
    python3 scripts/archive_completed_changes.py {{ARGS}}

# Safety gate for staged changes (also runs as the first pre-commit hook)
commit-gate:
    python3 scripts/check_commit_safety.py

# Run every check (validators, lint, spec validation, pre-commit hooks)
check: validate lint spec-validate
    SKIP=commit-safety pre-commit run --all-files
