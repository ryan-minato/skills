# GitHub Checks

Read this before adding, renaming, removing, or debugging a GitHub Actions
job, or changing the ruleset's required checks.

The tiers below are the intended merge policy. Until the pending remote
settings update is explicitly authorized and read back, only `scan-secrets`
is enforced by the current ruleset.

| Check | Command or source | Events | Tier | Healthy result |
|---|---|---|---|---|
| `checks / quality` | `just check` | Pull requests and pushes to `main` | Required | Repository validation, script tests, lint, formatting, hooks, and secret checks all pass. |
| `pr / policy` | `python3 scripts/check_pr_policy.py --event "$GITHUB_EVENT_PATH"` | Pull-request lifecycle events | Required | Title, body, readiness evidence, and applicable commit subjects satisfy policy. |
| `scan-secrets` | `.github/workflows/secret.yml` | Pull requests and all pushes | Required | The existing TruffleHog action reports no verified secret. |
| `Analyze (python)` | CodeQL default setup | Platform-managed | Advisory | Analysis completes without an open blocking alert. |
| `Analyze (actions)` | CodeQL default setup | Platform-managed | Advisory | Analysis completes without an open blocking alert. |

Required-check names are stable interfaces. Update this file, the producing
workflow, `.agents/knowledge/github/platform-settings.md`, and the remote
ruleset together when a name changes.

The dev container and quality workflow install the latest available
`pre-commit`, `rust-just`, and `ruff` releases without version pins. Ruff's
pre-commit hook requires a revision, so keep that revision at the current Ruff
release when the hook is updated. `scripts/validate_harness.py` enforces these
two forms of the tool policy.

The secret workflow is an approved exception: it keeps the unpinned
third-party `trufflesecurity/trufflehog@main` reference and its all-branch push
trigger unchanged. Revisit the exception only by an explicit maintainer
decision.
