## Why

Skills that build or deposit a project's harness hand the target copies of their own runtime scripts. `openspec-workflow` and `spec-kit-workflow` require a byte-identical copy of their bundled script in the target's CI. `meta-github-workflow` and `meta-gitlab-workflow` copy the scripts they run themselves into the target and its durable project skill. A copied product script carries the defensive style a script needs when its runner is not its author: a failed lookup read as "empty" makes a gate fail open, and the project cannot fix a script it is bound to keep identical. No skill says what management code the target should get instead. Issue #94 settles the principles; this change makes the skills follow them.

## What Changes

The principles, called R1–R6 below, apply to **management code**: code that never ships in the product (quality checks, environment preparation, CI and administration scripts, git hooks, inline workflow shell, scripts inside a project skill).

- R1 decouple: a skill's bundled script and a management script need not be identical, and no rule, check, or instruction keeps them identical. A skill ships an asset written for the management role — its content may still match the bundled script where the role happens to need the same code — and the target owns its copy.
- R2 language: the project's language when its community widely and maturely scripts in it, otherwise Python or Deno, Bash for simple operations in a controlled Linux environment.
- R3 dependencies: self-contained where the ecosystem has no widespread convention; the standard library only when it solves the task cleanly, never a reimplementation of a mature library.
- R4 invocation: `uv run` only where uv exists in every environment the script runs in.
- R5 errors: readability first and let unexpected failures crash. Every interface with an external tool, command, API, or a file another step wrote is checked for what the code relies on and fails at once with a message naming it. Fallbacks that turn a failure into a wrong answer go; structural safety, idempotence, and narrow retries stay.
- R6 shell: `set -euo pipefail` with local exceptions, `defaults.run.shell: bash` in generated GitHub workflows, and a command whose failure matters runs before a condition, not inside it.

Per skill:

- `meta-harness`: the `## Harness Methodology` section gains one principle: management code belongs to the project, is readable, fails fast, and is never bound to a skill's runtime script. Applying it, the agent writes a project-owned script for the role instead of binding the project to a skill's script, and reports such a binding, not matching content, as a finding in an audit.
- `meta-harness-architecture`: the same methodology line (the section is a validated byte-identical mirror). The builder writes or deposits management code by R1–R6 and records the chosen language and invocation in the target's harness knowledge. A new reference carries the details: language judgment with examples, dependencies, invocation, the interface checks and the fallbacks to remove, documented nulls, and the shell rules.
- `meta-python-defaults`: the Python realization. PEP 723 inline metadata, `uv run` only where uv exists everywhere the script runs, the standard-library path otherwise, and an error-handling idiom of interface checks that exit with a message and no blind, silent, or restating handlers. Locking a CI script's dependencies (`uv lock --script` or an `exclude-newer` cutoff) is presented as a recommendation. Its default follows the job's actual risk, and the user decides.
- `meta-github-workflow`:
  - every script it gives the target comes from a management asset written for the target's role, not from the scripts the builder runs; nothing keeps the two identical, and an asset may still match the builder's script. It ships new `sync_labels.py` and `run_log_digest.py` assets. `next_version.py` and `project_fields.py`, which the builder never runs, move from `scripts/` to `assets/`.
  - `check_commits.py` and `check_taxonomy.py` are rewritten by R5. The taxonomy check declares PyYAML inline and runs through uv, with the builder recommending uv where the target lacks it and presenting locking as the user's decision.
  - delivered workflows set `defaults.run.shell: bash`, and the aggregator gate no longer passes when its output pipe breaks.
- `meta-gitlab-workflow`: the same for its deliveries: a new `pipeline_log_digest.py` asset, `next_version.py` moved to `assets/`, the commit check rewritten, and job scripts that fail on the failing command.
- `openspec-workflow`:
  - the automation deposits a management script, `assets/spec_request.py`, as the project's own `scripts/spec_request.py`, in place of a byte-identical copy of the bundled `scripts/spec_changes.py`. It holds only what the workflows call: `snapshot`, `check`, `show`, `status`, `labels`.
  - an unknown change name in a comment command is answered with the related changes, every name rendered as code.
  - the comment workflow posts the reply and fails the run on a crash, passing only on a bad argument.
  - delivered workflows run under `bash` with pipefail. The bundled script and its local commands stay as they are.
- `spec-kit-workflow`: the same, with `assets/spec_kit_request.py` in place of a copy of `scripts/spec_kit_features.py`.
- `scaffold-ml`: the sealed image build runs the git reads it depends on before testing their result, so an unreadable tree stops the build instead of passing the dirty-tree guard.

## Skills touched

- `core/meta-harness` (new): the management-code principle in the methodology.
- `meta/meta-harness-architecture` (new): management code written for its role, its language and dependencies, fail-fast error handling, and shell.
- `meta/meta-python-defaults` (new): Python management-script conventions and the dependency-locking recommendation.
- `meta/meta-github-workflow` (modified): scripts delivered from management assets; delivered workflows and the taxonomy check follow the management-code rules.
- `meta/meta-gitlab-workflow` (modified): scripts delivered from management assets; job scripts fail on the failing command.
- `sdd/openspec-workflow` (modified): the automation deposits a management script; the management script's contract.
- `sdd/spec-kit-workflow` (modified): the same.
- `scaffold/scaffold-ml` (modified): the container recipe's sealed build.

## Installed behavior

- `meta-harness`, `meta-harness-architecture`, `meta-python-defaults`: an agent gains rules it did not have for the management code it writes → `feat`.
- `meta-github-workflow`, `meta-gitlab-workflow`, `openspec-workflow`, `spec-kit-workflow`: an agent no longer copies a runtime script into the target, and no longer delivers a gate or a command workflow that passes when its command failed → `fix`.
- `scaffold-ml`: the delivered sealed-build recipe no longer builds when git cannot read the tree → `fix`.

## Impact

- `skills/sdd/README.md` and `README.zh.md`: the `openspec-workflow` and `spec-kit-workflow` rows name the deposited management script beside the bundled one.
- `skills/meta/meta-spec-workflow`: the deposited contract's sentence that the framework skill "owns the request automation and its script" becomes "supplies"; the project owns the deposited script. The wording changes and the observable behavior does not, so the skill gets no delta spec.
- `ruff.toml` target versions for the moved and new assets, and the lint scope that covers them, belong to the companion repository change `management-code-harness`. That change also carries:
  - this repository's own rewrite of `scripts/spec_changes.py` and `scripts/sync_labels.py`;
  - the removal of the byte-identical copies check;
  - the management-code rules in the knowledge base and the review skill;
  - the `meta` catalog rule that script assets are working code.
- No symlink, `marketplace.json` entry, or skill description changes.

## Non-goals

- Public skills' own `scripts/`, including the `snapshot` and `labels` subcommands of the sdd bundled scripts that the delivered workflows no longer call. They keep the existing script rules.
- `engineering/devcontainer-authoring`, whose Features and Templates scaffolds carry workflows and test shell: a follow-up.
- `scaffold-data-science` and `scaffold-colab`. Their script assets are product code (`local-data-guard.py` runs inside the pipeline and its manifest ships with the product), and their task-runner recipes already conform.
- Porting assets to a target's own language. Assets ship in Python (standard library first, 3.10 or later), and the target may port its copy.
- Enforcing R5 with lint rules in `meta-python-defaults`.

## Tracked work

Issue #94.
