## Why

The `machine-learning` catalog (#85) now carries the durable skills for
ML experiment projects, but the builders that initialize such a project
still describe a different standard: `scaffold-ml` scaffolds two shapes
(a quick experiment on Pydantic Settings and requirements files, a
training project on Hydra) with no run provenance, no tracker, no
resolved-configuration dump, no multi-stage image, and no research-task
convention; `meta-python-defaults` would install a type-check gate over
tensor code and hooks that run tests; `meta-gpu-container` never records
the image digest as an environment identity; `meta-spec-workflow` knows no
research-task paradigm; the platform builders' ML references carry an
older run-record shape; and `meta-workflow-design`'s research profile
does not name the research task as the unit of work. A project built from
these builders today receives a harness that contradicts the skills it
will install tomorrow. Now, before the next ML project is scaffolded.

## What Changes

- `scaffold-ml` — **BREAKING**: one project shape replaces the quick
  experiment / maintainable training choice: OmegaConf typed schema plus
  YAML plus command-line overrides with a resolved dump saved per run, a
  dependency carrier the user chooses — a uv project (`pyproject.toml`
  plus `uv.lock` with index routing for accelerator wheels) by default,
  or `requirements.in` compiled by uv into a fully pinned
  `requirements.txt` with the torch backend flag for a scripts-only
  repository — an explicit Accelerate loop with one logging seam and one
  stage-trace seam, a run
  manifest written at start and finish, a tracker selected by precedence
  (keep existing → the hosting platform's experiment tracking → Trackio),
  a multi-stage container recipe whose pushed digest is the environment
  identity (containers stay opt-in), pytest `slow` and `gpu` markers with
  GPU tests that fail without hardware, Ruff at line 120 with docstring
  code formatting, hooks that never run tests, and an AGENTS.md deposit
  that points at the research-spec location, the tracker, the
  configuration directory, and — when installed — the durable
  machine-learning skills, named by role. Existing Hydra, Pydantic
  Settings, or requirements-file projects are preserved under the
  existing-choice rule (Hydra: no object instantiation, a pinned output
  directory). The Hydra reference and both asset trees are removed; the
  requirements-compile reference stays as the carrier's branch; the
  description no longer offers two modes.
- `meta-spec-workflow`: a research repository (the workflow contract's
  research profile) is offered the research-task protocol — OpenSpec
  with a project-local `research-task` schema whose spec carries an
  Objective and an Evaluation rather than requirements and scenarios;
  the gate covers objective and evaluation; the spec evolves while run
  history stays immutable; a task archives on its completion condition,
  negative results included; research tasks hold no domain. Ships the
  schema asset and a reference; the deposited contract gains a research
  section; the platform template lines gain a research variant. The
  description claims the case.
- `meta-python-defaults`: a tensor-code branch — when the project's core
  is training, fine-tuning, or inference on tensors, static type checking
  is never a gate over that code, docstrings carry shape, dtype, device,
  and mask conventions, GPU-only tests fail without hardware and run by
  explicit command, and git hooks run the formatter and linter only.
- `meta-gpu-container`: an image-identity section — a multi-stage build
  (environment → runtime → sealed with source), the pushed image digest
  recorded as the environment identity rather than the Dockerfile or a
  tag, and the host facts a container cannot pin recorded beside it; the
  deposit checklist gains the digest rule.
- `meta-github-workflow`, `meta-gitlab-workflow`: the ML references carry
  the canonical run-record field list (the one `experiment-provenance`
  publishes) and implement the research profile's decision that one
  research task maps to one pull or merge request carrying many
  hypotheses and runs, with the snapshot-reachability rule after a
  squash or a source-branch deletion and the tracker default for a new
  project (Trackio on GitHub, which has no tracker; GitLab's experiments
  on GitLab when the instance provides them). Descriptions unchanged.
- `meta-workflow-design`: the research profile names the research task —
  one objective and one evaluation — as the unit of tracked work, whose
  hypotheses and runs are not work items, and maps one task to one change
  request.

## Skills touched

- `scaffold/scaffold-ml` (new): description triggers, the single shape,
  the dependency carrier, the configuration surface, provenance and
  tracker, the image identity,
  tests and hooks, the deposited guidance, and the handoffs to the GPU
  container builder, the workflow and authority builders, the harness
  entry, and the durable machine-learning skills by role.
- `meta/meta-spec-workflow` (modified): the description (research
  repositories), the research-task approach, the deposited research
  section, the schema asset.
- `meta/meta-python-defaults` (new): description triggers and the
  tensor-code branch.
- `meta/meta-gpu-container` (new): description triggers and the image
  identity.
- `meta/meta-github-workflow` (modified): research tasks and pull
  requests; the run record.
- `meta/meta-gitlab-workflow` (modified): research tasks and merge
  requests; the run record.
- `meta/meta-workflow-design` (modified): the research task as the unit
  of tracked work.

## Installed behavior

- `scaffold-ml`: a project is scaffolded in one shape with provenance, a
  tracker, an image identity, the marker and hook rules, and pointers to
  the durable skills; the two-mode choice and Hydra by default are gone,
  and the dependency carrier is the user's explicit choice → `feat!`.
- `meta-spec-workflow`: a research repository gets a research-task
  contract and schema instead of a software-change one → `feat`.
- `meta-python-defaults`: a tensor-heavy project no longer receives a
  type-check gate over its model code or test-running hooks → `fix`
  (wrongly restrictive default) for the gate and hooks, `feat` for the
  docstring and GPU-test rules; correction wins → `fix`.
- `meta-gpu-container`: the deposit records the digest and the host
  facts → `feat`.
- `meta-github-workflow`, `meta-gitlab-workflow`: the run record matches
  the durable skill's canonical list and the research-task mapping is
  implemented → `fix` for the record fields (they drifted from the
  standard), `feat` for the mapping; correction wins → `fix`.
- `meta-workflow-design`: the research profile names its unit of work →
  `feat`.

## Impact

- README pair rows: the `scaffold` row for `scaffold-ml` (both
  languages) drops the two-mode wording; the `meta` rows for
  `meta-spec-workflow`, `meta-python-defaults`, and `meta-gpu-container`
  gain the new capabilities (both languages).
- `.claude-plugin/marketplace.json`: the `scaffold` and `meta` plugin
  descriptions if their wording moves (human-owned fields); the `skills[]`
  lists are unchanged (no directory added or removed).
- `skills/scaffold/CONTEXT.md`: one sentence that durable topic skills of
  another catalog are outside the range and are named by role with a
  deposited fallback — the companion repository change.
- `.agents/knowledge/harness-maintenance.md`: mirror rows for the run
  manifest module (`experiment-provenance/assets/run_manifest.py` ↔
  `scaffold-ml/assets/run_manifest.py`), the research-spec section
  headings (`research-workflow/assets/research-spec.md` ↔
  `meta-spec-workflow/assets/openspec/research-task/templates/research.md`
  ↔ the scaffold's AGENTS.md fallback skeleton), and the canonical
  run-record field list (`experiment-provenance/references/run-record.md`
  ↔ `meta-github-workflow/assets/experiment-record.md` ↔
  `meta-gitlab-workflow/references/mlops.md`) — the companion repository
  change.
- Mirrored files in `scripts/validate_harness.py`: none.
- Symlinks: unchanged.

## Non-goals

- Any change to the five `machine-learning` skills.
- A validator script for the scaffold (the data-science scaffold's
  `validate_scaffold.py` pattern); a later change if the scaffold earns
  one.
- Changing `scaffold-data-science`'s `data/<source>/` rule or reconciling
  it with the ML scaffold's `data/raw/`.
- Adding the research-task work item to the platform builders' semantic
  mappings beyond the reference sections named above.

## Tracked work

No issue: the follow-up planned in conversation when #85 was proposed.
