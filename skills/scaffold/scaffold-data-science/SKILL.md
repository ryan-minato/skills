---
name: scaffold-data-science
description: >-
  Disposable builder skill (delete after the harness is built): scaffolds an
  opinionated, reproducible Python data-science project and its agent harness.
  Use when creating or hardening an empty or early repository that ingests
  datasets; validates, cleans, transforms, joins, or aggregates data; performs
  reproducible exploratory or production analysis; or publishes data products
  across local, S3, or Hugging Face storage. Not for notebook-only exploration,
  mature migrations, or model training; preserves working choices.
compatibility: Requires Python 3.11+ for the bundled validator; uv is the empty-project default.
---

# Reproducible Data-Science Project

Build a Python data-science repository whose inputs are immutable, whose
production work runs from `src/`, and whose lockfile, configuration, source
identities, model revisions, and provenance make every product reproducible.
Keep existing working choices in an early repository; this is a scaffold, not
a migration mandate.

## Workflow

1. Inventory the project before writing: goal, package name, sources,
   products, input and output backends, data scale and media, report format,
   model use, and any working tools already present.
2. Initialize an absent package with `uv init --package` and commit `uv.lock`.
   In an existing project, retain its working package manager, lockfile, test
   tools, and quality gates unless the user requests migration. Put reusable logic in
   `src/<package>/`, thin launch modules in `src/<package>/workflows/`,
   source acquisition code in `src/<package>/sources/`, exploratory work in
   `notebooks/`, focused tests in `tests/`, and the paired report in
   `report/`.
3. Copy and rework every base asset; replace every `__UPPER_CASE__`
   placeholder and delete inapplicable rows:
   - [agents-md.md](assets/base/agents-md.md) to `AGENTS.md`
   - [architecture-md.md](assets/base/architecture-md.md) to
     `ARCHITECTURE.md`
   - [knowledge-project.md](assets/base/knowledge-project.md) to
     `.agents/knowledge/PROJECT.md`
   - [knowledge-data.md](assets/base/knowledge-data.md) to
     `.agents/knowledge/DATA.md`
   - [justfile.md](assets/base/justfile.md) to `justfile` only when the project
     has no settled task interface
   - merge only missing, compatible defaults from
     [pyproject-tool-config.md](assets/base/pyproject-tool-config.md) into
     `pyproject.toml`
   - [settings.py](assets/base/settings.py) to `src/<package>/settings.py`
     only when the project lacks a working configuration system
   - use [workflow-entry.py](assets/base/workflow-entry.py) for each real
     workflow entry
   - [project.toml](assets/base/project.toml) to `config/project.toml`
   - [env-example](assets/base/env-example) to `.env.example`
   - [editorconfig](assets/base/editorconfig) to `.editorconfig` only when absent
   - merge [gitignore](assets/base/gitignore) into `.gitignore`
   - [report.md](assets/base/report.md) to `report/report.md`
4. Select branches per source and product; mixed backends are valid:
   - Read [storage-local.md](references/storage-local.md) when any source or
     product is local.
   - Read [storage-s3.md](references/storage-s3.md) when any source or
     product uses S3.
   - Read
     [storage-huggingface.md](references/storage-huggingface.md) when any
     source, product, or model uses Hugging Face Hub.
   - Read
     [compute-structured.md](references/compute-structured.md) for
     structured or tabular data.
   - Read
     [compute-multimedia.md](references/compute-multimedia.md) for image,
     audio, video, or other multimedia data.
   - Read [model-inference.md](references/model-inference.md) when the
     pipeline loads a model.
   - Read
     [model-reimplementation.md](references/model-reimplementation.md)
     only when a model comes from an unstable experimental repository.
5. Resolve documentation from official first-party sources when a selected
   branch needs it. Do not create or maintain a documentation-URL index.
6. Model configuration per source and product, not with one global storage
   switch. Keep adjustable non-secrets in `config/project.toml`; preserve an
   existing configuration system or default to Pydantic Settings for TOML plus
   environment overrides. Put credentials only in ignored `.env`; track only
   safe names and examples in `.env.example`.
7. Make every workflow observable. Preserve working logging infrastructure or
   default to Loguru. Bind run, workflow, and step context; log identities,
   counts, timing, and failure state at useful levels. Never log credentials,
   full environments, PII, or raw records.
8. If quality tooling is unsettled, default to Ruff for lint and format,
   pytest for fast targeted tests, and Gitleaks for automated secret detection.
   Generate a pinned `.pre-commit-config.yaml` only when pre-commit is the
   selected local gate. Preserve working type checkers, coverage policies,
   task runners, hooks, and equivalent tools.
9. Test only custom reusable logic whose mistakes would corrupt a result:
   transformations, invariants, boundary/error cases, and reimplemented
   model behavior. Do not test notebooks, declarative configuration,
   third-party libraries, or trivial workflow glue. Keep large-data, GPU,
   and model-equivalence tests manual under `slow`.
10. Ask how the team handles commits, review, sensitivity scanning, pushes,
    and experimental worktrees; record the agreed rules in `AGENTS.md` without
    inventing workflow automation. Work tracking, planning, and
    agent-autonomy rules are designed with the `meta-workflow-design` and
    `meta-agent-authority` skills, not improvised here; if they are not
    installed, load the `ryan-minato-skills-installing` skill and install
    the whole `meta` catalog at project scope as it directs (its builders
    stack and are disposed together); never run an install command
    yourself. If the user declines, leave management design out and record
    the gap in
    the handoff.
11. Run
    [validate_scaffold.py](scripts/validate_scaffold.py) with
    `python3 scripts/validate_scaffold.py --project-root <target>`. Fix every
    issue, run the target's `just check`, inspect the generated harness with
    the user, and repeat until clean.

Done when: all selected storage, compute, and model branches are represented
without unused branch material; local source data cannot be overwritten by
the pipeline; every product carries provenance; all placeholders and links
are resolved; fast checks pass; and the user approves the scaffold.

## Invariants

- Local `data/` contains original inputs only, directly under
  `data/<source>/`.
- Only download workflows may write local source data. They download to a
  temporary sibling, verify identity, atomically publish a new path, and
  refuse overwrite. New upstream versions get new paths.
- Production steps read sources or earlier products and write only their
  own product locations. Local final products live under `output/final/`;
  local provenance lives under `output/_provenance/`.
- A run records source identities, resolved non-secret configuration, Git
    commit, selected lockfile digest, model revision, random seeds, step state, and
  timing.
- Data-science projects may consume models; training, finetuning, optimizers,
  training loaders, and checkpoint management belong in another project.
- DVC, MLflow, LakeFS, containers, CI, and external orchestrators are
  opt-in, never scaffold defaults.

## Gotchas

- A branch name such as `main` or `latest` is not an immutable data or model
  identity. Record a revision, version ID, ETag plus checksum, or equivalent.
- A passing secret scanner does not prove a diff is free of PII. Review small
  staged diffs directly; route larger diffs through an appropriate programmatic
  scan and record its result. If sensitive content reached history, stop and
  report it rather than hiding it in a later commit.
- Do not copy a template unchanged. A remaining placeholder is an unmade
  project decision.
