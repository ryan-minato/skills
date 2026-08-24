# __PROJECT_NAME__

__PROJECT_PURPOSE__

## Directory map

| Path | Holds | Rule |
|---|---|---|
| `src/__PACKAGE_NAME__/` | production analysis code | reusable logic lives here |
| `src/__PACKAGE_NAME__/workflows/` | thin workflow entries | run through Just |
| `src/__PACKAGE_NAME__/sources/` | source acquisition logic | only download workflows write original data |
| `notebooks/` | one-off exploration and visualization | import `src`; never own production logic |
| `config/project.toml` | adjustable non-secret configuration | validated by Pydantic Settings |
| `data/<source>/` | local original inputs, when used | immutable; no `raw/` layer |
| `output/` | local derived products, when used | never an input source |
| `report/` | paired Markdown or PDF analysis | update with product changes |
| `.agents/knowledge/` | project facts for agents | keep the when-to-read table current |

Delete rows for absent optional paths.

## Commands

| Command | Does |
|---|---|
| `just setup` | sync dependencies and install hooks |
| `just download-data` | acquire configured source versions |
| `just pipeline` | run production steps |
| `just test` | run fast, focused tests |
| `just check` | Ruff checks plus fast tests |
| `just report` | build or validate the report |
| `just safe-to-commit` | checks, automated secret scan, staged-diff sanity |
| `just safe-to-push` | complete checks and history scan |

## Standing rules

- uv owns dependencies; commit `uv.lock`.
- Production logic lives in `src/`; notebooks contain disposable exploration.
- Pydantic Settings loads `config/project.toml` plus environment overrides.
  Secrets live only in ignored `.env`; `.env.example` contains safe examples.
- Loguru logs run, workflow, step, identities, counts, timing, and failures.
  Never log credentials, PII, environments, or full original records.
- Ruff is the linter and formatter. There is no type checker.
- Tests protect custom, reusable, error-prone logic. There is no coverage
  target and no tests for notebooks, trivial glue, or third-party behavior.
- Analysis code may use models for inference; training belongs elsewhere.

## Data and products

- Local `data/` contains original inputs only under `data/<source>/`.
- Never create `data/raw/`; never put derived, cached, temporary, or final data
  under `data/`.
- Only download workflows may publish a new local source path. They verify
  identity, publish atomically, and refuse overwrite.
- Every product records source identities, resolved non-secret configuration,
  Git commit, lockfile digest, model revision, seeds, step state, and timing.

## Git safety

- Make small, reversible, atomic commits at meaningful checkpoints.
- Before every commit, run `just safe-to-commit`. Review a small staged diff
  directly; for a larger diff, use programmatic secret/PII scanning and record
  its result. Check credentials, PII, private contact or financial data,
  restricted source samples, and local paths.
- Treat sensitive content committed to history as leaked: stop and report it.
  A later deletion does not erase history.
- Run `just safe-to-push` before pushing. Never bypass hooks.
- Try uncertain ideas in disposable worktrees, one branch per idea. Compare
  them with the same checks, remove the worktrees and discarded branches, then
  implement the selected approach on the formal branch in the canonical
  worktree.
- Prefer revert or a new corrective commit over destructive broad resets.

## When to read what

| Situation | Read |
|---|---|
| changing package boundaries, workflows, or data flow | `ARCHITECTURE.md` |
| deciding scope, success, or non-goals | `.agents/knowledge/PROJECT.md` |
| touching a source, schema, identity, or product location | `.agents/knowledge/DATA.md` |

Update the matching document in the same change. Every link and table row must
describe the repository that exists, not the scaffold that once created it.
