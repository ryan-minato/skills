# <project name>

<Two or three sentences: the idea this experiment tests, and the signal
that will count as success or failure.>

## Files

- `train.py` / `eval.py` — workflow entries; run them via just, never ad hoc
- `config.yaml` — the experiment's knobs (loaded by a Pydantic Settings class)
- `<project_name>/` — shared code both entries import
- `data/` — inputs (read-only); `outputs/` — everything produced

## Commands

| Command | Does |
|---|---|
| `just setup` | create the environment from the committed lockfiles |
| `just train` | run training |
| `just eval` | run evaluation |
| `just lock` | recompile requirements after editing an `.in` file |
| `just test` | fast tests only |
| `just test-slow` | GPU/long tests — manual only, never hooks or CI |

## Data rules

- Inputs under `data/` are never edited; every artifact lands in `outputs/`.
- <where the data comes from, and how to fetch it on a fresh machine>

## Reproducibility

- The committed `requirements.txt` / `requirements.dev.txt` ARE the
  environment. Change dependencies only via the `.in` files + `just lock`,
  and commit all four files together.
- torch wheel variant: `<backend>` on <machine>; `<backend>` on <machine>.
- Seed and config are committed with any result worth keeping.

## Code style

- Readability first; the only product is a trained model. Extract shared
  code only when two places must stay logically consistent — repetition
  alone is fine.
- Let it crash: only known-dirty data is caught (counted and skipped);
  implementation errors fail fast. No defensive try/except.
- Ruff is the linter and formatter. No type checker. Docstrings on shared
  interfaces (args, returns, errors); one-liners on private helpers;
  comments only on long logic passages.
- Tests protect custom components' contracts, nothing else; expensive
  tests carry `@pytest.mark.slow` and run only by hand.
- Commits: short, imperative, one change each.
