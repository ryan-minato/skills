# <project name>

<Two or three sentences: what gets trained and evaluated here, and what
the model is for.>

## Directory map

| Path | Holds | Rule |
|---|---|---|
| `train.py`, `eval.py` | workflow entries | run via just |
| `<project_name>/` | shared training/eval code | absolute imports only |
| `configs/` | Hydra tree (`model/`, `data/`, `optim/`) | values, never class paths |
| `data/raw/` | original inputs | append-only; never edited |
| `data/interim/` | intermediate transforms | regenerable at will |
| `data/processed/` | training-ready data | regenerable at will |
| `outputs/` | runs, checkpoints, exports | never committed |
| `docs/` | the knowledge base | updated with the work, see below |

## Commands

| Command | Does |
|---|---|
| `just setup` | `uv sync` + install hooks |
| `just train` / `just eval` | run a workflow |
| `just lint` | format + lint (Ruff) |
| `just test` | fast tests only |
| `just test-slow` | GPU/long tests — manual only, never hooks or CI |

## Environment

- uv owns dependencies; `uv.lock` is committed. torch installs from the
  `<backend>` wheel index configured in `pyproject.toml` — when hardware
  changes, revisit that routing before anything else.
- <how to get the data onto a fresh machine>

## Configuration

- Hydra; `config_path` resolves relative to the entry script, and run
  outputs are pinned to `outputs/`. Expose what experiments vary; keep
  constants in code.

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

## When to read what

| Situation | Read |
|---|---|
| touching data loading or schemas | `docs/data.md` |
| starting or reviewing a training run | `docs/experiments.md` |
| wondering why something is built this way | `docs/decisions.md` |

Update the matching file in the same change: every finished run appends
to `docs/experiments.md` (config, checkpoint, result), every reversed or
new decision lands in `docs/decisions.md`.

## Safety limits

- Never edit `data/raw/`; never commit `data/` or `outputs/`.
- Never wire slow tests into hooks or CI.
- <anything else agents must not do without asking>
