# Data

## Sources

| Name | Kind | Identity (revision / checksum / version id) | Location | How to fetch on a fresh machine |
|---|---|---|---|---|
| `<train>` | dataset | `<immutable identity>` | `data/raw/<name>/` | `<command>` |
| `<eval>` | evaluation set | `<immutable identity>` | `data/raw/<name>/` | `<command>` |

## Evaluation

The benchmark that judges "better": `<evaluation set identity>`, metrics
`<name: definition>`, computed by `eval.py` (the same `evaluate` the
training loop calls). Record here any threshold agreed before a run.

## Rules

- `data/raw/` is the local cache of immutable inputs; nothing transforms
  in place. Derived data lands under `data/interim/` and
  `data/processed/` and is regenerable.
- Nothing under `data/` is committed; a run's manifest records which
  identities it used.
