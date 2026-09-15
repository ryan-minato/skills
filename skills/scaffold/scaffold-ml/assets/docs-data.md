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
