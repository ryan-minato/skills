# Marker-Comment Tags

Read when the target codebase already uses marker tags beyond TODO, FIXME,
and NOTE, or the user asks for a richer tag vocabulary.

## Format

One convention regardless of vocabulary:

```python
# TAG(owner-or-issue): imperative description
# TODO(#482): collapse this branch once the v1 API is removed
```

The parenthesized owner or issue reference makes every marker traceable;
a marker nobody owns is a wish, not a work item.

## Tag Meanings

| Tag | Means |
|---|---|
| TODO | Planned improvement; the code works but should evolve. |
| FIXME | Known defect or debt; the code works but is wrong in a way that matters. |
| NOTE | A non-obvious fact the reader needs before touching this code. |
| HACK | Deliberate ugliness taken for a stated reason; do not clean up without understanding it. |
| XXX | Danger / needs attention; a legacy catch-all — prefer FIXME or WARNING. |
| WARNING | Do not modify without reading this; breakage is non-local. |
| PERF | A non-obvious choice made for measured performance reasons. |
| SAFETY | The invariant that justifies unsafe-looking code. |
| BUG | A known, reproducible defect left in place, ideally with an issue link. |
| DEPRECATED | Kept only for compatibility; do not add new callers. |

## The Closed-Set Rule

The default vocabulary is **TODO, FIXME, NOTE** — small enough that every
tag stays searchable and unambiguous. Whatever vocabulary the project
picks, record it as a closed set: a tag outside the set is a typo, not a
new convention. If the project wants enforcement, most linters offer a
marker-comment rule family — name the category in the harness and fetch
the current rule syntax from the chosen linter's docs.
