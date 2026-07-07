# .github/release.yml schema

Load condition: editing the notes configuration beyond the shipped
categories.

The file configures **automatically generated release notes** — the
`--generate-notes` flag, the REST generate-notes endpoint, and the web
UI's "Generate release notes" button. It affects nothing else.

## Structure

```yaml
changelog:
  exclude:
    labels: [...]      # PRs with any of these labels are omitted entirely
    authors: [...]     # PRs by these logins are omitted (bots: name[bot])
  categories:
    - title: Section heading
      labels: [...]    # PR labels that route into this section
      exclude:         # per-category exclude, same shape as the global one
        labels: [...]
        authors: [...]
```

## Matching rules

- A PR lands in the **first** category whose `labels` it matches;
  category order in the file is both match priority and render order.
- `labels: ["*"]` is the catch-all — anything not matched earlier. Keep
  it last; an early catch-all swallows every later category.
- Matching is by PR label only. Commit messages, PR titles, and file
  paths play no part.
- Every label named here must exist in the repository and actually be
  applied to PRs — automate that with the labeler workflows from
  `github-issue-conventions` / `github-pr-conventions` rather than
  relying on hand-labeling.

## Useful exclusions

- `exclude.authors: ["dependabot[bot]"]` keeps dependency-bump noise out
  of the notes (add the section back deliberately with a
  `dependencies`-labeled category when the project wants it).
- A global `exclude.labels: ["status/duplicate", "status/wontfix"]`
  drops closed-as-unwanted PRs that were merged anyway (rare, but they
  read badly in notes).

## Testing a change

The REST preview endpoint renders notes without creating anything:

```bash
gh api -X POST repos/O/R/releases/generate-notes \
  -f tag_name=NEXT_TAG -f previous_tag_name=PREV_TAG -q .body
```

Edit the config, re-run, and diff the output until the sections look
right; nothing is published by the preview.
