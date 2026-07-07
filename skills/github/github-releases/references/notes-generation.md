# Generating release notes

Load condition: the notes decision table chose generated notes, or you
need to control what the generator includes.

## What `--generate-notes` does

GitHub builds the notes from **merged pull requests** between the previous
release and the new tag: one line per PR (title, author, PR link), a
first-time-contributors section, and a full-changelog compare link. It
keys on PRs, not commit messages — direct pushes without a PR do not
appear.

## Categories: `.github/release.yml`

When the repository has `.github/release.yml`, the generator groups PR
lines into its `changelog.categories` by PR **label** and hides anything
matched by `changelog.exclude`. Two consequences:

- The label taxonomy must actually be applied to PRs, or everything lands
  in the `*` catch-all category.
- Authoring or changing that file is `github-release-conventions` work —
  do not edit it as a side effect of cutting a release.

## Controlling the range

The generator diffs from the previous release tag by default. When
releasing from a maintenance branch or after deleting releases, pin the
start explicitly:

```bash
gh release create TAG -R O/R --draft --generate-notes \
  --notes-start-tag PREV_TAG
```

## Preview without creating anything

The REST preview endpoint returns the generated name and body as JSON —
useful to curate into `NOTES.md` before any draft exists:

```bash
gh api -X POST repos/O/R/releases/generate-notes \
  -f tag_name=TAG [-f previous_tag_name=PREV] -q .body > NOTES.md
```

The tag does not need to exist yet; `tag_name` is what the notes will be
generated *as of* (uses the default branch tip when the tag is absent).

## Mixing generated and hand-written content

Generate first (preview endpoint above), then edit `NOTES.md`: add a
summary paragraph on top, trim noise, keep the PR list. The pre-publish
gate reviews the final file — generated text gets no exemption, since PR
titles can carry anything their authors wrote.
